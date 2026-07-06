"""v23 = full leader-recipe build, our implementation.

Two pieces:
1. SeqNet — hierarchical action-order transformer with SANITIZATION-ROBUST tokens
   (fixes v22's flaws): per-action (action_type, street, hero-relative ROLE — not
   raw seat, validator 16-grid amount-BUCKET index, pot-flow id) + 3 bounded
   continuous (log1p amount_bb / pot_after_bb / pot_delta_bb). Hand encoder =
   embeds+proj+POSITION-embed -> 2-layer transformer (order-aware) -> attention
   pool (learned query). Chunk encoder = 1-layer transformer over hand vectors,
   NO positional (permutation-invariant) -> attention pool -> MLP head.
2. V23Model — stacked ensemble: 5 tree base learners (v10 estimators on 293
   base feats) + SeqNet on raw chunks -> logistic meta (fit on OOF) -> isotonic
   -> topk serving head. Benchmark-only / honest.
"""
from __future__ import annotations
import math, os
from typing import Any, Dict, List, Sequence
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = object  # type: ignore

from poker44_ml.features import chunk_features as _base_cf
from poker44_bump.model_v5 import _topk_squeeze

# ---- token scheme (mirrors the live payload canonicalization) ----
_ACT = {"check": 1, "call": 2, "bet": 3, "raise": 4, "fold": 5}
_STREET = {"preflop": 1, "flop": 2, "turn": 3, "river": 4, "": 5}
_BUCKETS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0)
N_ACT, N_STREET, N_ROLE, N_BKT, N_FLOW = 6, 6, 3, len(_BUCKETS) + 1, 5
MAX_A, MAX_H, CONT = 12, 64, 3


def _act_id(v) -> int:
    r = str(v or "").strip().lower()
    if r in _ACT: return _ACT[r]
    for k in ("raise", "bet", "call", "check"):
        if k in r: return _ACT[k]
    if "fold" in r or r == "muck": return _ACT["fold"]
    return _ACT["check"]


def _bkt_id(bb: float) -> int:
    v = max(0.0, float(bb))
    if v <= 0.0: return 1
    return int(np.argmin([abs(b - v) for b in _BUCKETS])) + 1


def _flow_id(pb: float, pa: float) -> int:
    d = max(0.0, pa - pb)
    if d <= 1e-6: return 1
    if d <= 1.0: return 2
    if d <= 4.0: return 3
    return 4


def _spread_idx(total: int, limit: int) -> List[int]:
    if total <= limit: return list(range(total))
    last = total - 1
    idx = {int(round(i * last / (limit - 1))) for i in range(limit)}
    j = 0
    while len(idx) < limit:
        idx.add(j); j += 1
    return sorted(idx)[:limit]


def encode_chunk_v23(chunk: List[dict]):
    hands = [chunk[i] for i in _spread_idx(len(chunk or []), MAX_H)]
    H = max(1, len(hands))
    ids = np.zeros((H, MAX_A, 5), dtype=np.int64)      # act, street, role, bucket, flow
    cont = np.zeros((H, MAX_A, CONT), dtype=np.float32)
    am = np.zeros((H, MAX_A), dtype=np.float32)
    hm = np.zeros((H,), dtype=np.float32)
    for hi, h in enumerate(hands):
        h = h or {}
        md = h.get("metadata") or {}
        hero = int(md.get("hero_seat") or 0)
        bb = float(md.get("bb") or 0.02) or 0.02
        acts = (h.get("actions") or [])[:MAX_A]
        if acts: hm[hi] = 1.0
        for ai, a in enumerate(acts):
            a = a or {}
            amt = float(a.get("normalized_amount_bb") or 0.0)
            if amt == 0.0: amt = float(a.get("amount") or 0.0) / bb
            pb = float(a.get("pot_before") or 0.0) / bb
            pa = float(a.get("pot_after") or 0.0) / bb
            seat = int(a.get("actor_seat") or 0)
            ids[hi, ai, 0] = _act_id(a.get("action_type"))
            ids[hi, ai, 1] = _STREET.get(str(a.get("street") or "").strip().lower(), _STREET[""])
            ids[hi, ai, 2] = 1 if (hero and seat == hero) else 2
            ids[hi, ai, 3] = _bkt_id(amt)
            ids[hi, ai, 4] = _flow_id(pb, pa)
            cont[hi, ai, 0] = math.log1p(max(amt, 0.0))
            cont[hi, ai, 1] = math.log1p(max(pa, 0.0))
            cont[hi, ai, 2] = math.log1p(max(pa - pb, 0.0))
            am[hi, ai] = 1.0
    return ids, cont, am, hm


class _AttnPool(nn.Module):
    def __init__(self, d, heads, p):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.mha = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)

    def forward(self, x, pad_mask):  # x [B,L,d], pad_mask True=pad
        B = x.shape[0]
        allpad = pad_mask.all(dim=1)
        pm = pad_mask.clone(); pm[allpad, 0] = False
        out, _ = self.mha(self.q.expand(B, 1, -1), x, x, key_padding_mask=pm)
        return out.squeeze(1)


class SeqNet(nn.Module):
    def __init__(self, d=64, heads=4, act_layers=2, hand_layers=1, p=0.1, ff=2):
        super().__init__()
        self.e_act = nn.Embedding(N_ACT, 16, padding_idx=0)
        self.e_street = nn.Embedding(N_STREET, 8, padding_idx=0)
        self.e_role = nn.Embedding(N_ROLE, 4, padding_idx=0)
        self.e_bkt = nn.Embedding(N_BKT, 16, padding_idx=0)
        self.e_flow = nn.Embedding(N_FLOW, 8, padding_idx=0)
        self.in_proj = nn.Linear(16 + 8 + 4 + 16 + 8 + CONT, d)
        self.pos = nn.Embedding(MAX_A, d)          # action ORDER (hand encoder only)
        mk = lambda n: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, ff * d, dropout=p, batch_first=True), n)
        self.hand_enc = mk(act_layers)
        self.hand_pool = _AttnPool(d, heads, p)
        self.chunk_enc = mk(hand_layers)            # no positional = perm-invariant
        self.chunk_pool = _AttnPool(d, heads, p)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, 1))

    def forward(self, ids, cont, amask, hmask):
        B, H, A, _ = ids.shape
        f = torch.cat([self.e_act(ids[..., 0]), self.e_street(ids[..., 1]),
                       self.e_role(ids[..., 2]), self.e_bkt(ids[..., 3]),
                       self.e_flow(ids[..., 4]), cont], dim=-1)
        x = self.in_proj(f).view(B * H, A, -1)
        x = x + self.pos(torch.arange(A, device=x.device)).unsqueeze(0)
        apad = (amask.view(B * H, A) < 0.5)
        allpad = apad.all(dim=1); ap = apad.clone(); ap[allpad, 0] = False
        x = self.hand_enc(x, src_key_padding_mask=ap)
        hv = self.hand_pool(x, apad).view(B, H, -1)
        hpad = (hmask < 0.5)
        allh = hpad.all(dim=1); hp = hpad.clone(); hp[allh, 0] = False
        z = self.chunk_enc(hv, src_key_padding_mask=hp)
        cv = self.chunk_pool(z, hpad)
        return self.head(cv).squeeze(-1)


def batch_chunks(chunks: Sequence[List[dict]]):
    encs = [encode_chunk_v23(list(c or [])) for c in chunks]
    Hm = max(e[0].shape[0] for e in encs); B = len(encs)
    ids = np.zeros((B, Hm, MAX_A, 5), np.int64); cont = np.zeros((B, Hm, MAX_A, CONT), np.float32)
    am = np.zeros((B, Hm, MAX_A), np.float32); hm = np.zeros((B, Hm), np.float32)
    for i, (a, c, m, h) in enumerate(encs):
        n = a.shape[0]; ids[i, :n] = a; cont[i, :n] = c; am[i, :n] = m; hm[i, :n] = h
    return (torch.from_numpy(ids), torch.from_numpy(cont),
            torch.from_numpy(am), torch.from_numpy(hm))


class V23Model:
    """Stack: 5 trees (base feats) + SeqNet (raw chunks) -> logistic meta -> isotonic -> topk."""

    def __init__(self, estimators, feature_names, seq_states, seq_cfg, meta_model,
                 isotonic=None, topk_cfg=None, metadata=None):
        self.estimators = list(estimators)
        self.feature_names = list(feature_names)
        self.seq_states = list(seq_states) if isinstance(seq_states, (list, tuple)) else [seq_states]
        self.seq_cfg = dict(seq_cfg)
        self.meta_model = meta_model
        self.isotonic = isotonic
        self.topk_cfg = dict(topk_cfg or {"positive_fraction": 0.15})
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("model_version", "v23-leader-recipe-stack")
        self.metadata.setdefault("model_name", "poker44-bump-v23")
        self.metadata.setdefault("framework",
                                 "stack(5 trees + action-order seq-transformer) + logistic-meta + isotonic + topk")
        self.metadata.setdefault("conformal_threshold", 0.5)
        self.metadata.setdefault("data_attestation",
                                 "No validator-private data used; released benchmark labels only.")
        self.metadata["topk_cfg"] = self.topk_cfg
        self.threshold = 0.5
        self.head_mode = "topk"
        self.subsample = False
        self._nets = None

    def _seq(self):
        if self._nets is None:
            nets = []
            for st in self.seq_states:
                net = SeqNet(**self.seq_cfg); net.load_state_dict(st); net.eval()
                nets.append(net)
            self._nets = nets
        return self._nets

    @property
    def has_seq(self) -> bool:
        return bool(self.seq_states)

    def _rows(self, chunks):
        rows = []
        for c in chunks:
            c = list(c or [])
            bf = _base_cf(c) if c else {"hand_count": 0.0}
            bf["hand_count"] = float(len(c))
            rows.append([float(bf.get(n, 0.0)) for n in self.feature_names])
        return np.asarray(rows, dtype=np.float64)

    def _base_matrix(self, chunks) -> np.ndarray:
        X = self._rows(chunks)
        cols = [np.clip(e.predict_proba(X)[:, 1], 0, 1) for e in self.estimators]
        if self.has_seq:                     # seq column optional (v24b+ drop it)
            nets = self._seq()
            seq_p = []
            with torch.no_grad():
                for s in range(0, len(chunks), 64):
                    t = batch_chunks(chunks[s:s + 64])
                    probs = [torch.sigmoid(net(*t)).numpy() for net in nets]
                    seq_p.append(np.mean(probs, axis=0))
            cols.append(np.clip(np.concatenate(seq_p), 0, 1))
        return np.stack(cols, axis=1)

    def predict_raw(self, chunks) -> np.ndarray:
        chunks = [list(c or []) for c in chunks]
        if not chunks:
            return np.zeros((0,), dtype=np.float64)
        Z = self._base_matrix(chunks)
        p = self.meta_model.predict_proba(Z)[:, 1]
        if self.isotonic is not None:
            p = np.asarray(self.isotonic.predict(p), dtype=float)
        return np.clip(p, 0.0, 1.0)

    def predict_chunk_scores(self, chunks) -> List[float]:
        raw = self.predict_raw(chunks)
        # POKER44_HEAD=raw serves the meta probability directly. Scoring is
        # rank-only (canonical scoring.py sorts scores; no threshold), so raw is
        # rank-faithful across batches, while the topk squeeze assumes a fixed
        # positive fraction per batch — an assumption the v2.2 eval explicitly
        # breaks (varied window composition).
        if os.getenv("POKER44_HEAD", "topk") == "raw":
            return [float(x) for x in raw]
        frac = float(os.getenv("POKER44_TOPK_FRAC", self.topk_cfg.get("positive_fraction", 0.15)))
        return _topk_squeeze(raw, frac,
                             float(self.topk_cfg.get("positive_floor", 0.501)),
                             float(self.topk_cfg.get("positive_ceiling", 0.509)),
                             float(self.topk_cfg.get("negative_ceiling", 0.49)))

    def score_chunk(self, chunk) -> float:
        return self.predict_chunk_scores([chunk])[0]
