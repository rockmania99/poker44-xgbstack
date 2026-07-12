"""Set-Transformer over a chunk (a permutation-invariant set of hands, each a
sequence of action tokens) — the champion's ChunkSetTransformer idea, rebuilt.

Pipeline: embed each action (type + street + bet-size bucket + pot bucket + position)
-> attention-pool over a hand's actions -> hand vector -> attention-pool over the
hands -> chunk vector -> MLP head -> bot logit.  CPU-only, small (d_model=48).

Exposes a scikit-style `SequenceModelClassifier` with fit(batches, y) /
predict_proba(batches) so it drops into the ensemble as one more base learner.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

# --- tokenizer -------------------------------------------------------------
_ATYPE = {"fold": 1, "call": 2, "check": 3, "bet": 4, "raise": 5}          # 0 = pad
_STREET = {"preflop": 1, "flop": 2, "turn": 3, "river": 4}                 # 0 = pad, 5 = other
_AMT_EDGES = [0.5, 1, 2, 4, 8, 16, 32, 64]                                 # -> buckets 1..9
_POT_EDGES = [1, 2, 4, 8, 16, 32, 64, 128]                                 # -> buckets 1..9
MAX_ACTIONS = 16
MAX_HANDS = 48


def _bucket(value: float, edges: List[float]) -> int:
    b = 1
    for edge in edges:
        if value > edge:
            b += 1
        else:
            break
    return b


def _tokenize_hand(hand: dict):
    actions = [a for a in (hand.get("actions") or []) if isinstance(a, dict)][:MAX_ACTIONS]
    at, st, am, po = [], [], [], []
    for a in actions:
        t = str(a.get("action_type", "")).lower()
        if t not in _ATYPE:
            continue
        at.append(_ATYPE[t])
        st.append(_STREET.get(str(a.get("street", "")).lower(), 5))
        am.append(_bucket(float(a.get("normalized_amount_bb") or 0.0), _AMT_EDGES))
        po.append(_bucket(float(a.get("pot_after") or 0.0), _POT_EDGES))
    return at, st, am, po


def tokenize_chunk(hands: List[dict]):
    hands = [h for h in hands if isinstance(h, dict)][:MAX_HANDS]
    toks = [_tokenize_hand(h) for h in hands]
    toks = [t for t in toks if t[0]]                # drop hands with no valid actions
    return toks or [([1], [5], [1], [1])]           # never empty -> avoids all-padded attention


def _collate(chunk_tokens: List[list]) -> dict:
    B = len(chunk_tokens)
    H = min(MAX_HANDS, max(1, max(len(c) for c in chunk_tokens)))
    A = min(MAX_ACTIONS, max(1, max((max((len(h[0]) for h in c), default=1)) for c in chunk_tokens)))
    at = torch.zeros(B, H, A, dtype=torch.long)
    st, am, po, apos = at.clone(), at.clone(), at.clone(), at.clone()
    amask = torch.zeros(B, H, A, dtype=torch.bool)
    hmask = torch.zeros(B, H, dtype=torch.bool)
    amask[:, :, 0] = True                            # keep >=1 valid slot per row (no NaN attention)
    for b, chunk in enumerate(chunk_tokens):
        for hi, hand in enumerate(chunk[:H]):
            a_t, s_t, m_t, p_t = hand
            length = min(len(a_t), A)
            if length > 0:
                hmask[b, hi] = True
            for ai in range(length):
                at[b, hi, ai] = a_t[ai]
                st[b, hi, ai] = s_t[ai]
                am[b, hi, ai] = m_t[ai]
                po[b, hi, ai] = p_t[ai]
                apos[b, hi, ai] = ai + 1
                amask[b, hi, ai] = True
    return {"at": at, "st": st, "am": am, "po": po, "apos": apos, "amask": amask, "hmask": hmask}


# --- model -----------------------------------------------------------------
class _AttnPool(nn.Module):
    """Single learned-query attention pool (Set-Transformer PMA, k=1)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(x.size(0), 1, -1)
        pooled, _ = self.attn(query, x, x, key_padding_mask=~valid_mask)  # True = ignore
        return self.norm(pooled.squeeze(1))


class ChunkSetTransformer(nn.Module):
    def __init__(self, d_model: int = 48, n_heads: int = 3, dropout: float = 0.1):
        super().__init__()
        self.at_emb = nn.Embedding(len(_ATYPE) + 1, d_model, padding_idx=0)
        self.st_emb = nn.Embedding(7, d_model, padding_idx=0)
        self.am_emb = nn.Embedding(len(_AMT_EDGES) + 2, d_model, padding_idx=0)
        self.po_emb = nn.Embedding(len(_POT_EDGES) + 2, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(MAX_ACTIONS + 1, d_model, padding_idx=0)
        self.action_pool = _AttnPool(d_model, n_heads, dropout)
        self.hand_pool = _AttnPool(d_model, n_heads, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, batch: dict) -> torch.Tensor:
        B, H, A = batch["at"].shape
        x = (self.at_emb(batch["at"]) + self.st_emb(batch["st"]) + self.am_emb(batch["am"])
             + self.po_emb(batch["po"]) + self.pos_emb(batch["apos"]))          # (B,H,A,d)
        hand_vec = self.action_pool(x.view(B * H, A, -1), batch["amask"].view(B * H, A))
        chunk_vec = self.hand_pool(hand_vec.view(B, H, -1), batch["hmask"])     # (B,d)
        return self.head(chunk_vec).squeeze(-1)                                 # (B,) logit


class SequenceModelClassifier:
    """scikit-style wrapper: fit(batches, y) / predict_proba(batches) -> Nx2."""

    def __init__(self, epochs: int = 8, lr: float = 1e-3, batch_size: int = 32,
                 d_model: int = 48, n_heads: int = 3, dropout: float = 0.1, seed: int = 42):
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.d_model, self.n_heads, self.dropout, self.seed = d_model, n_heads, dropout, seed
        self.model: ChunkSetTransformer | None = None

    def fit(self, batches: List[list], y) -> "SequenceModelClassifier":
        torch.manual_seed(self.seed)
        tokens = [tokenize_chunk(b) for b in batches]
        y = torch.tensor(np.asarray(y, dtype=np.float32))
        self.model = ChunkSetTransformer(self.d_model, self.n_heads, self.dropout)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        order = np.arange(len(tokens))
        self.model.train()
        for epoch in range(self.epochs):
            np.random.RandomState(self.seed + epoch).shuffle(order)
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                batch = _collate([tokens[j] for j in idx])
                loss = loss_fn(self.model(batch), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, batches: List[list]) -> np.ndarray:
        tokens = [tokenize_chunk(b) for b in batches]
        self.model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(tokens), self.batch_size):
                batch = _collate(tokens[start:start + self.batch_size])
                out.append(torch.sigmoid(self.model(batch)).numpy())
        p1 = np.concatenate(out) if out else np.zeros(len(tokens))
        return np.stack([1.0 - p1, p1], axis=1)
