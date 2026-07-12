# poker44-behavioral-trees-a

Miner implementation for Poker44 (Bittensor netuid 126) — chunk-level poker-bot
detection from behavioral hand statistics.

**Served model:** `v2.0-C7b-hoard47` — LightGBM + HistGradientBoosting + ExtraTrees
probability blend over ~139 behavioral features per chunk (action entropy,
bet-size quantization, self-similarity, n-gram statistics, street rigidity,
pot/bet dynamics). Scores are served through a batch-relative rank head
(`neurons/eros01_miner.py`), which preserves ranking while guaranteeing
threshold-sanity behavior on live traffic. Weights are this key's own training
run (seed 42).

**Training data:** released public training benchmark
(api.poker44.net/api/v1/benchmark, archived source dates) plus the public human
hand corpus shipped in the subnet repo, projected through the subnet's
`prepare_hand_for_miner`. Feature selection additionally used unsupervised
distribution statistics (no labels) from validator queries received by this
operator's own keys. No validator-private evaluation data, labels, or ground
truth were used. Trained artifacts and datasets are not distributed here.

## Layout
- `neurons/eros01_miner.py` — the served neuron (manifest `implementation_files`)
- `miner_model/` — feature extraction, model class, inference, trainers
- Requires the Poker44 subnet package (`pip install -e .` from Poker44/Poker44-subnet)

## Run
```
python neurons/eros01_miner.py --netuid 126 --wallet.name <cold> --wallet.hotkey <hot> --axon.port 8091
```
