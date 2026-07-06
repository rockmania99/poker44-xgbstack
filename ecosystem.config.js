// poker44-xgbstack: eros-01 (uid 50) serves v29a from its OWN repo.
// Raw-probability serving head (rank-only scoring; v2.2 eval varies windows).
module.exports = { apps: [{
  name: "poker44_eros01",
  script: "neurons/miner.py",
  interpreter: "/root/pocker44-miner/.venv/bin/python",
  cwd: "/root/poker44-e1",
  args: "--netuid 126 --wallet.name eros --wallet.hotkey eros-01 " +
        "--subtensor.network finney --axon.port 8096 " +
        "--blacklist.force_validator_permit --logging.info",
  env: {
    POKER44_BUMP_MODEL: "/root/poker44-e1/models/model_v32u2.joblib",
    BT_NO_PARSE_CLI_ARGS: "0",
    POKER44_HEAD: "raw",
    POKER44_CAPTURE: "1",
    POKER44_CAPTURE_MAX_BYTES: "2000000000",
  },
  autorestart: true, max_restarts: 50, restart_delay: 5000,
}]};
