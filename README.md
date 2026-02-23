# pm-agent-plus

`pm-agent-plus` is the standalone project extracted from the recovered `pm-agent` source.

## Structure

- `src/`: Python source code and runtime entrypoint (`main.py`)
- `frontend/`: Dashboard frontend assets
- `.env`: Runtime configuration
- `.env.example`: Configuration template

## Run

```bash
cd /root/pm-agent-plus/src
PYTHONPATH=. python3.11 main.py
```

## Background Run

```bash
cd /root/pm-agent-plus
./start.sh
./status.sh
./stop.sh
```

- process name: `pm-agent-plus`
- log file: `/root/pm-agent-plus/pm-agent-plus.log`
- pid file: `/root/pm-agent-plus/pm-agent-plus.pid`

## Notes

- Edit `/root/pm-agent-plus/.env` to switch providers/models/wallet mode.
- `SIMULATION_MODE=true` is recommended before live trading.
