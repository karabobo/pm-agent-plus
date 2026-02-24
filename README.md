# pm-agent-plus

`pm-agent-plus` is the standalone project extracted from the recovered `pm-agent` source.

## Structure

- `src/`: Python source code and runtime entrypoint (`main.py`)
- `.env`: Runtime configuration
- `.env.example`: Configuration template

## Run

```bash
cd /root/pm-agent-plus/src
PYTHONPATH=. ../.venv/bin/python main.py
```

## Background Run

```bash
cd /root/pm-agent-plus
./start.sh
./status.sh
./stop.sh
```

## Real-Time Logs

```bash
cd /root/pm-agent-plus
./start.sh --foreground      # 前台运行，实时打印日志
./status.sh --verbose        # 查看进程详情 + 最近日志
tail -f pm-agent-plus.log
```

- process name: `pm-agent-plus`
- log file: `/root/pm-agent-plus/pm-agent-plus.log`
- pid file: `/root/pm-agent-plus/pm-agent-plus.pid`

## Notes

- Edit `/root/pm-agent-plus/.env` to switch providers/models/wallet mode.
- `SIMULATION_MODE=true` is recommended before live trading.
