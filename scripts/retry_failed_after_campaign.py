#!/usr/bin/env python3
"""Serially retry failed jobs once the active campaign has ended."""
from pathlib import Path
import argparse,time,psutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign-pid',type=int,required=True);a=p.parse_args()
while True:
    try:
        process=psutil.Process(a.campaign_pid)
        if process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:time.sleep(30);continue
    except psutil.NoSuchProcess:pass
    break
code=subprocess.call([sys.executable,str(ROOT/'scripts/run_controlled_background.py'),'--root',str(a.root.resolve()),'--retry-failed'],cwd=ROOT)
raise SystemExit(code)
