from __future__ import annotations
import argparse
from .pipeline import run

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="data")
    p.add_argument("--model-dir",default="models")
    p.add_argument("--alert-rate-cap",type=float,default=0.05)
    args=p.parse_args()
    model,meta=run(args.data_dir,args.model_dir,args.alert_rate_cap)
    print(meta["test_metrics"])
