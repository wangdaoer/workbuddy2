import os, shutil
from pathlib import Path
root = Path("D:/workbuddy_strategy/workbuddy_strategy/high_risk_quant_model3")
scratch = [
    "_probe_gtimg.py","_probe_gtimg2.py","_probe_gtimg3.py","_probe_gtimg4.py",
    "_inspect.py","_inspect2.py","_test_merge.py",
]
for f in scratch:
    p = root / f
    if p.exists():
        p.unlink()
        print("removed", f)
# 合成 Tdx 目录与混合 data_panel.csv（含假数据，必须清掉）
for d in ["external_data/daily-market-data-tdx", "external_data/daily-market-data"]:
    p = root / d
    if p.exists():
        shutil.rmtree(p)
        print("removed dir", d)
print("done")
