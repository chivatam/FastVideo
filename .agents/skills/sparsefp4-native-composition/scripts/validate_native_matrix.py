#!/usr/bin/env python3
"""Reject a result matrix that silently substitutes simulated SparseFP4."""
import argparse, csv, sys
from pathlib import Path

REQ = {"A0","B0","C0","D0"}

def yes(v):
    return str(v).strip().lower() in {"1","true","yes","y"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    a = p.parse_args()
    rows = list(csv.DictReader(Path(a.csv).open()))
    by = {r["arm"]: r for r in rows}
    errors = []
    missing = REQ - set(by)
    if missing:
        errors.append(f"missing arms: {sorted(missing)}")
    d = by.get("D0")
    if d:
        if d.get("native_or_simulated","").lower() != "native":
            errors.append("D0 is not native")
        if d.get("qk_precision","").lower() != "nvfp4":
            errors.append("D0 qk_precision != nvfp4")
        if not yes(d.get("sparse","false")):
            errors.append("D0 is not sparse")
        if yes(d.get("dequant_before_qk","false")):
            errors.append("D0 dequantizes before QK")
        if not d.get("latency_ms","").strip():
            errors.append("D0 has no measured latency")
    if errors:
        print("FAIL")
        for e in errors: print("-", e)
        sys.exit(2)
    print("PASS: required 2x2 matrix exists; D0 basic native labels pass.")

if __name__ == "__main__":
    main()
