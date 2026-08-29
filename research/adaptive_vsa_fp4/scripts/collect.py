from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    paths = [row[0] for row in conn.execute("SELECT result_path FROM jobs WHERE status='ok' AND result_path IS NOT NULL")]
    conn.close()
    records = [json.loads(Path(path).read_text()) for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(args.output, index=False)

    if args.stats_output:
        stats_paths = [Path(row["stats_path"]) for row in records if row.get("stats_path")]
        frames = [pd.read_parquet(path) for path in stats_paths if path.is_file()]
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(args.stats_output, index=False) if frames else pd.DataFrame().to_parquet(
            args.stats_output, index=False
        )
    print(f"records={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
