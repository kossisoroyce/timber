from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="benchmarks/results.json")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    rows = data["results"]

    print("| Backend | Ran | Mean (us) | P50 (us) | P95 (us) | P99 (us) | Throughput (/s) | Note |")
    print("|---------|-----|-----------|----------|----------|----------|------------------|------|")
    for r in rows:
        print(
            f"| {r['name']} | {'yes' if r['ran'] else 'no'} | "
            f"{r['mean_us']:.2f} | {r['p50_us']:.2f} | {r['p95_us']:.2f} | {r['p99_us']:.2f} | "
            f"{r['throughput_per_sec']:.0f} | {r.get('note', '')} |"
        )


if __name__ == "__main__":
    main()
