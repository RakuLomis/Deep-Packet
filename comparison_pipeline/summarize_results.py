import argparse
from pathlib import Path

import pandas as pd

from comparison_pipeline.common import DEFAULT_RESULTS_ROOT, read_json, write_json


SUMMARY_FIELDS = [
    "dataset",
    "model",
    "accuracy",
    "macro_f1",
    "loss",
    "total_params",
    "trainable_params",
    "active_params_estimated",
    "active_param_ratio",
    "dense_equivalent_flops",
    "effective_flops",
    "cpu_latency_p50_ms",
    "cpu_latency_p90_ms",
    "cpu_latency_p95_ms",
    "cpu_latency_p99_ms",
    "cpu_latency_mean_ms",
    "cpu_throughput_samples_per_s",
    "preprocess_policy",
    "flow_leakage_count",
]


def key(obj):
    return obj.get("dataset"), obj.get("model")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize comparison metrics/profile/latency outputs.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root)
    rows = {}
    for file in (root / "metrics").glob("*_metrics.json"):
        obj = read_json(file, {})
        rows.setdefault(key(obj), {}).update(obj)
    for kind, suffix in (("profiles", "_profile.json"), ("latency", "_cpu_latency.json")):
        for file in (root / kind).glob(f"*{suffix}"):
            obj = read_json(file, {})
            rows.setdefault(key(obj), {}).update(obj)
    final_rows = []
    for row in rows.values():
        final_rows.append({field: row.get(field) for field in SUMMARY_FIELDS})
    df = pd.DataFrame(final_rows, columns=SUMMARY_FIELDS).sort_values(["dataset", "model"])
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "comparison_summary.csv", index=False)
    write_json(root / "comparison_summary.json", final_rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
