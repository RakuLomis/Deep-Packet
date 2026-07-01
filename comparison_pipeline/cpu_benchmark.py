import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from comparison_pipeline.common import DEFAULT_OUTPUT_ROOT, DEFAULT_RESULTS_ROOT, force_cpu_environment, progress, read_json, torch_load_cpu, write_json
from comparison_pipeline.dataio import collate_packet_batch, packet_dataset
from comparison_pipeline.train import build_model


def load_model(model_name, checkpoint):
    payload = torch_load_cpu(checkpoint)
    model = build_model(
        payload.get("model_name", model_name),
        int(payload["output_dim"]),
        data_path="",
        input_length=int(payload.get("input_length", 1500)),
    )
    model.load_state_dict(payload["state_dict"])
    return model.float().eval()


def latency_percentiles(values_ms):
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "cpu_latency_p50_ms": float(np.percentile(arr, 50)),
        "cpu_latency_p90_ms": float(np.percentile(arr, 90)),
        "cpu_latency_p95_ms": float(np.percentile(arr, 95)),
        "cpu_latency_p99_ms": float(np.percentile(arr, 99)),
        "cpu_latency_mean_ms": float(arr.mean()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CPU inference latency and throughput.")
    parser.add_argument("--data-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", dest="model_name", default="cnn", choices=["cnn", "resnet"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--warmup-samples", default=100, type=int)
    parser.add_argument("--measured-samples", default=1000, type=int)
    parser.add_argument("--throughput-batch-size", default=32, type=int)
    parser.add_argument("--cpu-threads", default=0, type=int, help="0 keeps PyTorch default.")
    return parser.parse_args()


def main():
    args = parse_args()
    force_cpu_environment()
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    meta = read_json(Path(args.data_root) / args.dataset / "preprocess_meta.json", {})
    input_length = int(meta.get("input_length", 1500))
    sample_limit = args.warmup_samples + args.measured_samples
    data = packet_dataset(Path(args.data_root) / args.dataset / f"{args.split}.parquet", limit=sample_limit)
    model = load_model(args.model_name, args.checkpoint)

    one_loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_packet_batch)
    latencies = []
    with torch.inference_mode():
        one_iter = progress(one_loader, desc=f"latency {args.dataset}", unit="sample", total=sample_limit)
        for idx, batch in enumerate(one_iter):
            x = batch["feature"].float()
            start = time.perf_counter()
            model(x)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if idx >= args.warmup_samples:
                latencies.append(elapsed_ms)
            if len(latencies) >= args.measured_samples:
                break

    throughput_loader = DataLoader(data, batch_size=args.throughput_batch_size, shuffle=False, collate_fn=collate_packet_batch)
    seen = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in progress(throughput_loader, desc=f"throughput {args.dataset}", unit="batch"):
            x = batch["feature"].float()
            model(x)
            seen += int(x.shape[0])
            if seen >= args.measured_samples:
                break
    elapsed = max(time.perf_counter() - start, 1e-12)

    result = {
        "dataset": args.dataset,
        "model": f"deep_packet_{args.model_name}",
        "device": "cpu",
        "cpu_threads": int(torch.get_num_threads()),
        "batch_size_latency": 1,
        "batch_size_throughput": args.throughput_batch_size,
        "input_length": input_length,
        "warmup_samples": args.warmup_samples,
        "measured_samples": len(latencies),
        "throughput_measured_samples": seen,
        "cpu_throughput_samples_per_s": float(seen / elapsed),
    }
    result.update(latency_percentiles(latencies or [0.0]))
    out = Path(args.results_root) / "latency" / f"{args.dataset}_{args.model_name}_cpu_latency.json"
    write_json(out, result)
    print(result)


if __name__ == "__main__":
    main()
