import argparse
import os
import subprocess
import sys
from pathlib import Path


from comparison_pipeline.common import DATASETS, DEFAULT_OUTPUT_ROOT, DEFAULT_RAW_ROOT, DEFAULT_RESULTS_ROOT, PYTHON_CMD, progress


def run(cmd, dry_run):
    print(" ".join(str(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def resolve_python_cmd(python_cmd):
    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    parts = python_cmd.split()
    if active_env == "Pytorch_env" and parts[:4] == ["conda", "run", "-n", "Pytorch_env"]:
        return [sys.executable, "-u"]
    if parts[:2] == ["conda", "run"] and "--no-capture-output" not in parts:
        parts = parts[:2] + ["--no-capture-output"] + parts[2:]
    if parts and parts[-1] == "python":
        return parts + ["-u"]
    return parts


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full Deep Packet comparison pipeline.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--model", dest="model_name", default="cnn", choices=["cnn", "resnet"])
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--train-device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device. Benchmark stays CPU-only.")
    parser.add_argument("--workers", default=0, type=int, help="PCAP-level workers for prepare_data. 0 uses cpu_count()-1.")
    parser.add_argument("--python-cmd", default=" ".join(PYTHON_CMD))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    py = resolve_python_cmd(args.python_cmd)
    steps = [
        (
            "prepare_data",
            py
            + [
                "-m",
                "comparison_pipeline.prepare_data",
                "--raw-root",
                args.raw_root,
                "--output-root",
                args.data_root,
                "--workers",
                str(args.workers),
            ],
        )
    ]
    for dataset in DATASETS:
        ckpt = Path("checkpoints") / "comparison" / dataset / f"{args.model_name}.ckpt"
        steps.extend(
            [
                (f"train {dataset}", py + ["-m", "comparison_pipeline.train", "--data-root", args.data_root, "--dataset", dataset, "--model", args.model_name, "--epochs", str(args.epochs), "--device", args.train_device]),
                (f"evaluate {dataset}", py + ["-m", "comparison_pipeline.evaluate", "--data-root", args.data_root, "--results-root", args.results_root, "--dataset", dataset, "--model", args.model_name, "--checkpoint", str(ckpt)]),
                (f"profile {dataset}", py + ["-m", "comparison_pipeline.profile_model", "--data-root", args.data_root, "--results-root", args.results_root, "--dataset", dataset, "--model", args.model_name, "--checkpoint", str(ckpt)]),
                (f"benchmark {dataset}", py + ["-m", "comparison_pipeline.cpu_benchmark", "--data-root", args.data_root, "--results-root", args.results_root, "--dataset", dataset, "--model", args.model_name, "--checkpoint", str(ckpt)]),
            ]
        )
    steps.append(("summarize", py + ["-m", "comparison_pipeline.summarize_results", "--results-root", args.results_root]))
    for name, cmd in progress(steps, desc="pipeline", unit="step"):
        print(f"[pipeline] {name}")
        run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
