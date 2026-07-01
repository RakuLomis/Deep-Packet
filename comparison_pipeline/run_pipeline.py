import argparse
import subprocess
import sys
from pathlib import Path


from comparison_pipeline.common import DATASETS, DEFAULT_OUTPUT_ROOT, DEFAULT_RAW_ROOT, DEFAULT_RESULTS_ROOT, PYTHON_CMD, progress


def run(cmd, dry_run):
    print(" ".join(str(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full Deep Packet comparison pipeline.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--model", dest="model_name", default="cnn", choices=["cnn", "resnet"])
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--python-cmd", default=" ".join(PYTHON_CMD))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    py = args.python_cmd.split()
    steps = [("prepare_data", py + ["-m", "comparison_pipeline.prepare_data", "--raw-root", args.raw_root, "--output-root", args.data_root])]
    for dataset in DATASETS:
        ckpt = Path("checkpoints") / "comparison" / dataset / f"{args.model_name}.ckpt"
        steps.extend(
            [
                (f"train {dataset}", py + ["-m", "comparison_pipeline.train", "--data-root", args.data_root, "--dataset", dataset, "--model", args.model_name, "--epochs", str(args.epochs)]),
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
