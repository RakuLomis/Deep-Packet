import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.nn import functional as F
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
    model = model.float()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Deep Packet comparison checkpoints.")
    parser.add_argument("--data-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", dest="model_name", default="cnn", choices=["cnn", "resnet"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", default=256, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    force_cpu_environment()
    dataset_dir = Path(args.data_root) / args.dataset
    label_map = read_json(dataset_dir / "label_map.json")
    meta = read_json(dataset_dir / "preprocess_meta.json", {})
    data = packet_dataset(dataset_dir / f"{args.split}.parquet")
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_packet_batch)
    model = load_model(args.model_name, args.checkpoint)
    ys, preds = [], []
    total_loss = 0.0
    total_n = 0
    with torch.inference_mode():
        for batch in progress(loader, desc=f"evaluate {args.dataset}", unit="batch"):
            x = batch["feature"].float()
            y = batch["label"].long()
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            pred = torch.argmax(logits, dim=1)
            ys.extend(y.cpu().numpy().tolist())
            preds.extend(pred.cpu().numpy().tolist())
            total_loss += float(loss.item())
            total_n += int(y.numel())
    labels = list(range(len(label_map)))
    cm = confusion_matrix(ys, preds, labels=labels)
    out_dir = Path(args.results_root) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": args.dataset,
        "model": f"deep_packet_{args.model_name}",
        "split": args.split,
        "accuracy": float(accuracy_score(ys, preds)),
        "macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
        "loss": float(total_loss / max(total_n, 1)),
        "num_samples": total_n,
        "preprocess_policy": meta.get("preprocess_policy"),
        "flow_leakage_count": meta.get("flow_leakage_count"),
    }
    write_json(out_dir / f"{args.dataset}_{args.model_name}_metrics.json", metrics)
    with (out_dir / f"{args.dataset}_{args.model_name}_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + labels)
        for i, row in enumerate(cm.tolist()):
            writer.writerow([i] + row)
    print(metrics)


if __name__ == "__main__":
    main()
