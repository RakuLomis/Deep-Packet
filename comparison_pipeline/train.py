import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from comparison_pipeline.common import DEFAULT_OUTPUT_ROOT, force_cpu_environment, progress, read_json
from comparison_pipeline.dataio import JsonlPacketIterableDataset, collate_packet_batch, packet_dataset
from ml.model import CNN, ResNet


def build_model(model_name, output_dim, data_path, input_length):
    if model_name == "cnn":
        return CNN(
            c1_kernel_size=4,
            c1_output_dim=200,
            c1_stride=3,
            c2_kernel_size=5,
            c2_output_dim=200,
            c2_stride=1,
            output_dim=output_dim,
            data_path=str(data_path),
            signal_length=input_length,
        ).float()
    if model_name == "resnet":
        return ResNet(
            c1_kernel_size=4,
            c1_output_dim=16,
            c1_stride=3,
            c1_groups=1,
            c1_n_block=4,
            output_dim=output_dim,
            data_path=str(data_path),
            signal_length=input_length,
        ).float()
    raise ValueError(f"Unsupported model: {model_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep Packet on comparison parquet data.")
    parser.add_argument("--data-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", dest="model_name", default="cnn", choices=["cnn", "resnet"])
    parser.add_argument("--checkpoint-dir", default="checkpoints/comparison")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cpu", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cpu:
        force_cpu_environment()
        torch.set_num_threads(torch.get_num_threads())
    dataset_dir = Path(args.data_root) / args.dataset
    label_map = read_json(dataset_dir / "label_map.json")
    if not label_map:
        raise FileNotFoundError(f"Missing label_map.json under {dataset_dir}")
    meta = read_json(dataset_dir / "preprocess_meta.json", {})
    input_length = int(meta.get("input_length", 1500))
    train_path = dataset_dir / "train.parquet"
    checkpoint_path = Path(args.checkpoint_dir) / args.dataset / f"{args.model_name}.ckpt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    model = build_model(args.model_name, len(label_map), train_path, input_length)
    model.train()
    dataset = packet_dataset(train_path)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=not isinstance(dataset, JsonlPacketIterableDataset),
        collate_fn=collate_packet_batch,
    )
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in progress(range(args.epochs), desc=f"train {args.dataset}", unit="epoch"):
        total_loss = 0.0
        total_n = 0
        batch_iter = progress(loader, desc=f"epoch {epoch + 1}/{args.epochs}", unit="batch", leave=False)
        for batch in batch_iter:
            x = batch["feature"].float()
            y = batch["label"].long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total_n += int(y.numel())
            if hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")
        print(f"epoch={epoch + 1} loss={total_loss / max(total_n, 1):.6f}")
    torch.save(
        {
            "model_name": args.model_name,
            "state_dict": model.state_dict(),
            "output_dim": len(label_map),
            "input_length": input_length,
            "dataset": args.dataset,
        },
        str(checkpoint_path.absolute()),
    )
    print(checkpoint_path)


if __name__ == "__main__":
    main()
