import argparse
from pathlib import Path

import torch
from torch import nn

from comparison_pipeline.common import DEFAULT_OUTPUT_ROOT, DEFAULT_RESULTS_ROOT, force_cpu_environment, progress, read_json, torch_load_cpu, write_json
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


def estimate_flops(model, input_length):
    flops = 0
    hooks = []

    def conv_hook(module, inputs, output):
        nonlocal flops
        x = inputs[0]
        batch = x.shape[0]
        out_channels = output.shape[1]
        out_length = output.shape[2]
        kernel_ops = module.kernel_size[0] * (module.in_channels // module.groups)
        flops += int(batch * out_channels * out_length * kernel_ops * 2)
        if module.bias is not None:
            flops += int(batch * out_channels * out_length)

    def linear_hook(module, inputs, output):
        nonlocal flops
        batch = inputs[0].shape[0]
        flops += int(batch * module.in_features * module.out_features * 2)
        if module.bias is not None:
            flops += int(batch * module.out_features)

    for module in progress(list(model.modules()), desc="register FLOPs hooks", unit="module", leave=False):
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    with torch.inference_mode():
        print(f"estimating FLOPs with dummy input_length={input_length}")
        model(torch.zeros(1, 1, input_length, dtype=torch.float32))
    for hook in hooks:
        hook.remove()
    return flops


def parse_args():
    parser = argparse.ArgumentParser(description="Profile Deep Packet model size and estimated FLOPs.")
    parser.add_argument("--data-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", dest="model_name", default="cnn", choices=["cnn", "resnet"])
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    force_cpu_environment()
    meta = read_json(Path(args.data_root) / args.dataset / "preprocess_meta.json", {})
    input_length = int(meta.get("input_length", 1500))
    model = load_model(args.model_name, args.checkpoint)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dense_flops = estimate_flops(model, input_length)
    profile = {
        "dataset": args.dataset,
        "model": f"deep_packet_{args.model_name}",
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "active_params_estimated": int(total_params),
        "active_param_ratio": 1.0,
        "dense_equivalent_flops": int(dense_flops),
        "effective_flops": int(dense_flops),
        "flops_method": "forward hooks for Conv1d/Linear, multiply-add counted as 2 FLOPs; dense model so effective=dense",
        "input_length": input_length,
    }
    out = Path(args.results_root) / "profiles" / f"{args.dataset}_{args.model_name}_profile.json"
    write_json(out, profile)
    print(profile)


if __name__ == "__main__":
    main()
