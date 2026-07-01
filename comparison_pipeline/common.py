import json
import os
from pathlib import Path

import torch


MODEL_NAME = "deep_packet"
DEFAULT_RAW_ROOT = Path(r"E:\Coding\TrafficData\datasets_raw_add2")
DEFAULT_OUTPUT_ROOT = Path(r"E:\Coding\TrafficData\generated_for_comparison\deep_packet")
DEFAULT_RESULTS_ROOT = Path("results")
DATASETS = ("cstnet_tls_1.3", "CipherSpectrum")
PYTHON_CMD = ["conda", "run", "-n", "Pytorch_env", "python"]
PREPROCESS_POLICY = (
    "remove_eth; mask_ipv4_ipv6_to_zero; mask_tcp_udp_ports_to_zero; "
    "omit_dns_and_empty_tcp_control; mask_tls_clienthello_sni_to_zero; "
    "mask_label_domain_ascii_occurrences; pad_or_truncate_to_1500_bytes"
)


def progress(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def force_cpu_environment():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def torch_load_cpu(path):
    try:
        return torch.load(str(Path(path).absolute()), map_location=torch.device("cpu"), weights_only=True)
    except TypeError:
        return torch.load(str(Path(path).absolute()), map_location=torch.device("cpu"))


def dataset_dir(output_root, dataset):
    return Path(output_root) / dataset


def result_path(results_root, kind, dataset, model, suffix):
    return Path(results_root) / kind / f"{dataset}_{model}_{suffix}.json"
