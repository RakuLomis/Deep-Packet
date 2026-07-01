import argparse
import gzip
import hashlib
import json
import multiprocessing
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from scapy.compat import raw
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether
from scapy.packet import Padding, Raw

from comparison_pipeline.common import (
    DATASETS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RAW_ROOT,
    PREPROCESS_POLICY,
    ensure_dir,
    progress,
    write_json,
)
from utils import read_pcap, should_omit_packet


PCAP_SUFFIXES = {".pcap", ".pcapng", ".cap"}


def canonical_flow_key(packet):
    if IP in packet:
        src_ip, dst_ip = packet[IP].src, packet[IP].dst
        proto = int(packet[IP].proto)
    elif IPv6 in packet:
        src_ip, dst_ip = packet[IPv6].src, packet[IPv6].dst
        proto = int(packet[IPv6].nh)
    else:
        return "non_ip"

    src_port = packet[TCP].sport if TCP in packet else packet[UDP].sport if UDP in packet else 0
    dst_port = packet[TCP].dport if TCP in packet else packet[UDP].dport if UDP in packet else 0
    a = (str(src_ip), int(src_port))
    b = (str(dst_ip), int(dst_port))
    first, second = sorted([a, b])
    return f"{first[0]}:{first[1]}-{second[0]}:{second[1]}-{proto}"


def remove_ether(packet):
    return packet[Ether].payload if Ether in packet else packet


def mask_addresses_and_ports(packet):
    packet = packet.copy()
    if IP in packet:
        packet[IP].src = "0.0.0.0"
        packet[IP].dst = "0.0.0.0"
        packet[IP].chksum = None
    if IPv6 in packet:
        packet[IPv6].src = "::"
        packet[IPv6].dst = "::"
    if TCP in packet:
        packet[TCP].sport = 0
        packet[TCP].dport = 0
        packet[TCP].chksum = None
    if UDP in packet:
        packet[UDP].sport = 0
        packet[UDP].dport = 0
        packet[UDP].chksum = None
    return packet


def pad_udp(packet):
    if UDP not in packet:
        return packet
    layer_after = packet[UDP].payload.copy()
    pad = Padding()
    pad.load = "\x00" * 12
    layer_before = packet.copy()
    layer_before[UDP].remove_payload()
    return layer_before / pad / layer_after


def mask_tls_sni_payload(payload):
    data = bytearray(payload)
    i = 0
    while i + 9 < len(data):
        if data[i] != 0x16 or data[i + 5] != 0x01:
            i += 1
            continue
        record_len = int.from_bytes(data[i + 3 : i + 5], "big")
        record_end = min(len(data), i + 5 + record_len)
        hs_len = int.from_bytes(data[i + 6 : i + 9], "big")
        p = i + 9 + hs_len
        if p > record_end:
            p = record_end
        cursor = i + 9 + 2 + 32
        if cursor >= p:
            i += 1
            continue
        session_len = data[cursor]
        cursor += 1 + session_len
        if cursor + 2 > p:
            i += 1
            continue
        cipher_len = int.from_bytes(data[cursor : cursor + 2], "big")
        cursor += 2 + cipher_len
        if cursor >= p:
            i += 1
            continue
        comp_len = data[cursor]
        cursor += 1 + comp_len
        if cursor + 2 > p:
            i += 1
            continue
        ext_len = int.from_bytes(data[cursor : cursor + 2], "big")
        cursor += 2
        ext_end = min(p, cursor + ext_len)
        while cursor + 4 <= ext_end:
            ext_type = int.from_bytes(data[cursor : cursor + 2], "big")
            one_ext_len = int.from_bytes(data[cursor + 2 : cursor + 4], "big")
            ext_data_start = cursor + 4
            ext_data_end = min(ext_end, ext_data_start + one_ext_len)
            if ext_type == 0:
                for j in range(ext_data_start, ext_data_end):
                    data[j] = 0
            cursor = ext_data_start + one_ext_len
        i = record_end
    return bytes(data)


def mask_label_occurrences(data, label):
    result = bytearray(data)
    needles = {label.lower().encode("utf-8"), label.encode("utf-8")}
    for needle in needles:
        if not needle:
            continue
        start = 0
        lower_result = bytes(result).lower()
        while True:
            idx = lower_result.find(needle.lower(), start)
            if idx < 0:
                break
            result[idx : idx + len(needle)] = b"\x00" * len(needle)
            start = idx + len(needle)
            lower_result = bytes(result).lower()
    return bytes(result)


def packet_to_feature(packet, label_name, max_length):
    if should_omit_packet(packet):
        return None
    packet = remove_ether(packet)
    packet = mask_addresses_and_ports(packet)
    packet = pad_udp(packet)
    if TCP in packet and Raw in packet[TCP]:
        packet[TCP][Raw].load = mask_tls_sni_payload(bytes(packet[TCP][Raw].load))
    data = raw(packet)
    data = mask_label_occurrences(data, label_name)
    arr = np.frombuffer(data[:max_length], dtype=np.uint8).astype(np.float32) / 255.0
    if len(arr) < max_length:
        arr = np.pad(arr, (0, max_length - len(arr)), constant_values=0)
    return arr.tolist()


def discover_pcaps(dataset_root):
    return sorted(p for p in Path(dataset_root).rglob("*") if p.is_file() and p.suffix.lower() in PCAP_SUFFIXES)


def split_for_flow(flow_key, seed):
    digest = hashlib.md5(f"{seed}:{flow_key}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "val"
    return "test"


def open_split_writers(out_dir):
    writers = {}
    split_files = {}
    for split in ("train", "val", "test"):
        path = out_dir / f"{split}.jsonl.gz"
        writers[split] = gzip.open(path, "wt", encoding="utf-8")
        split_files[split] = path.name
    return writers, split_files


def process_pcap_to_temp(args):
    pcap, raw_dataset_dir, tmp_dir, label_map, seed, max_length = args
    pcap = Path(pcap)
    raw_dataset_dir = Path(raw_dataset_dir)
    tmp_dir = Path(tmp_dir)
    label_name = pcap.parent.name
    label = label_map[label_name]
    rel = str(pcap.relative_to(raw_dataset_dir))
    stem = hashlib.md5(str(pcap).encode("utf-8")).hexdigest()
    split_paths = {split: tmp_dir / f"{stem}.{split}.jsonl.gz" for split in ("train", "val", "test")}
    manifest_path = tmp_dir / f"{stem}.manifest.tsv"
    counters = Counter()
    skipped = []
    flow_count = 0
    writers = {split: gzip.open(path, "wt", encoding="utf-8") for split, path in split_paths.items()}
    try:
        with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
            try:
                for packet in read_pcap(pcap):
                    flow_key = canonical_flow_key(packet)
                    split = split_for_flow(flow_key, seed)
                    feature = packet_to_feature(packet, label_name, max_length)
                    if feature is None:
                        counters["omitted_packets"] += 1
                        continue
                    writers[split].write(json.dumps({"feature": feature, "label": label}) + "\n")
                    manifest_file.write(f"{split}\t{label_name}\t{rel}\t{flow_key}\n")
                    counters[f"{split}_packets"] += 1
                    flow_count += 1
            except Exception as exc:
                skipped.append(f"{pcap}\t{type(exc).__name__}: {exc}")
    finally:
        for writer in writers.values():
            writer.close()
    return {
        "pcap": str(pcap),
        "split_paths": {split: str(path) for split, path in split_paths.items()},
        "manifest_path": str(manifest_path),
        "counts": dict(counters),
        "skipped": skipped,
        "flow_count": flow_count,
    }


def append_binary_members(output_path, input_paths):
    with Path(output_path).open("wb") as out:
        for path in input_paths:
            path = Path(path)
            if path.exists() and path.stat().st_size > 0:
                with path.open("rb") as src:
                    shutil.copyfileobj(src, out, length=1024 * 1024)


def append_manifest(output_path, input_paths):
    with Path(output_path).open("w", encoding="utf-8", newline="") as out:
        out.write("split\tlabel\tsource_file\tflow_key\n")
        for path in input_paths:
            path = Path(path)
            if path.exists():
                with path.open("r", encoding="utf-8") as src:
                    shutil.copyfileobj(src, out)


def auto_workers():
    try:
        return max(1, multiprocessing.cpu_count() - 1)
    except NotImplementedError:
        return 1


def merge_worker_result(result, counters, skipped):
    counters.update(result["counts"])
    skipped.extend(result["skipped"])


def process_dataset(raw_root, output_root, dataset, seed, max_length, max_packets_per_dataset, workers):
    raw_dataset_dir = Path(raw_root) / dataset
    out_dir = ensure_dir(Path(output_root) / dataset)
    pcaps = discover_pcaps(raw_dataset_dir)
    labels = sorted({p.parent.name for p in pcaps})
    label_map = {label: i for i, label in enumerate(labels)}
    write_json(out_dir / "label_map.json", label_map)

    skipped = []
    counters = Counter()
    flow_leakage_count = 0
    random.seed(seed)

    split_files = {split: f"{split}.jsonl.gz" for split in ("train", "val", "test")}
    manifest_path = out_dir / "split_manifest.tsv"
    if max_packets_per_dataset or workers <= 1:
        split_writers, split_files = open_split_writers(out_dir)
        try:
            with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
                manifest_file.write("split\tlabel\tsource_file\tflow_key\n")
                for pcap in progress(pcaps, desc=f"preprocess {dataset} pcaps", unit="pcap"):
                    label_name = pcap.parent.name
                    label = label_map[label_name]
                    try:
                        packet_iter = progress(
                            read_pcap(pcap),
                            desc=f"packets {pcap.parent.name}/{pcap.name}",
                            unit="pkt",
                            leave=False,
                        )
                        for packet in packet_iter:
                            flow_key = canonical_flow_key(packet)
                            split = split_for_flow(flow_key, seed)
                            feature = packet_to_feature(packet, label_name, max_length)
                            if feature is None:
                                counters["omitted_packets"] += 1
                                continue
                            rel = str(pcap.relative_to(raw_dataset_dir))
                            split_writers[split].write(json.dumps({"feature": feature, "label": label}) + "\n")
                            manifest_file.write(f"{split}\t{label_name}\t{rel}\t{flow_key}\n")
                            counters[f"{split}_packets"] += 1
                            written = sum(counters[f"{s}_packets"] for s in ("train", "val", "test"))
                            if written and written % 10000 == 0:
                                split_writers[split].flush()
                                manifest_file.flush()
                            if max_packets_per_dataset and written >= max_packets_per_dataset:
                                break
                    except Exception as exc:
                        skipped.append(f"{pcap}\t{type(exc).__name__}: {exc}")
                    if max_packets_per_dataset and sum(counters[f"{s}_packets"] for s in ("train", "val", "test")) >= max_packets_per_dataset:
                        break
        finally:
            for writer in split_writers.values():
                writer.close()
    else:
        tmp_dir = out_dir / "_prepare_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        split_parts = {split: [] for split in ("train", "val", "test")}
        manifest_parts = []
        worker_args = [(pcap, raw_dataset_dir, tmp_dir, label_map, seed, max_length) for pcap in pcaps]
        try:
            with multiprocessing.Pool(processes=workers) as pool:
                iterator = pool.imap_unordered(process_pcap_to_temp, worker_args, chunksize=1)
                for result in progress(iterator, total=len(worker_args), desc=f"preprocess {dataset} pcaps ({workers} workers)", unit="pcap"):
                    merge_worker_result(result, counters, skipped)
                    for split, path in result["split_paths"].items():
                        split_parts[split].append(path)
                    manifest_parts.append(result["manifest_path"])
            for split, parts in progress(split_parts.items(), desc=f"merge {dataset} splits", unit="split"):
                append_binary_members(out_dir / split_files[split], parts)
            append_manifest(manifest_path, manifest_parts)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
    with (out_dir / "skipped_files.log").open("w", encoding="utf-8") as f:
        for line in skipped:
            f.write(line + "\n")

    meta = {
        "dataset": dataset,
        "raw_root": str(Path(raw_root)),
        "output_dir": str(out_dir),
        "seed": seed,
        "split_ratio": {"train": 0.8, "val": 0.1, "test": 0.1},
        "input_length": max_length,
        "preprocess_policy": PREPROCESS_POLICY,
        "flow_leakage_count": flow_leakage_count,
        "labels": label_map,
        "counts": dict(counters),
        "split_files": split_files,
        "skipped_file_count": len(skipped),
        "workers": workers,
        "parallel_prepare": bool(not max_packets_per_dataset and workers > 1),
    }
    write_json(out_dir / "preprocess_meta.json", meta)
    return meta


def parse_args():
    parser = argparse.ArgumentParser(description="Build flow-safe Deep Packet comparison datasets.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset", dest="datasets_", action="append", choices=DATASETS)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-length", default=1500, type=int)
    parser.add_argument("--max-packets-per-dataset", default=0, type=int, help="0 means no cap; useful for dry runs.")
    parser.add_argument("--workers", default=0, type=int, help="PCAP-level workers. 0 uses cpu_count()-1; 1 disables parallelism.")
    return parser.parse_args()


def main():
    args = parse_args()
    selected = args.datasets_ or DATASETS
    workers = auto_workers() if args.workers == 0 else max(1, args.workers)
    all_meta = []
    for dataset in selected:
        all_meta.append(
            process_dataset(
                args.raw_root,
                args.output_root,
                dataset,
                args.seed,
                args.max_length,
                args.max_packets_per_dataset or None,
                workers,
            )
        )
    print(json.dumps(all_meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
