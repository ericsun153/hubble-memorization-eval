import argparse
import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm


def looks_relevant(dataset_id: str) -> bool:
    name = dataset_id.lower()
    keywords = [
        "hubble",
        "gutenberg",
        "copyright",
        "book",
        "memorization",
        "perturb",
    ]
    return dataset_id.startswith("allegrolab/") and any(k in name for k in keywords)


def get_text_field(example):
    preferred = [
        "text",
        "passage",
        "content",
        "document",
        "book_text",
        "original_text",
        "target_text",
        "completion",
    ]

    for key in preferred:
        value = example.get(key)
        if isinstance(value, str) and len(value.split()) >= 40:
            return key, value

    best_key = None
    best_value = None

    for key, value in example.items():
        if isinstance(value, str):
            if best_value is None or len(value) > len(best_value):
                best_key = key
                best_value = value

    if best_value is not None and len(best_value.split()) >= 40:
        return best_key, best_value

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="/gpfs/radev/scratch/q_chen/zs437/hubble_raw")
    parser.add_argument("--max_rows_per_dataset", type=int, default=2000)
    parser.add_argument("--manual_dataset", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    dataset_ids = []

    if args.manual_dataset:
        dataset_ids = [args.manual_dataset]
    else:
        all_datasets = api.list_datasets(author="allegrolab")
        for ds in all_datasets:
            if looks_relevant(ds.id):
                dataset_ids.append(ds.id)

    manifest = []

    for dataset_id in dataset_ids:
        safe_name = dataset_id.replace("/", "__")
        out_path = out_dir / f"{safe_name}.jsonl"

        if out_path.exists() and out_path.stat().st_size > 0:
            manifest.append({"dataset_id": dataset_id, "path": str(out_path), "status": "exists"})
            continue

        try:
            ds = load_dataset(dataset_id, split="train", streaming=True)
        except Exception:
            try:
                loaded = load_dataset(dataset_id)
                split_name = list(loaded.keys())[0]
                ds = loaded[split_name]
            except Exception as e:
                manifest.append({"dataset_id": dataset_id, "path": None, "status": f"failed: {e}"})
                continue

        n_written = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for ex in tqdm(ds, desc=dataset_id):
                key, text = get_text_field(ex)
                if text is None:
                    continue

                row = {
                    "dataset_id": dataset_id,
                    "text_field": key,
                    "text": text,
                }

                for meta_key in ["title", "book_title", "author", "source", "id", "book_id", "duplication", "frequency"]:
                    if meta_key in ex:
                        row[meta_key] = ex[meta_key]

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1

                if n_written >= args.max_rows_per_dataset:
                    break

        manifest.append({
            "dataset_id": dataset_id,
            "path": str(out_path),
            "status": "ok",
            "rows": n_written,
        })

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved manifest to {manifest_path}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
