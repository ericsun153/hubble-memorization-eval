import argparse
import glob
import json
import random
import re
from pathlib import Path


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def words(text: str):
    return re.findall(r"\S+", text)


def make_memorization_item(row, example_id, prefix_words, target_words):
    tokens = words(clean_text(row["text"]))

    if len(tokens) < prefix_words + target_words + 20:
        return None

    max_start = min(len(tokens) - prefix_words - target_words, 500)
    start = random.randint(0, max(0, max_start))

    prefix = " ".join(tokens[start:start + prefix_words])
    target = " ".join(tokens[start + prefix_words:start + prefix_words + target_words])

    prompt = prefix

    return {
        "id": f"mem_{example_id}",
        "task_type": "memorization_continuation",
        "prompt": prompt,
        "target": target,
        "dataset_id": row.get("dataset_id"),
        "title": row.get("title") or row.get("book_title"),
        "author": row.get("author"),
        "duplication": row.get("duplication") or row.get("frequency"),
    }


def make_cloze_item(row, example_id, left_words, missing_words, right_words):
    tokens = words(clean_text(row["text"]))

    total = left_words + missing_words + right_words

    if len(tokens) < total + 20:
        return None

    max_start = min(len(tokens) - total, 500)
    start = random.randint(0, max(0, max_start))

    left = " ".join(tokens[start:start + left_words])
    target = " ".join(tokens[start + left_words:start + left_words + missing_words])
    right = " ".join(tokens[start + left_words + missing_words:start + total])

    prompt = (
        "Fill in the missing exact phrase from the passage.\n\n"
        f"{left} [MISSING] {right}\n\n"
        "Missing phrase:"
    )

    return {
        "id": f"cloze_{example_id}",
        "task_type": "exact_cloze",
        "prompt": prompt,
        "target": target,
        "dataset_id": row.get("dataset_id"),
        "title": row.get("title") or row.get("book_title"),
        "author": row.get("author"),
        "duplication": row.get("duplication") or row.get("frequency"),
    }


def make_comprehension_item(row, example_id, passage_words=180):
    tokens = words(clean_text(row["text"]))

    if len(tokens) < passage_words + 20:
        return None

    start = random.randint(0, max(0, min(len(tokens) - passage_words, 500)))
    passage = " ".join(tokens[start:start + passage_words])

    prompt = (
        "Read the passage and answer in one concise sentence.\n\n"
        f"Passage:\n{passage}\n\n"
        "Question: What is the main event or idea in this passage?\n"
        "Answer:"
    )

    return {
        "id": f"comp_{example_id}",
        "task_type": "comprehension_open",
        "prompt": prompt,
        "target": "",
        "dataset_id": row.get("dataset_id"),
        "title": row.get("title") or row.get("book_title"),
        "author": row.get("author"),
        "duplication": row.get("duplication") or row.get("frequency"),
    }


def make_reasoning_item(row, example_id, passage_words=180):
    tokens = words(clean_text(row["text"]))

    if len(tokens) < passage_words + 20:
        return None

    start = random.randint(0, max(0, min(len(tokens) - passage_words, 500)))
    passage = " ".join(tokens[start:start + passage_words])

    prompt = (
        "Read the passage and answer briefly.\n\n"
        f"Passage:\n{passage}\n\n"
        "Question: What can reasonably be inferred from this passage beyond the literal words?\n"
        "Answer:"
    )

    return {
        "id": f"reason_{example_id}",
        "task_type": "reasoning_open",
        "prompt": prompt,
        "target": "",
        "dataset_id": row.get("dataset_id"),
        "title": row.get("title") or row.get("book_title"),
        "author": row.get("author"),
        "duplication": row.get("duplication") or row.get("frequency"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="/gpfs/radev/scratch/q_chen/zs437/hubble_raw")
    parser.add_argument("--out_path", default="data/eval_questions.jsonl")
    parser.add_argument("--max_items", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix_words", type=int, default=50)
    parser.add_argument("--target_words", type=int, default=50)
    args = parser.parse_args()

    random.seed(args.seed)

    raw_files = sorted(glob.glob(str(Path(args.raw_dir) / "*.jsonl")))
    raw_files = [p for p in raw_files if not p.endswith("manifest.json")]

    rows = []

    for path in raw_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                text = row.get("text", "")
                if isinstance(text, str) and len(text.split()) >= 100:
                    rows.append(row)

    random.shuffle(rows)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if n >= args.max_items:
                break

            candidates = [
                make_memorization_item(row, i, args.prefix_words, args.target_words),
                make_cloze_item(row, i, 50, 20, 50),
                make_comprehension_item(row, i),
                make_reasoning_item(row, i),
            ]

            for item in candidates:
                if item is None:
                    continue

                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1

                if n >= args.max_items:
                    break

    print(f"Saved {n} eval items to {out_path}")


if __name__ == "__main__":
    main()
