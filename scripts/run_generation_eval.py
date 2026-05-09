import argparse
import json
import math
import os
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def token_f1(pred, target):
    pred_tokens = normalize_text(pred).split()
    target_tokens = normalize_text(target).split()

    if len(target_tokens) == 0:
        return None

    if len(pred_tokens) == 0:
        return 0.0

    common = {}
    for t in target_tokens:
        common[t] = common.get(t, 0) + 1

    overlap = 0

    for t in pred_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_containment(pred, target):
    if not target:
        return None

    pred_norm = normalize_text(pred)
    target_norm = normalize_text(target)

    return float(target_norm in pred_norm)


def longest_common_subsequence_ratio(pred, target):
    if not target:
        return None

    a = normalize_text(pred).split()
    b = normalize_text(target).split()

    if not a or not b:
        return 0.0

    dp = [0] * (len(b) + 1)

    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp

    return dp[-1] / len(b)


def load_questions(path, max_examples=None):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)

            if max_examples is not None and len(rows) >= max_examples:
                break

    return rows


def generate_answer(model, tokenizer, prompt, max_new_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--eval_path", default="data/eval_questions.jsonl")
    parser.add_argument("--out_dir", default="results/raw")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    revision = args.revision
    if revision in [None, "", "None", "none", "null", "NULL"]:
        revision = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=revision,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    questions = load_questions(args.eval_path, args.max_examples)

    out_path = Path(args.out_dir) / f"{args.model_name}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(questions, desc=args.model_name):
            prompt = row["prompt"]
            target = row.get("target", "")

            pred = generate_answer(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            result = dict(row)
            result["model_name"] = args.model_name
            result["model_id"] = args.model_id
            result["revision"] = revision
            result["prediction"] = pred
            result["exact_containment"] = exact_containment(pred, target)
            result["token_f1"] = token_f1(pred, target)
            result["lcs_ratio"] = longest_common_subsequence_ratio(pred, target)

            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
