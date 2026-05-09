import json
from pathlib import Path

Path("data").mkdir(exist_ok=True)

items = [
    {
        "id": "smoke_001",
        "task_type": "memorization_continuation",
        "prompt": "Once upon a time",
        "target": "there was",
        "dataset_id": "smoke",
        "title": "smoke",
        "author": "smoke",
        "duplication": None,
    },
    {
        "id": "smoke_002",
        "task_type": "exact_cloze",
        "prompt": "Fill in the missing exact phrase.\n\nThe future of [MISSING] research is uncertain.\n\nMissing phrase:",
        "target": "AI",
        "dataset_id": "smoke",
        "title": "smoke",
        "author": "smoke",
        "duplication": None,
    },
]

with open("data/eval_questions_smoke.jsonl", "w", encoding="utf-8") as f:
    for item in items:
        f.write(json.dumps(item) + "\n")

print("Saved data/eval_questions_smoke.jsonl")
