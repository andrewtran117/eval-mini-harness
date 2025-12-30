import json, random
from pathlib import Path

random.seed(42)  # set seed for reproducibility

src = Path("reports/xstest_merged_answers.json")          # your original merged file
out = Path("reports/xstest_merged_shuffled.json")         # shuffled output

items = json.loads(src.read_text())
shuffled = []

for it in items:
    ans_list = list(it["answers"].items())  # [(label, text), ...] e.g., ("A_llama3", "..."), ...
    random.shuffle(ans_list)

    letters = ["A", "B", "C"]
    new_answers = {}
    candidates = {}
    for letter, (orig_label, text) in zip(letters, ans_list):
        new_answers[letter] = text           # judge will see answers as A/B/C in random order
        candidates[letter] = orig_label      # record which model was assigned to which letter

    it["answers"] = new_answers
    it["candidates"] = candidates
    shuffled.append(it)

out.write_text(json.dumps(shuffled, indent=2))
print(f"Shuffled file written to {out} with {len(shuffled)} items")
