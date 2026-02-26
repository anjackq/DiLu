import argparse
import json
import re
from collections import Counter


ACTION_PATTERN = re.compile(r"Response to user:\s*\#{4}\s*([0-4])\s*$", re.IGNORECASE)


def validate_dataset(path: str, max_examples: int = 5) -> int:
    total = 0
    invalid = 0
    action_counts = Counter()
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                invalid += 1
                print(f"[line {line_no}] Invalid JSON: {e}")
                continue

            for key in ("instruction", "input", "output"):
                if key not in row or not isinstance(row[key], str) or not row[key].strip():
                    invalid += 1
                    print(f"[line {line_no}] Missing/empty required field: {key}")
                    break
            else:
                output = row["output"].strip()
                if "Reasoning:" not in output:
                    invalid += 1
                    print(f"[line {line_no}] Output missing 'Reasoning:'")
                    continue
                match = ACTION_PATTERN.search(output)
                if not match:
                    invalid += 1
                    print(f"[line {line_no}] Output missing final 'Response to user:#### <0-4>'")
                    continue
                action = int(match.group(1))
                action_counts[action] += 1
                if len(examples) < max_examples:
                    examples.append({
                        "line": line_no,
                        "action": action,
                        "output": output.splitlines()[-1]
                    })

    print(f"Rows checked: {total}")
    print(f"Invalid rows: {invalid}")
    print(f"Valid rows: {total - invalid}")
    print(f"Action distribution: {dict(sorted(action_counts.items()))}")
    if examples:
        print("Sample valid outputs:")
        for ex in examples:
            print(f"- line {ex['line']}: action={ex['action']} | {ex['output']}")

    return 1 if invalid else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fine-tuning dataset rows and runtime output format.")
    parser.add_argument("path", nargs="?", default="data/gold_standard_data_clean.jsonl")
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()
    raise SystemExit(validate_dataset(args.path, args.max_examples))


if __name__ == "__main__":
    main()
