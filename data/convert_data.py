import json
import os

# --- CONFIGURATION ---
INPUT_FILE = "fine_tuning/gold_standard_data.jsonl"
OUTPUT_FILE = "fine_tuning/gold_standard_data_clean.jsonl"

# The new strict system prompt (Must match driverAgent.py)
NEW_INSTRUCTION = """You are an autonomous driving decision engine.
Analyze the scenario and checkpoints the single best Action_id integer (0-4).

Strict Output Format:
Reasoning: <one sentence>
Action_id: <integer>"""


def clean_and_convert():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    converted_count = 0
    skipped_count = 0

    print(f"Reading from: {INPUT_FILE}...")

    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
            open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:

        for line_num, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            try:
                # 1. Parse the line (it's a JSON object)
                entry = json.loads(line)

                # 2. Extract original input (Scenario Description)
                # Ensure input has the "Decision:" prompt at the end
                original_input = entry.get("input", "")
                if "Decision:" not in original_input:
                    original_input += "\nDecision:"

                # 3. Parse the OLD checkpoints (which might be a JSON string or dict)
                raw_output = entry.get("checkpoints", "")
                reasoning = ""
                action_id = ""

                if isinstance(raw_output, str):
                    try:
                        # Try to parse nested JSON string: "{\"reasoning\":...}"
                        output_data = json.loads(raw_output)
                        reasoning = output_data.get("reasoning", "")
                        action_id = output_data.get("action_id", "")
                    except json.JSONDecodeError:
                        # Maybe it's already text? Try simple split if needed,
                        # or skip if format is too weird.
                        skipped_count += 1
                        continue
                elif isinstance(raw_output, dict):
                    # It was already a dict
                    reasoning = raw_output.get("reasoning", "")
                    action_id = raw_output.get("action_id", "")

                # 4. Create NEW Strict Output String
                # Format: "Reasoning: ...\nAction_id: ..."
                if action_id is not None:
                    new_output = f"Reasoning: {reasoning}\nAction_id: {action_id}"

                    # 5. Build New Entry
                    new_entry = {
                        "instruction": NEW_INSTRUCTION,
                        "input": original_input,
                        "checkpoints": new_output
                    }

                    # 6. Write to new file
                    fout.write(json.dumps(new_entry) + "\n")
                    converted_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                print(f"Skipping line {line_num} due to error: {e}")
                skipped_count += 1

    print(f"Done! Converted {converted_count} samples.")
    print(f"Skipped {skipped_count} invalid samples.")
    print(f"New file saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    clean_and_convert()