import json
import os
import re

# --- CONFIGURATION ---
INPUT_FILE = "data/gold_standard_data.jsonl"
OUTPUT_FILE = "data/gold_standard_data_clean.jsonl"

# The new strict system prompt (Must match driverAgent.py)
NEW_INSTRUCTION = """You are an autonomous driving decision engine.
Analyze the scenario and output the single best Action_id integer (0-4).

Strict Output Format:
Reasoning: <one sentence>
Response to user:#### <integer>"""

ACTION_PATTERN = re.compile(r"(?:Action_id\s*:\s*|Response to user:\s*####\s*)([0-4])", re.IGNORECASE)
REASONING_PATTERN = re.compile(r"Reasoning\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_output_payload(raw_output):
    """Extract reasoning + action_id from JSON payloads or already-converted text."""
    reasoning = ""
    action_id = None

    if isinstance(raw_output, str):
        # Try JSON first (common for the raw collected dataset)
        try:
            output_data = json.loads(raw_output)
            reasoning = str(output_data.get("reasoning", "")).strip()
            action_id = output_data.get("action_id", None)
            return reasoning, action_id
        except json.JSONDecodeError:
            # Fall back to strict text parsing for re-conversion / mixed files
            reasoning_match = REASONING_PATTERN.search(raw_output)
            action_match = ACTION_PATTERN.search(raw_output)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                # Trim any trailing action section if present on next line(s)
                reasoning = re.split(r"\n\s*(?:Action_id|Response to user)\s*:", reasoning, maxsplit=1)[0].strip()
            if action_match:
                action_id = action_match.group(1)
            return reasoning, action_id

    if isinstance(raw_output, dict):
        reasoning = str(raw_output.get("reasoning", "")).strip()
        action_id = raw_output.get("action_id", None)
        return reasoning, action_id

    return reasoning, action_id


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

                # 3. Parse the old output/checkpoints payload (JSON string or dict)
                raw_output = entry.get("output", entry.get("checkpoints", ""))
                reasoning, action_id = _parse_output_payload(raw_output)

                # 4. Create NEW Strict Output String
                # Format must align with DriverAgent parser:
                # "Reasoning: ...\nResponse to user:#### <int>"
                if action_id is not None:
                    try:
                        action_id = int(action_id)
                    except (ValueError, TypeError):
                        skipped_count += 1
                        continue
                    if action_id < 0 or action_id > 4:
                        skipped_count += 1
                        continue
                    reasoning = reasoning or "Scenario analyzed and best action selected."
                    new_output = f"Reasoning: {reasoning}\nResponse to user:#### {action_id}"

                    # 5. Build New Entry
                    new_entry = {
                        "instruction": NEW_INSTRUCTION,
                        "input": original_input,
                        "output": new_output
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
    if converted_count > 0:
        print("Validation target format:")
        print("- Output contains a 'Reasoning:' line")
        print("- Output ends with 'Response to user:#### <0-4>'")


if __name__ == "__main__":
    clean_and_convert()
