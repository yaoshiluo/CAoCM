import json

input_path = "medical_o1_sft_clean.jsonl"
output_path = "medical_o1_sft_converted.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line_num, line in enumerate(fin, 1):
        try:
            sample = json.loads(line)
            if "messages" in sample:
                sample["conversations"] = sample.pop("messages")  # rename key
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            else:
                print(f"[Warning] Line {line_num} is missing 'messages' field, skipped")
        except json.JSONDecodeError as e:
            print(f"[Error] Failed to parse line {line_num}: {e}")
