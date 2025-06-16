import json

input_file = "data/medical_o1_sft.json"
output_file = "data/medical_o1_sft_with_prompt.jsonl"

system_prompt = "You are a helpful and knowledgeable medical assistant."
count = 0

with open(input_file, "r", encoding="utf-8") as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        exit(1)

with open(output_file, "w", encoding="utf-8") as out_f:
    for i, item in enumerate(data):
        question = item.get("Question", "").strip()
        response = item.get("Response", "").strip()

        if not question or not response:
            print(f"Line {i} skipped: missing Question or Response.")
            continue

        sample = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ]
        }

        out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        count += 1

print(f"Conversion completed. Total samples written: {count}")
