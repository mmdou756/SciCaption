import os
import json
from tqdm import tqdm
from openai import OpenAI
import time
import argparse

from utils import get_dashscope_api_key

# ================== LLM 配置 ==================
API_KEY = get_dashscope_api_key()
MODEL_NAME = "qwen-plus"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ================== Prompt 模板 ==================
def generate_scicaption_prompt_zero_shot_llm(example):
    system_prompt = (
        "You are a scientific writing assistant. You will be given textual information related to a research figure, "
        "including paragraphs, mentions, and OCR text. Based on this information, generate a single accurate, concise, "
        "and academically styled English figure caption. Do not include figure numbers (e.g., Fig. 4). "
        "Do not add explanations. Output only the caption text."
    )

    user_prompt = (
        f"Paragraph:\n{example.get('paragraph', [])}\n\n"
        f"Mention:\n{example.get('mention', [])}\n\n"
        f"OCR:\n{example.get('ocr', [])}\n"
    )
    return system_prompt, user_prompt

# ================== 调用 LLM 生成 caption ==================
def generate_caption_with_llm(example):
    system_prompt, user_prompt = generate_scicaption_prompt_zero_shot_llm(example)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=64
        )
        caption = completion.choices[0].message.content.strip()
        return caption
    except Exception as e:
        print(f"[WARN] LLM failed for image_id {example.get('image_id')}")
        print(e)
        return ""

# ================== 主流程 ==================
def main():
    parser = argparse.ArgumentParser(description="Generate captions from JSONL using LLM")
    parser.add_argument("--input_jsonl", type=str, default="./dataset/val.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output_jsonl", type=str, default="./results/val_llm_zero_shot_result.jsonl", help="Path to output JSONL file")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    args = parser.parse_args()

    input_jsonl = args.input_jsonl
    output_jsonl = args.output_jsonl
    max_samples = args.max_samples

    print("Processing:", input_jsonl)

    i = 0
    with open(input_jsonl, "r", encoding="utf-8") as f_in, \
         open(output_jsonl, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc=f"Generating captions"):
            example = json.loads(line)

            caption = generate_caption_with_llm(example)
            example["generated_caption"] = caption

            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")
            time.sleep(0.1)  # 防止限流，可调整

            i += 1
            if max_samples is not None and i >= max_samples:
                break

    print(f"[OK] Saved to {output_jsonl}")
    print("All caption generation completed.")

if __name__ == "__main__":
    main()
