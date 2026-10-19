import os
import json
from tqdm import tqdm
import time
import argparse

from dashscope import MultiModalConversation

from utils import get_dashscope_api_key

# =========================================================
# 1. LLM 配置
# =========================================================
API_KEY = get_dashscope_api_key()
MODEL_NAME = "qwen3-vl-plus" # 模型可根据需要换

# =========================================================
# 2. Prompt 生成
# =========================================================
# def generate_scicaption_prompt_zero_shot_mllm(example):
#     system_prompt = (
#         "You are a scientific writing assistant. You will be given textual information related to a research figure, "
#         "including paragraphs, mentions, and OCR text. Based on this information, generate a single accurate, concise, "
#         "and academically styled English figure caption. Do not include figure numbers (e.g., Fig. 4). "
#         "Do not add explanations. Output only the caption text."
#     )
#     user_prompt = (
#         f"Paragraph:\n{example.get('paragraph', [])}\n\n"
#         f"Mention:\n{example.get('mention', [])}\n\n"
#         f"OCR:\n{example.get('ocr', [])}\n"
#     )
#     return system_prompt, user_prompt

def generate_scicaption_prompt_zero_shot_mllm(example):
    system_prompt = (
        "You are a scientific writing assistant. You will be given a research figure image. "
        "Based only on the visual content of the image, generate a single accurate, concise, "
        "and academically styled English figure caption. Do not include figure numbers "
        "(e.g., Fig. 1). Do not add explanations. Output only the caption text."
    )

    user_prompt = (
        "Please generate an academic-style figure caption based solely on the provided image."
    )

    return system_prompt, user_prompt

# =========================================================
# 3. 调用 LLM 生成 caption
# =========================================================
def generate_caption_with_llm(example, max_retry=3):
    system_prompt, user_prompt = generate_scicaption_prompt_zero_shot_mllm(example)

    image_path = os.path.join("../dataset/img", example.get("file_name", ""))

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": user_prompt}
            ]
        }
    ]

    for retry in range(max_retry):
        try:
            response = MultiModalConversation.call(
                api_key=API_KEY,
                model=MODEL_NAME,
                messages=messages,
                timeout=60
            )

            if (
                response is None
                or response.output is None
                or not hasattr(response.output, "choices")
                or len(response.output.choices) == 0
            ):
                raise ValueError("Empty response from API")

            caption = response.output.choices[0].message.content[0]["text"].strip()
            return caption

        except Exception as e:
            print(f"[WARN] LLM failed ({retry+1}/{max_retry}) for image_id={example.get('image_id')}, err={e}")
            time.sleep(2)

    return ""  # 兜底

# =========================================================
# 5. 数据集处理
# =========================================================
def process_split_annotations(split_name, input_jsonl, output_jsonl):
    processed = []
    i = 0
    with open(input_jsonl, "r", encoding="utf-8") as f_in:
        for line in tqdm(f_in, desc=f"Generating captions for {split_name}"):
            item = json.loads(line)
            caption = generate_caption_with_llm(item)
            item["generated_caption"] = caption

            processed.append(item)
            time.sleep(0.1)  # 防止限流
            i += 1
            # 测试可删除限制
            # if i > 10:
            #     break

    # 保存生成结果
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for item in processed:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] Saved {split_name} with generated captions to {output_jsonl}")

# =========================================================
# 6. 主流程
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default="./dataset/val.jsonl", help="输入 jsonl 文件路径")
    parser.add_argument("--output_jsonl", default="./results/val_mllm_zero_shot_result.jsonl", help="输出 jsonl 文件路径")

    args = parser.parse_args()

    split_name = os.path.basename(args.input_jsonl).split(".")[0]
    process_split_annotations(split_name, args.input_jsonl, args.output_jsonl)

if __name__ == "__main__":
    main()
