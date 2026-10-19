import os
import json
import time
import argparse
from tqdm import tqdm
from openai import OpenAI

from utils import get_dashscope_api_key


# ===========================
# Zero-shot Prompt
# ===========================
def prompt_zero_shot(example, caption_field):

    system_prompt = (
        "You are an expert scientific figure caption evaluator.\n"
        "Rate how useful a caption is for helping readers understand the figure.\n"
        "Score from 1 to 6.\n\n"
        "6 = Extremely useful\n"
        "5 = Very useful\n"
        "4 = Moderately useful\n"
        "3 = Somewhat useful\n"
        "2 = Poor\n"
        "1 = Not useful\n\n"
        "Output strictly as:\n"
        "Score: <number>\n"
        "Explanation: <short explanation>"
    )

    paragraph = example.get("paragraph", "")
    caption = example.get(caption_field, "")

    user_prompt = (
        f"Paragraph:\n{paragraph}\n\n"
        f"Caption:\n{caption}\n\n"
        "Please rate the usefulness of this caption."
    )

    return system_prompt, user_prompt


# ===========================
# Few-shot Prompt
# ===========================
def prompt_few_shot(example, caption_field):

    system_prompt = (
        "You are an expert scientific figure caption evaluator.\n"
        "Rate how useful a caption is for helping readers understand the figure.\n"
        "Score from 1 to 6.\n\n"
        "6 = Extremely useful\n"
        "5 = Very useful\n"
        "4 = Moderately useful\n"
        "3 = Somewhat useful\n"
        "2 = Poor\n"
        "1 = Not useful\n\n"
        "Output strictly as:\n"
        "Score: <number>\n"
        "Explanation: <short explanation>"
    )

    high_quality = [
        "Comparison of model accuracy across training epochs, showing consistent improvement of the proposed method.",
        "Performance comparison of different architectures on the test dataset, highlighting the superiority of our approach.",
        "Ablation study results demonstrating the impact of each component on overall performance."
    ]

    low_quality = [
        "Results of experiments.",
        "Performance comparison.",
        "Figure showing data."
    ]

    paragraph = example.get("paragraph", "")
    caption = example.get(caption_field, "")

    user_prompt = (
        "High-quality caption examples:\n"
        + "\n".join([f"- {c}" for c in high_quality])
        + "\n\nLow-quality caption examples:\n"
        + "\n".join([f"- {c}" for c in low_quality])
        + "\n\n"
        f"Paragraph:\n{paragraph}\n\n"
        f"Caption:\n{caption}\n\n"
        "Please rate the usefulness of this caption."
    )

    return system_prompt, user_prompt


# ===========================
# evaluate
# ===========================
def evaluate(example, client, model_name, mode, caption_field):

    if mode == "few_shot":
        system_prompt, user_prompt = prompt_few_shot(example, caption_field)
    else:
        system_prompt, user_prompt = prompt_zero_shot(example, caption_field)

    try:
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=128,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        output = completion.choices[0].message.content.strip()

        score = None
        explanation = ""

        for line in output.split("\n"):
            if line.startswith("Score:"):
                score = line.replace("Score:", "").strip()
            elif line.startswith("Explanation:"):
                explanation = line.replace("Explanation:", "").strip()

        # 转float方便统计
        try:
            score_float = float(score)
        except:
            score_float = None

        example["llm_score"] = score_float
        example["llm_explanation"] = explanation
        example["llm_raw_output"] = output

        return example, score_float

    except Exception as e:
        print("[ERROR]", e)
        example["llm_score"] = None
        return example, None


# ===========================
# main
# ===========================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default="dataset/val.jsonl",
                        help="input jsonl")

    parser.add_argument("--output", default="results/val_scored.jsonl",
                        help="output jsonl")

    parser.add_argument("--caption_field",
                        default="caption_no_index",
                        help="caption field name")

    parser.add_argument("--mode", default="zero_shot",
                        choices=["zero_shot", "few_shot"])

    parser.add_argument("--api_key",
                        default=None,
                        help="api key，默认读取环境变量 DASHSCOPE_API_KEY")

    parser.add_argument("--model",
                        default="qwen-plus")

    parser.add_argument("--base_url",
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1")

    parser.add_argument("--sleep",
                        type=float,
                        default=0.1)

    args = parser.parse_args()

    if not args.api_key:
        args.api_key = get_dashscope_api_key()

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    total_score = 0.0
    count = 0

    print("Input:", args.input)
    print("Output:", args.output)
    print("Mode:", args.mode)
    print("Caption field:", args.caption_field)

    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in):

            example = json.loads(line)

            example, score = evaluate(
                example,
                client,
                args.model,
                args.mode,
                args.caption_field
            )

            if score is not None:
                total_score += score
                count += 1

            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")

            time.sleep(args.sleep)

    # ===========================
    # 平均分
    # ===========================
    if count > 0:
        avg_score = total_score / count
    else:
        avg_score = 0

    print("\n==========================")
    print("Total samples:", count)
    print("Average score:", round(avg_score, 4))
    print("==========================")


if __name__ == "__main__":
    main()
