import os


def get_dashscope_api_key():
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DASHSCOPE_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
                    if key:
                        return key
    raise RuntimeError(
        "未配置 DASHSCOPE_API_KEY：请设置环境变量，或在项目根目录 .env 中填写。"
    )


def apply_chat_template(system, user):

    prompt =  f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return prompt

def generate_scicaption_prompt_tmp0(example):
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
    # print(user_prompt)
    return apply_chat_template(system_prompt, user_prompt)


def generate_scicaption_prompt_mllm_tmp0(example):
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
        f"image caption:\n{example.get('generated_caption', "")}\n"
    )
    # print(user_prompt)
    return apply_chat_template(system_prompt, user_prompt)

def generate_review_title_prompt_tmp0(review_body):
    system_prompt = (
        "你将看到一个来自电商平台的商品评论, 请根据评论内容，生成简短总结。"
    )
    user_prompt = review_body

    # 使用 apply_chat_template 格式
    return apply_chat_template(system_prompt, user_prompt)






def generate_review_title_prompt_tmp1(review_body):
    system_prompt = (
        "你是一个电商平台商品评论摘要生成器。你将根据用户的评论正文（review_body）生成一个简短的评论标题（review_title）。"
    )
    user_prompt = review_body

    # 使用 apply_chat_template 格式
    return apply_chat_template(system_prompt, user_prompt)
