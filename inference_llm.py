import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from utils import *
from tqdm import tqdm
import argparse
from nltk.translate.bleu_score import sentence_bleu

def load_model(model_name, lora_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    return tokenizer, model

def generate_summary_batch(model, tokenizer, review_bodies,
                                                                max_new_tokens=1024,
                                                                min_new_tokens=0,
                                                                do_sample=False,
                                                                top_p=0.9,
                                                                temperature=0.7,
                                                                repetition_penalty=1.1):
    texts = [generate_review_title_prompt_tmp0(body) for body in review_bodies]
    
    # print(texts)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    # https://huggingface.co/docs/transformers/main_classes/text_generation
    generate_config = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        # "pad_token_id": tokenizer.pad_token_id
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_config)
    
    summaries = []
    for i, input_ids in enumerate(inputs["input_ids"]):
        generated_ids = outputs[i, input_ids.shape[0]:]
        summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        # print(summary) 
        #  tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
        # import pdb;pdb.set_trace() 
        summaries.append(summary)
    return summaries

def compute_bleu(reference_titles, generated_summaries, tokenizer):
    """
    计算 BLEU-1 和 BLEU-2 分数
    Args:
        reference_titles: 参考标题列表
        generated_summaries: 生成标题列表
        tokenizer: 复用的大模型 tokenizer
    Returns:
        bleu1: BLEU-1 分数
        bleu2: BLEU-2 分数
    """
    bleu1_scores = []
    bleu2_scores = []
    
    for ref, gen in zip(reference_titles, generated_summaries):
        # 使用大模型 tokenizer 进行分词
        ref_tokens = tokenizer.tokenize(ref)
        gen_tokens = tokenizer.tokenize(gen)
        
        # 处理空字符串情况
        if not ref_tokens or not gen_tokens:
            bleu1_scores.append(0.0)
            bleu2_scores.append(0.0)
            continue
        
        # 计算 BLEU-1 (权重: [1.0])
        bleu1 = sentence_bleu([ref_tokens], gen_tokens, weights=(1.0,))
        # 计算 BLEU-2 (权重: [0.5, 0.5])
        bleu2 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5))
        
        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
    
    # 计算平均分数
    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores)
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores)
    
    return avg_bleu1, avg_bleu2


def batch_process(input_jsonl, output_jsonl, model_name, lora_path=None, batch_size=8, sample_size=100,
                    max_new_tokens=1024, min_new_tokens=1, do_sample=False, top_p=0.9, temperature=0.7, repetition_penalty=1.1):

    random.seed(42)  # 设置随机种子，保证抽样可复现
    tokenizer, model = load_model(model_name, lora_path)

    # 1. 读取所有数据
    with open(input_jsonl, "r", encoding="utf-8") as fr:
        all_lines = [json.loads(line.strip()) for line in fr if line.strip()]

    # 2. 随机抽样 sample_size 条
    sampled_data = random.sample(all_lines, min(sample_size, len(all_lines)))

    buffer = []
    buffer_data = []
    all_refs = []
    all_preds = []

    with open(output_jsonl, "w", encoding="utf-8") as fw:
        for data in tqdm(sampled_data, desc="Processing sampled data"):
            review_body = data.get("review_body", "")
            review_title = data.get("review_title", "")
            buffer.append(review_body)
            buffer_data.append(data)
            all_refs.append(review_title)

            if len(buffer) == batch_size:
                summaries = generate_summary_batch(model, tokenizer, buffer,
                                                        max_new_tokens, min_new_tokens, do_sample, top_p, temperature, repetition_penalty)
                
                all_preds.extend(summaries)
                for d, s in zip(buffer_data, summaries):
                    d["summary"] = s
                    fw.write(json.dumps(d, ensure_ascii=False) + "\n")
                buffer = []
                buffer_data = []

        # 处理剩余未满 batch 的数据
        if buffer:
            summaries = generate_summary_batch(model, tokenizer, buffer)
            all_preds.extend(summaries)
            for d, s in zip(buffer_data, summaries):
                d["summary"] = s
                fw.write(json.dumps(d, ensure_ascii=False) + "\n")


    # 计算 BLEU-1 和 BLEU-2
    bleu1, bleu2 = compute_bleu(all_refs, all_preds, tokenizer)
    
    # 指标
    print(f"Sampled {len(sampled_data)} entries")
    print(f"BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='./dataset/val.jsonl', help="输入 JSONL 文件路径")
    parser.add_argument("--output", type=str, default="./results/val_llm_fintune_result.jsonl", help="输出 JSONL 文件路径")
    parser.add_argument("--model_name", type=str, default="./model_weights/Qwen/Qwen3-0.6B", help="基础模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA 权重路径（可选）")
    parser.add_argument("--batch_size", type=int, default=64, help="一次推理的批量大小")
    parser.add_argument("--sample_size", type=int, default=1280, help="随机抽样验证条数")
    # 生成相关参数
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="生成的最大 token 数")
    parser.add_argument("--min_new_tokens", type=int, default=0, help="生成的最小 token 数")
    parser.add_argument("--do_sample", type=bool, default=False, help="是否开启采样生成")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p 采样的累积概率阈值")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚系数")

    args = parser.parse_args()

    # batch_process(args.input, args.output, args.model_name, args.lora_path, args.batch_size, args.sample_size)

    batch_process(
        input_jsonl=args.input,
        output_jsonl=args.output,
        model_name=args.model_name,
        lora_path=args.lora_path,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty
    )

