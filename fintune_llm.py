import torch
import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
from utils import generate_scicaption_prompt_tmp0, generate_scicaption_prompt_mllm_tmp0


def initialize_model(model_path: str, use_lora: bool = True):
    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # 使用bfloat16混合精度
        device_map="auto"            # 自动设备分配
    )

    # 配置LoRA适配器
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=[  # 覆盖关键注意力模块
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            inference_mode=False,
            r=8,                  # LoRA秩（平衡效果与效率）
            lora_alpha=32,         # 缩放因子（一般=2r 或 4r）
            lora_dropout=0.1,       # 正则化比率
        )
        model = get_peft_model(model, lora_config)

    # 启用梯度检查点（显存优化）
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    print(model.print_trainable_parameters()
)
    return model, tokenizer

def preprocess_data(file_path, sample_size=None, DEFAULT_SEED=42):
    raw_dataset = load_dataset('json', data_files=file_path)['train']
    # import pdb;pdb.set_trace()
    '''
    Dataset({
        features: ['review_body', 'review_title'],
        num_rows: 15000
    })
    (Pdb) raw_dataset[0]
    {'review_body': 'These are AWFUL. They are see through, the fabric feels like tablecloth, and they fit like children’s clothing. Customer service did seem to be nice though, but I regret missing my return date for these. I wouldn’t even donate them because the quality is so poor.', 'review_title': 'Don’t waste your time!'}
    '''
    # 数据采样与随机化
    if sample_size:
        sample_size = min(len(raw_dataset), sample_size)
        raw_dataset = raw_dataset.shuffle(seed=DEFAULT_SEED).select(range(sample_size))
    return raw_dataset


def tokenize_function(args, tokenizer, examples):
    """
    将文本数据转换为模型所需的输入格式
    
    处理逻辑:
    1. 构建系统提示和用户指令
    2. 将标签转换为文本形式
    3. 拼接指令和响应生成完整序列
    4. 处理长度限制和特殊token
    
    参数:
        examples (dict): 包含'review_body'和'review_title'的样本字典
        例子： {'review_body': '刚收到货，烧水时有嗡嗡的声音，不知道是不是产品缺陷。', 'review_title': '烧水时发出较大嗡嗡的声音'}
        
    返回:
        dict: 包含input_ids, attention_mask, labels的字典
    """
    MAX_LENGTH = 1024  # 模型最大上下文长度
    
    # ref https://github.com/QwenLM/Qwen/blob/main/finetune.py
    # 构建指令部分
    if args.prompt_type == 'text':
        instruction = tokenizer(
            generate_scicaption_prompt_tmp0(examples),
            add_special_tokens=False  # 禁用自动添加特殊token
        )
    else:
        instruction = tokenizer(
            generate_scicaption_prompt_mllm_tmp0(examples),
            add_special_tokens=False  # 禁用自动添加特殊token
        )
    '''
    (Pdb) generate_scicaption_prompt_tmp0(examples)
    '<|im_start|>system\nYou are a scientific writing assistant. You will be given textual information related to a research figure, including paragraphs, mentions, and OCR text. Based on this information, generate a single accurate, concise, and academically styled English figure caption. Do not include figure numbers (e.g., Fig. 4). Do not add explanations. Output only the caption text.<|im_end|>\n<|im_start|>user\nParagraph:\n[\'The symbol = ? signifies that the equation must be solved rather than having to hold for all values of variables. The purpose of σ P is for the unification algorithm (Section 3.2) to record a partial solution. An illustration of this concept will be seen in Figure 8.\', "For the selector algorithm and proof see Section 4. For example, consider the unification problem y : 2 = a : x and δ = (x → 1 : 2, y → a : 1) as a unifier. The unification tree was shown in Figure 8. Select(δ , y : 2 = a : x) would produce selections (Subst3, Decomp2\', Orient1, Subst1), which corresponds to the right-most path in the tree. The unifier at the end of this path is σ = (x → y ′ : 2, y → a : y ′ ) which is more general than δ by instantiation λ = (y ′ → 1). Now we are able to state our completeness result, which follows directly from Lemma 5."]\n\nMention:\n[[\'An illustration of this concept will be seen in Figure 8.\'], [\'The unification tree was shown in Figure 8.\']]\n\nOCR:\n[\'Lioi\', \'~v\']\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    (Pdb) instruction
    {'input_ids': [151644, 8948, 198, 2610, 525, 264, 12344, 4378, 17847, 13, 1446, 686, 387, 2661, 62533, 1995, 5435, 311, 264, 3412, 7071, 11, 2670, 42643, 11, 33845, 11, 323, 80577, 1467, 13, 20205, 389, 419, 1995, 11, 6923, 264, 3175, 13382, 11, 63594, 11, 323, 10406, 336, 2673, 11096, 6364, 7071, 17256, 13, 3155, 537, 2924, 7071, 5109, 320, 68, 1302, 2572, 23095, 13, 220, 19, 568, 3155, 537, 912, 40841, 13, 9258, 1172, 279, 17256, 1467, 13, 151645, 198, 151644, 872, 198, 42165, 510, 677, 785, 7735, 284, 937, 77764, 429, 279, 23606, 1969, 387, 27956, 4751, 1091, 3432, 311, 3331, 369, 678, 2750, 315, 7332, 13, 576, 7428, 315, 47723, 393, 374, 369, 279, 650, 2404, 12111, 320, 9620, 220, 18, 13, 17, 8, 311, 3255, 264, 7130, 6291, 13, 1527, 39034, 315, 419, 7286, 686, 387, 3884, 304, 19037, 220, 23, 15670, 330, 2461, 279, 9367, 12111, 323, 11064, 1490, 11113, 220, 19, 13, 1752, 3110, 11, 2908, 279, 650, 2404, 3491, 379, 549, 220, 17, 284, 264, 549, 856, 323, 69334, 284, 320, 87, 11397, 220, 16, 549, 220, 17, 11, 379, 11397, 264, 549, 220, 16, 8, 438, 264, 650, 3049, 13, 576, 650, 2404, 4916, 572, 6839, 304, 19037, 220, 23, 13, 8427, 7, 85386, 1154, 379, 549, 220, 17, 284, 264, 549, 856, 8, 1035, 8193, 37399, 320, 3136, 267, 18, 11, 62378, 17, 516, 61474, 16, 11, 3719, 267, 16, 701, 892, 33210, 311, 279, 1290, 62398, 1815, 304, 279, 4916, 13, 576, 650, 3049, 518, 279, 835, 315, 419, 1815, 374, 47723, 284, 320, 87, 11397, 379, 636, 110, 549, 220, 17, 11, 379, 11397, 264, 549, 379, 636, 110, 873, 892, 374, 803, 4586, 1091, 69334, 553, 81128, 48338, 284, 320, 88, 636, 110, 11397, 220, 16, 568, 4695, 582, 525, 2952, 311, 1584, 1039, 79314, 1102, 11, 892, 11017, 5961, 504, 85810, 220, 20, 1189, 2533, 44, 2939, 510, 56330, 2082, 39034, 315, 419, 7286, 686, 387, 3884, 304, 19037, 220, 23, 13, 4089, 2509, 785, 650, 2404, 4916, 572, 6839, 304, 19037, 220, 23, 98758, 2533, 93495, 510, 677, 43, 815, 72, 516, 63470, 85, 4432, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    '''
    # 构建响应部分（真实标签）
    response = tokenizer(
        f"{examples['caption_no_index']}", 
        add_special_tokens=False
    )
    '''
    (Pdb) response
    {'input_ids': [100228, 52510, 13343, 104351, 104590, 117377, 117377, 104668], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
    '''
    # 将instructio部分和response部分的input_ids拼接，并在末尾添加eos token作为标记结束的token
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    '''
    (Pdb) input_ids
    [151644, 8948, 198, 56568, 44063, 101038, 46944, 101919, 110514, 109348, 85641, 11, 220, 14880, 100345, 85641, 43815, 3837, 43959, 220, 16, 4142, 17, 26853, 98, 9370, 98237, 99534, 102050, 90395, 100662, 57218, 52129, 85641, 107668, 102064, 9909, 104811, 5373, 8903, 16744, 57191, 105205, 74276, 31196, 45943, 85641, 5122, 151645, 198, 151644, 872, 198, 99900, 104381, 81668, 3837, 100228, 52510, 13343, 18830, 117377, 117377, 104668, 3837, 103950, 104074, 82700, 106136, 1773, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271, 100228, 52510, 13343, 104351, 104590, 117377, 117377, 104668, 151643]
    '''
    # 注意力掩码，表示模型需要关注的位置
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    '''
    (Pdb) attention_mask
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    '''
    # 对于instruction，使用-100表示这些位置不计算loss（即模型不需要预测这部分）
    labels = [-100]*len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    '''
    (Pdb) labels
    [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 100228, 52510, 13343, 104351, 104590, 117377, 117377, 104668, 151643]
    '''
    # 长度截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    # import pdb;pdb.set_trace()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def train(model, tokenizer, dataset, args):
    """
    参数：
        model: 待训练模型
        tokenizer: 文本处理器
        dataset: 训练数据集
        args: argparse解析后的训练超参数
    """
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,       # 根据显存调整
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累积（模拟更大batch）
        logging_steps=args.logging_steps,                  # 监控训练状态
        num_train_epochs=args.epochs,                      # 训练轮数
        save_steps=args.save_steps,                        # 检查点间隔
        learning_rate=args.learning_rate,                  # 学习率（LoRA典型值）
        # save_on_each_node=True,                            # 分布式训练设置
        gradient_checkpointing=True,                       # 启用梯度检查点
        bf16=True,                                         # 使用bfloat16混合精度训练
        remove_unused_columns=False,                       # 保留未使用列（用于标签处理）
        optim="adamw_torch",                               # 优化器
        report_to="none",
        logging_dir="./logs"
    )

    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    # 启动训练流程
    trainer.train()
    
    # 保存最终模型
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-0.6B", help="预训练模型路径")
    parser.add_argument("--data_path", type=str, default="./dataset/train.jsonl", help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="模型保存路径")
    parser.add_argument("--batch_size", type=int, default=12, help="每个设备的训练batch大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--save_steps", type=int, default=50, help="保存检查点间隔")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--sample_size", type=int, default=None, help="数据采样大小，可选")
    parser.add_argument("--use_lora", type=bool, default=True, help="是否启用LoRA微调")
    parser.add_argument("--prompt_type", type=str, default='text', help="训练使用的prompt")

    args = parser.parse_args()


    # 模型 和 tokenizer
    model, tokenizer = initialize_model(args.model_path, use_lora=args.use_lora)

    # 加载数据
    dataset = preprocess_data(args.data_path, sample_size=args.sample_size)

    # 应用分词处理
    tokenized_dataset = dataset.map(lambda examples: tokenize_function(args, tokenizer, examples), remove_columns=dataset.column_names)

    # 启动训练
    train(model, tokenizer, tokenized_dataset, args)
