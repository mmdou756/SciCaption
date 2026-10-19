# sciCaption: 论文图注生成工具

本项目旨在基于大语言模型（LLM）与多模态大语言模型（MLLM）自动生成科研论文图表标题（caption），支持零样本推理、微调训练及多模态融合微调流程。

---

## 项目结构

```

.
├── dataset/                # 原始数据和训练/验证/测试集划分
├── dataset_process.py      # 数据转换、划分与预处理脚本
├── download_model.py       # 下载预训练模型或权重文件
├── llm_zero_shot.py        # LLM 零样本推理脚本
├── mllm_zero_shot.py       # MLLM 零样本推理脚本
├── fintune_llm.py          # LLM 微调训练主程序
├── inference_llm.py        # 单条或批量推理脚本
├── evaluate_caption.py     # 生成图注的评估脚本
├── model_weights/          # 存放训练或下载的模型权重
├── output/                 # 中间输出、日志或模型检查点
├── results/                # 存放评估结果和生成的图注
├── readme.md               # 项目说明与使用指南
├── requirements.txt        # Python 依赖包列表
├── utils.py                # 公共工具函数（数据处理、日志、评估等）

```

---

## 0. 环境配置

- Python ≥ 3.10
- 同项目4 LLM微调


* 国内访问 Hugging Face 仓库可能需要配置代理：

```bash
source /etc/network_turbo
```

* 安装依赖：

```bash
pip install -r requirements.txt
```

依赖包括：`transformers`, `torch`, `peft`, `datasets`, `rouge_score` 等。

* 配置 DashScope API Key（推理与评估脚本使用 `qwen-plus` / `qwen3-vl-plus`）：

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

也可以复制 `.env.example` 为 `.env` 并填写，脚本会自动读取。

---

## 1. 数据准备与预处理

### 数据集

* 本地调试使用 1000 条示例图片与文本对；图片与标注体积较大，未随本仓库分发，可按下文方式下载。
* 大规模数据集来源：arXiv、ACL 论文，包含：

  * 训练集 33.3 万条
  * 验证集 4.7 万条
  * 测试集 4.7 万条

Hugging Face 下载示例：

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="CrowdAILab/scicap", repo_type='dataset')
```

### 数据格式

* 原始 JSON 转换为 JSONL，每行一个样本：

```json
{
  "image_id": 501710,
  "file_name": "000000501710.png",
  "caption_no_index": "The sensor selection algorithm at the Fusion Centre.",
  "paragraph": ["For example, in Fig. 3 the source sequence is covered by the subset consisting of the first, second ..."],
  "mention": [
    ["The implementation of the selection algorithm at the FC is shown in Fig. 4.", "As shown in Fig. 4, if no subset is covering the source sequence, the FC requests transmission ..."],
    ["However, as shown in Fig. 4, in these cases,..."]
  ],
  "ocr": ["Receive CSls", "From sensors", "eamata", "Request transmissicn", "Jkat (erallkte", "irom=", "Isensors"]
}
```

字段说明：

| 字段                 | 说明                                 |
| ------------------ | ---------------------------------- |
| `file_name`        | 图像文件名                              |
| `image_id`         | 图像唯一标识                             |
| `figure_type`      | 图表类型（如 Graph Plot、Flowchart、Table） |
| `ocr`              | 图表 OCR 文本，可作为辅助信息                  |
| `caption`          | 原始论文图注                             |
| `caption_no_index` | 去掉编号的图注                            |
| `paragraph`        | 图表所在段落文本，提供上下文信息                   |
| `mention`          | 文中引用图表的句子集合                        |

### 数据预处理

```bash
python dataset_process.py
```

该脚本会将原始 JSON 转换为训练/验证/测试所需的 JSONL 格式。

---

## 2. 模型权重下载

项目默认使用预训练模型 `Qwen/Qwen3-0.6B`：

```bash
python download_model.py
```

模型文件结构示例：

```
├── model.safetensors      # 模型权重 (~1.5G)
├── config.json
├── configuration.json
├── generation_config.json
├── merges.txt
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```

---

## 3. 图注生成评价

使用 LLM 对生成的图注进行自动化评分。这里我们直接使用给的label进行验证。

```bash
python3 evaluate_caption.py \
  --input dataset/val.jsonl \
  --output results/val_scored.jsonl \
  --caption_field caption_no_index \
  --mode zero_shot \
  --model qwen-plus
```

参数说明：

| 参数                | 说明                            |
| ----------------- | ----------------------------- |
| `--input`         | 输入 JSONL 文件路径                 |
| `--output`        | 输出评分结果 JSONL 文件路径             |
| `--caption_field` | 待评估字段                         |
| `--mode`          | 评分模式：`zero_shot` 或 `few_shot` |
| `--model`         | 使用模型名称                        |
| `--sleep`         | API 请求间隔（秒）                   |

---

## 4. LLM 零样本推理

构造包含图像上下文、段落文本、mention 与 OCR 信息的 prompt，直接生成学术风格英文图注。

```bash
python llm_zero_shot.py \
        --input_jsonl ./dataset/val.jsonl \
        --output_jsonl  ./results/val_llm_zero_shot_result.jsonl
```

验证生成结果：

```bash
python3 evaluate_caption.py \
  --input results/val_llm_zero_shot_result.jsonl \
  --output results/val_llm_zero_shot_result_scored.jsonl \
  --caption_field generated_caption \
  --mode zero_shot \
  --model qwen-plus
```
==========================
Total samples: 300
Average score: 5.15
==========================
---

## 5. LLM 微调训练

基于已有图注进行微调训练，提高生成准确性与一致性。

```bash
python fintune_llm.py \
    --model_path "Qwen/Qwen3-0.6B" \
    --data_path "./dataset/train.jsonl" \
    --output_dir "./output" \
    --batch_size 12 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --epochs 3 \
    --save_steps 50 \
    --learning_rate 1e-4 \
    --use_lora True \
    --prompt_type "text"
```

微调参数说明：

| 参数                              | 说明                       |
| ------------------------------- | ------------------------ |
| `--model_path`                  | 预训练模型路径                  |
| `--data_path`                   | JSONL 训练数据路径             |
| `--output_dir`                  | 模型保存路径                   |
| `--batch_size`                  | 每设备 batch 大小             |
| `--gradient_accumulation_steps` | 梯度累积步数                   |
| `--epochs`                      | 训练轮数                     |
| `--learning_rate`               | 学习率                      |
| `--use_lora`                    | 是否启用 LoRA 微调             |
| `--prompt_type`                 | prompt 类型（text / mllm 等） |

### 推理

```bash
python3 inference_llm.py \
    --input ./dataset/val.jsonl \
    --output ./results/val_llm_fintune_result.jsonl \
    --lora_path ./output/checkpoint-45 \
    --model_name ./model_weights/Qwen/Qwen3-0.6B
```

验证：

```bash
python evaluate_caption.py \
  --input results/val_llm_fintune_result.jsonl \
  --output results/val_llm_fintune_result_scored.jsonl \
  --caption_field generated_caption \
  --mode zero_shot \
  --model qwen-plus
```

---

## 6. MLLM 零样本推理

考虑到文本中存在大量干扰信息，可能影响生成质量，因此该方法仅依赖图像本身。通过构造针对多模态大语言模型（MLLM）的 prompt（zero-shot、few-shot 或 chain-of-thought），直接生成学术风格英文图注，实现纯视觉驱动的自动化图注生成。

```bash
python mllm_zero_shot.py \
    --input_jsonl ./dataset/val.jsonl \
    --output_jsonl ./results/val_mllm_zero_shot_result.jsonl
```

验证：

```bash
python evaluate_caption.py \
  --input results/val_mllm_zero_shot_result.jsonl \
  --output results/val_mllm_zero_shot_result_scored.jsonl \
  --caption_field generated_caption \
  --mode zero_shot \
  --model qwen-plus
```

---

## 7. MLLM + LLM 微调流程

1. 使用 MLLM 对图像生成初步图注。
2. 将生成图注与原始段落文本拼接，用于二次微调 LLM。
3. 微调后的 LLM 可生成更规范、一致的学术图注。

### 数据整理与微调

```bash
python mllm_zero_shot.py \
    --input_jsonl ./dataset/train.jsonl \
    --output_jsonl ./results/train_mllm_zero_shot_result.jsonl
```

```bash
python fintune_llm.py \
    --model_path Qwen/Qwen3-0.6B \
    --data_path ./results/train_mllm_zero_shot_result.jsonl \
    --output_dir ./output \
    --batch_size 12 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --epochs 5 \
    --save_steps 100 \
    --learning_rate 1e-4 \
    --use_lora True \
    --prompt_type mllm
```

### 推理与验证

```bash
python3 inference_llm.py \
    --input ./dataset/val.jsonl \
    --output ./results/val_mllm_fintune_result.jsonl \
    --lora_path ./output/checkpoint-45 \
    --model_name ./model_weights/Qwen/Qwen3-0.6B
```

```bash
python evaluate_caption.py \
  --input results/val_mllm_fintune_result.jsonl \
  --output results/val_mllm_fintune_result_scored.jsonl \
  --caption_field generated_caption \
  --mode zero_shot \
  --model qwen-plus
```