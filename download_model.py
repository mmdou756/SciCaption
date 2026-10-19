import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = './model_weights'
#模型下载
from modelscope import snapshot_download
# model_dir = snapshot_download('Qwen/Qwen3-32B', cache_dir='./model_weights', revision='master')


# #模型下载
model_dir = snapshot_download('Qwen/Qwen3-0.6B', cache_dir='./model_weights', revision='master')

print("下载完成，开始尝试加载...")
# 尝试加载模型
model_path = './model_weights/Qwen/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B",
        use_fast=False,
        trust_remote_code=True
    )

text = "This is my QQ number: 420889338"

# 1. 查看分词结果
tokens = tokenizer.tokenize(text)
print("分词结果 tokens：", tokens)

# 2. 转换为 token id
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("对应的 token IDs：", token_ids)

# 3. 直接用 tokenizer 编码（包含特殊 token，如 bos/eos）
encoded = tokenizer(text)
print("直接编码结果（包含 input_ids 和 attention_mask）：", encoded)
'''
分词结果 tokens： ['This', 'Ġis', 'Ġmy', 'ĠQQ', 'Ġnumber', ':', 'Ġ', '4', '2', '0', '8', '8', '9', '3', '3', '8']
对应的 token IDs： [1986, 374, 847, 73115, 1372, 25, 220, 19, 17, 15, 23, 23, 24, 18, 18, 23]
直接编码结果（包含 input_ids 和 attention_mask）： {'input_ids': [1986, 374, 847, 73115, 1372, 25, 220, 19, 17, 15, 23, 23, 24, 18, 18, 23], 'attentio
'''
# Ġ 为 前面有空格 的标记。
# 4.将 token id 变换为 文本
decoded_text = tokenizer.decode(token_ids)
print(decoded_text)  

# 输出：This is my QQ number: 420889338

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # 使用bfloat16混合精度
    device_map="auto"            # 自动设备分配
)
