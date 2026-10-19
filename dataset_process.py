import os
import json

# 输入文件路径
input_dir = "./dataset"
input_files = ["train.json", "val.json"]

# 输出文件路径
output_files = {
    "train.json": os.path.join(input_dir, "train.jsonl"),
    "val.json": os.path.join(input_dir, "val.jsonl"),
}

def process_json_file(json_path, output_jsonl):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 建立 image_id → image_info 的映射
    image_dict = {}
    for img in data.get("images", []):
        image_dict[img["id"]] = {
            "file_name": img.get("file_name", ""),
            "ocr": img.get("ocr", [])
        }

    # 写入对应 jsonl 文件
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for ann in data.get("annotations", []):
            image_id = ann.get("image_id")
            image_info = image_dict.get(image_id, {})

            record = {
                "image_id": image_id,
                "file_name": image_info.get("file_name", ""),
                "caption_no_index": ann.get("caption_no_index", ""),
                "paragraph": ann.get("paragraph", []),
                "mention": ann.get("mention", []),
                "ocr": image_info.get("ocr", [])
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Saved:", output_jsonl)


# 分别处理 train 和 val
for file_name in input_files:
    json_path = os.path.join(input_dir, file_name)
    output_jsonl = output_files[file_name]
    print("Processing:", json_path)
    process_json_file(json_path, output_jsonl)

print("All files processed.")
