import json
import os
import torch
from unsloth import FastVisionModel
from qwen_vl_utils import process_vision_info

TEST_IMAGE_PATH = "/home/featurize/kaggle/competitions/museumscat-specimen-collection-annotation-task/images/1705416.jpeg"

if not os.path.exists(TEST_IMAGE_PATH):
  print(
      f"❌ 请先修改 TEST_IMAGE_PATH 为有效的测试图片路径！当前路径不存在: {TEST_IMAGE_PATH}"
  )
  exit(1)

# 2. 加载训练好的 LoRA 模型
print("正在加载训练好的 LoRA 模型...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="qwen2_vl_specimen_lora",  # 训练保存的文件夹路径
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

# 3. 构建提示词（必须与训练时的 Prompt 保持一致）
instruction = (
    "Extract 'verbatimDate' and 'verbatimLocality' from this specimen label as"
    " JSON. Preserve exact characters."
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": TEST_IMAGE_PATH},
        {"type": "text", "text": instruction},
    ],
}]

# 4. 预处理输入
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = tokenizer(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# 5. 模型推理生成
print("开始推理抽取...")
with torch.no_grad():
  generated_ids = model.generate(
      **inputs,
      max_new_tokens=256,
      use_cache=True,
      temperature=0.1,  # 降低随机性，提高提取精准度
  )

# 截取新生成的 Token 文本
generated_ids_trimmed = [
    out_ids[len(in_ids) :]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = tokenizer.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print("\n" + "=" * 40)
print("📌 模型推理输出结果：")
print("=" * 40)
print(output_text)
print("=" * 40)