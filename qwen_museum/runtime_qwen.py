from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator
import json
import os
import torch
from datasets import Dataset
from PIL import Image
from trl import SFTTrainer


# =========================================================
# 1. 加载模型并初始化 QLoRA Adapter (4-bit 低显存模式)
# =========================================================
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",  # 4-bit 预量化基座
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# 挂载 LoRA 权重，同时微调 Vision 模块与 Language 模块
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,  # 允许提取历史图像视觉特征
    finetune_language_layers=True,  # 允许增强丹麦语/特殊字符生成能力
    r=16,  # LoRA Rank
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# =========================================================
# 2. 从 train.jsonl 构建标准的 Qwen2-VL 对话数据集
# =========================================================
instruction = (
    "Extract 'verbatimDate' and 'verbatimLocality' from this specimen label as"
    " JSON. Preserve exact characters."
)


def load_and_convert_jsonl(jsonl_path):
    dataset_list = []
    # 设置图片所在的目录路径
    image_base_path = "/home/featurize/kaggle/competitions/museumscat-specimen-collection-annotation-task/images/"
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # 构建完整的图片路径
            image_path = os.path.join(image_base_path, data["file_name"])
            # 构建符合 VLM 训练的标准 Chat 格式
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},  # 使用完整路径
                        {"type": "text", "text": instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": data["ground_truth"]}],
                },
            ]
            dataset_list.append({"messages": conversation})
    return Dataset.from_list(dataset_list)


raw_dataset = load_and_convert_jsonl("train.jsonl")

# =========================================================
# 3. 配置 SFTTrainer 训练器
# =========================================================
FastVisionModel.for_training(model)  # 开启 Training 模式

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,  # 兼容新版 trl
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=raw_dataset,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    dataset_num_proc=2,
    max_seq_length=1024,
    args=dict(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=15,
        max_steps=350,  
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        output_dir="qwen2_vl_specimen_output",
        report_to="tensorboard",
        logging_dir="qwen2_vl_specimen_output/runs",
        optim="adamw_8bit",
        seed=3407,
    ),
)

# 启动微调
trainer_stats = trainer.train()

model.save_pretrained("qwen2_vl_specimen_lora")
tokenizer.save_pretrained("qwen2_vl_specimen_lora")
print("LoRA 权重保存成功！")
print("微调完成并成功保存 LoRA 权重！")