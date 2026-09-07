import csv
import json
import os
import re
import time
import torch
import torch.nn.functional as F
from unsloth import FastVisionModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import glob
import pandas as pd
from PIL import Image
import sys

# ============ 配置参数 ============
IMAGE_DIR = "/home/featurize/kaggle/competitions/museumscat-specimen-collection-annotation-task/images/"
TEST_CSV_PATH = os.path.join(IMAGE_DIR, "test.csv")
IMAGE_PATTERNS = ["*.jpeg", "*.jpg", "*.png"]
OUTPUT_JSON_PATH = "batch_results.json"
OUTPUT_CSV_PATH = "batch_results.csv"
BATCH_SIZE = 8
SAVE_EVERY_N_BATCHES = 1

FORCE_REPROCESS = True  # 强制重新处理

FIELDS = ["verbatimDate", "verbatimLocality"]
MISSING_VALUE = "MISSING"
MISSING_CONFIDENCE = 0.0

# 🆕 调试配置
DEBUG_MODE = True  # 开启调试模式
DEBUG_LOG_FILE = "inference_debug.log"  # 调试日志文件
PROCESS_ONLY_N_IMAGES = None  # 设置为数字可限制处理图片数量，如10，None表示全部
LOG_IMAGE_SIZES = True  # 记录每张图片的尺寸


class DebugTimer:
    """调试计时器类"""
    def __init__(self, name, log_file=None):
        self.name = name
        self.start_time = None
        self.log_file = log_file
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        msg = f"⏱️  {self.name}: {elapsed:.3f}秒"
        print(msg)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{msg}\n")


def log_debug(msg, log_file=DEBUG_LOG_FILE):
    """写入调试日志"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{full_msg}\n")


def get_image_info(img_path):
    """获取图片详细信息"""
    try:
        img = Image.open(img_path)
        width, height = img.size
        mode = img.mode
        format = img.format
        file_size = os.path.getsize(img_path) / (1024 * 1024)  # MB
        return {
            'width': width,
            'height': height,
            'mode': mode,
            'format': format,
            'size_mb': file_size,
            'pixels': width * height
        }
    except Exception as e:
        return {'error': str(e)}


def extract_json(text):
    """从模型输出中提取 JSON"""
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    return None


def get_image_paths_from_csv(csv_path, image_dir):
    """从CSV读取图片路径"""
    if not os.path.exists(csv_path):
        log_debug(f"❌ CSV文件不存在: {csv_path}")
        return []
    
    try:
        df = pd.read_csv(csv_path)
        
        possible_columns = ['image_path', 'filename', 'image', 'file_path', 'path', 'image_name']
        image_col = None
        
        for col in possible_columns:
            if col in df.columns:
                image_col = col
                break
        
        if image_col is None:
            image_col = df.columns[0]
            log_debug(f"⚠️  未找到图片路径列，使用第一列: {image_col}")
        
        image_paths = df[image_col].astype(str).tolist()
        
        full_paths = []
        for path in image_paths:
            path = path.strip().strip('"').strip("'")
            if not os.path.isabs(path) and not path.startswith(image_dir):
                full_path = os.path.join(image_dir, path)
            else:
                full_path = path
            full_paths.append(full_path)
        
        log_debug(f"✅ 从CSV读取了 {len(full_paths)} 个图片路径")
        return full_paths
        
    except Exception as e:
        log_debug(f"❌ 读取CSV文件失败: {e}")
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                image_paths = []
                for row in reader:
                    if row:
                        path = row[0].strip().strip('"').strip("'")
                        if not os.path.isabs(path) and not path.startswith(image_dir):
                            full_path = os.path.join(image_dir, path)
                        else:
                            full_path = path
                        image_paths.append(full_path)
            log_debug(f"✅ 从CSV读取了 {len(image_paths)} 个图片路径")
            return image_paths
        except Exception as e2:
            log_debug(f"❌ 备用方法也失败: {e2}")
            return []


def get_image_paths(image_dir, patterns):
    """扫描目录获取图片"""
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    return sorted(image_paths)


def load_existing_progress(json_path, force_reprocess=False):
    """加载已有进度"""
    if force_reprocess:
        log_debug("🔄 强制重新处理模式：忽略已有进度文件")
        return [], set()
    
    if not os.path.exists(json_path):
        return [], set()

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        processed = {r["image_path"] for r in results if "image_path" in r}
        log_debug(f"📂 加载已有进度: {len(results)} 条结果, {len(processed)} 张已处理")
        return results, processed
    except (json.JSONDecodeError, OSError, KeyError) as e:
        log_debug(f"⚠️  读取已有进度文件失败（{e}），将从头开始处理")
        return [], set()


def csv_rows_from_results(results):
    """从results生成CSV行"""
    rows = []
    for r in results:
        fields = r.get("fields", {})
        rows.append({
            "image_file": r.get("image_name", ""),
            "verbatimDate": fields.get("verbatimDate", {}).get("value", MISSING_VALUE),
            "verbatimDate_confidence": fields.get("verbatimDate", {}).get("confidence", MISSING_CONFIDENCE),
            "verbatimLocality": fields.get("verbatimLocality", {}).get("value", MISSING_VALUE),
            "verbatimLocality_confidence": fields.get("verbatimLocality", {}).get("confidence", MISSING_CONFIDENCE),
        })
    return rows


def build_char_spans(tokenizer, token_ids):
    """构建字符跨度"""
    spans = []
    prev_text = ""
    ids_so_far = []
    for tid in token_ids:
        ids_so_far.append(tid)
        cur_text = tokenizer.decode(
            ids_so_far, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        start, end = len(prev_text), len(cur_text)
        spans.append((start, end))
        prev_text = cur_text
    return spans, prev_text


def find_field_container(parsed_json):
    """查找字段容器"""
    if not isinstance(parsed_json, dict):
        return None
    if any(f in parsed_json for f in FIELDS):
        return parsed_json
    for v in parsed_json.values():
        found = find_field_container(v)
        if found is not None:
            return found
    return None


def field_value_and_confidence(full_text, spans, token_probs, field_name, field_container):
    """计算字段值和置信度"""
    if not isinstance(field_container, dict) or field_name not in field_container or field_container[field_name] in (None, ""):
        return MISSING_VALUE, MISSING_CONFIDENCE

    value = str(field_container[field_name])

    pattern = re.compile(r'"' + re.escape(field_name) + r'"\s*:\s*"((?:\\.|[^"\\])*)"')
    match = pattern.search(full_text)

    if match:
        vstart, vend = match.start(1), match.end(1)
    else:
        idx = full_text.find(value)
        if idx == -1:
            return value, MISSING_CONFIDENCE
        vstart, vend = idx, idx + len(value)

    covering_probs = [p for (s, e), p in zip(spans, token_probs) if e > vstart and s < vend]
    if not covering_probs:
        return value, MISSING_CONFIDENCE

    confidence = sum(covering_probs) / len(covering_probs)
    return value, round(float(confidence), 4)


def save_json(results, image_dir, json_path):
    """保存JSON结果"""
    success_count = sum(1 for r in results if r.get("result") is not None)
    output_data = {
        "metadata": {
            "total_images": len(results),
            "success_count": success_count,
            "failed_count": len(results) - success_count,
            "image_directory": image_dir,
        },
        "results": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def save_csv(rows, csv_path):
    """保存CSV结果"""
    fieldnames = ["image_file", "verbatimDate", "verbatimDate_confidence",
                  "verbatimLocality", "verbatimLocality_confidence"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process_batch(model, tokenizer, image_paths, instruction, batch_size=4,
                   json_path=OUTPUT_JSON_PATH, csv_path=OUTPUT_CSV_PATH,
                   save_every=SAVE_EVERY_N_BATCHES, image_dir="",
                   results=None, csv_rows=None):
    """处理批次"""
    results = results if results is not None else []
    csv_rows = csv_rows if csv_rows is not None else []

    # 🆕 统计信息
    stats = {
        'total_load_time': 0,
        'total_token_time': 0,
        'total_infer_time': 0,
        'total_post_time': 0,
        'total_images': 0,
        'image_sizes': [],
        'slow_images': [],
    }

    for batch_idx, i in enumerate(tqdm(range(0, len(image_paths), batch_size), desc="处理批次")):
        batch_paths = image_paths[i:i + batch_size]
        
        # ====== 阶段1: 图片加载和消息构建 ======
        with DebugTimer(f"批次{batch_idx} - 图片加载", DEBUG_LOG_FILE):
            t1_start = time.time()
            batch_messages = []
            for img_path in batch_paths:
                # 记录图片信息
                if LOG_IMAGE_SIZES:
                    img_info = get_image_info(img_path)
                    stats['image_sizes'].append({
                        'path': os.path.basename(img_path),
                        'info': img_info
                    })
                    if DEBUG_MODE:
                        log_debug(f"📷 {os.path.basename(img_path)}: {img_info.get('width')}x{img_info.get('height')}, "
                                 f"{img_info.get('size_mb', 0):.2f}MB, {img_info.get('format')}")
                
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": instruction},
                    ],
                }]
                batch_messages.append(messages)
            t1_end = time.time()
            load_time = t1_end - t1_start
            stats['total_load_time'] += load_time
        
        # ====== 阶段2: Tokenization ======
        with DebugTimer(f"批次{batch_idx} - Tokenization", DEBUG_LOG_FILE):
            t2_start = time.time()
            batch_texts, batch_images = [], []
            for messages in batch_messages:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, _ = process_vision_info(messages)
                batch_texts.append(text)
                batch_images.append(image_inputs[0] if image_inputs else None)
            
            inputs = tokenizer(
                text=batch_texts,
                images=batch_images,
                padding=True,
                return_tensors="pt",
                max_pixels=1280*28*28,  # 🆕 限制图像token数量
                min_pixels=256*28*28,
            )
            inputs = inputs.to("cuda")
            t2_end = time.time()
            token_time = t2_end - t2_start
            stats['total_token_time'] += token_time
            
            # 记录输入张量信息
            if DEBUG_MODE:
                log_debug(f"  📊 输入张量: input_ids shape={inputs.input_ids.shape}, "
                         f"pixel_values shape={inputs.pixel_values.shape if hasattr(inputs, 'pixel_values') else 'N/A'}")
        
        # ====== 阶段3: 模型推理 ======
        with DebugTimer(f"批次{batch_idx} - 模型推理", DEBUG_LOG_FILE):
            t3_start = time.time()
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    max_length=None,
                    use_cache=True,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            t3_end = time.time()
            infer_time = t3_end - t3_start
            stats['total_infer_time'] += infer_time

            # 记录生成token数量
            generated_ids = gen_out.sequences
            if DEBUG_MODE:
                for idx, ids in enumerate(generated_ids):
                    total_tokens = ids.shape[0]
                    new_tokens = total_tokens - inputs.input_ids[idx].shape[0]
                    log_debug(f"  📝 生成tokens: {new_tokens} (总tokens: {total_tokens})")
        
        # ====== 阶段4: 后处理 ======
        with DebugTimer(f"批次{batch_idx} - 后处理", DEBUG_LOG_FILE):
            t4_start = time.time()
            scores = gen_out.scores
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # 逐样本处理
            for row_idx, (img_path, trimmed_ids, output_text) in enumerate(
                zip(batch_paths, generated_ids_trimmed, output_texts)
            ):
                img_start = time.time()
                
                trimmed_ids_list = trimmed_ids.tolist()
                token_probs = []
                for step in range(len(trimmed_ids_list)):
                    step_logits = scores[step][row_idx]
                    step_prob = F.softmax(step_logits, dim=-1)[trimmed_ids_list[step]].item()
                    token_probs.append(step_prob)

                spans, rebuilt_text = build_char_spans(tokenizer, trimmed_ids_list)
                full_text = output_text if output_text else rebuilt_text

                try:
                    parsed_json = json.loads(output_text)
                    parse_error = None
                except json.JSONDecodeError:
                    parsed_json = extract_json(output_text)
                    parse_error = None if parsed_json is not None else "JSON解析失败"

                field_container = find_field_container(parsed_json)
                field_values = {}
                for field in FIELDS:
                    value, conf = field_value_and_confidence(
                        full_text, spans, token_probs, field, field_container
                    )
                    field_values[field] = (value, conf)

                results.append({
                    "image_path": img_path,
                    "image_name": os.path.basename(img_path),
                    "result": parsed_json,
                    "raw_output": output_text,
                    "fields": {k: {"value": v, "confidence": c} for k, (v, c) in field_values.items()},
                    **({"error": parse_error} if parse_error else {}),
                })

                csv_rows.append({
                    "image_file": os.path.basename(img_path),
                    "verbatimDate": field_values["verbatimDate"][0],
                    "verbatimDate_confidence": field_values["verbatimDate"][1],
                    "verbatimLocality": field_values["verbatimLocality"][0],
                    "verbatimLocality_confidence": field_values["verbatimLocality"][1],
                })
                
                stats['total_images'] += 1
                img_elapsed = time.time() - img_start
                
                # 🆕 记录慢图（超过10秒）
                if img_elapsed > 10:
                    stats['slow_images'].append({
                        'path': os.path.basename(img_path),
                        'time': img_elapsed,
                        'size': get_image_info(img_path)
                    })
                    log_debug(f"⚠️  慢图警告: {os.path.basename(img_path)} 处理耗时 {img_elapsed:.1f}秒")
            
            t4_end = time.time()
            post_time = t4_end - t4_start
            stats['total_post_time'] += post_time

        # 🆕 批次统计
        batch_total = load_time + token_time + infer_time + post_time
        log_debug(f"📊 批次 {batch_idx} 汇总: 总时间={batch_total:.2f}s, "
                 f"图片数={len(batch_paths)}, "
                 f"平均={batch_total/len(batch_paths):.2f}s/张")

        if (batch_idx + 1) % save_every == 0:
            save_json(results, image_dir, json_path)
            save_csv(csv_rows, csv_path)

    # 🆕 输出完整统计
    log_debug("\n" + "="*60)
    log_debug("📊 性能统计汇总:")
    log_debug(f"  ✅ 处理总图片数: {stats['total_images']}")
    log_debug(f"  ⏱️  总加载时间: {stats['total_load_time']:.2f}s")
    log_debug(f"  ⏱️  总Tokenization时间: {stats['total_token_time']:.2f}s")
    log_debug(f"  ⏱️  总推理时间: {stats['total_infer_time']:.2f}s")
    log_debug(f"  ⏱️  总后处理时间: {stats['total_post_time']:.2f}s")
    log_debug(f"  ⏱️  总时间: {sum([stats['total_load_time'], stats['total_token_time'], stats['total_infer_time'], stats['total_post_time']]):.2f}s")
    
    if stats['image_sizes']:
        sizes = [s['info'].get('pixels', 0) for s in stats['image_sizes'] if 'error' not in s['info']]
        if sizes:
            log_debug(f"  📐 图片像素统计: min={min(sizes):,}, max={max(sizes):,}, avg={sum(sizes)/len(sizes):,.0f}")
        
        formats = [s['info'].get('format', 'unknown') for s in stats['image_sizes'] if 'error' not in s['info']]
        if formats:
            from collections import Counter
            format_counts = Counter(formats)
            log_debug(f"  📁 图片格式分布: {dict(format_counts)}")
    
    if stats['slow_images']:
        log_debug(f"  ⚠️  慢图列表 (>10秒):")
        for slow in stats['slow_images']:
            log_debug(f"    - {slow['path']}: {slow['time']:.1f}s, {slow['size'].get('width')}x{slow['size'].get('height')}")
    
    log_debug("="*60 + "\n")

    return results, csv_rows


def main():
    # 清空或创建调试日志
    if os.path.exists(DEBUG_LOG_FILE):
        os.remove(DEBUG_LOG_FILE)
    log_debug("🚀 开始推理任务")
    log_debug(f"📂 工作目录: {os.getcwd()}")
    log_debug(f"🖥️  PyTorch版本: {torch.__version__}")
    log_debug(f"🖥️  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_debug(f"🖥️  GPU数量: {torch.cuda.device_count()}")
        log_debug(f"🖥️  当前GPU: {torch.cuda.get_device_name(0)}")
        log_debug(f"🖥️  显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 备份旧文件
    if FORCE_REPROCESS:
        if os.path.exists(OUTPUT_JSON_PATH):
            backup_name = f"{OUTPUT_JSON_PATH}.backup"
            os.rename(OUTPUT_JSON_PATH, backup_name)
            log_debug(f"📦 已备份旧结果: {backup_name}")
        if os.path.exists(OUTPUT_CSV_PATH):
            backup_name = f"{OUTPUT_CSV_PATH}.backup"
            os.rename(OUTPUT_CSV_PATH, backup_name)
            log_debug(f"📦 已备份旧CSV: {backup_name}")
    
    # 获取图片路径
    if not os.path.exists(TEST_CSV_PATH):
        log_debug(f"❌ test.csv 文件不存在: {TEST_CSV_PATH}")
        log_debug("将回退到扫描目录模式...")
        if not os.path.exists(IMAGE_DIR):
            log_debug(f"❌ 图片文件夹不存在: {IMAGE_DIR}")
            return
        all_image_paths = get_image_paths(IMAGE_DIR, IMAGE_PATTERNS)
        if not all_image_paths:
            log_debug(f"❌ 在 {IMAGE_DIR} 中未找到任何图片文件")
            return
    else:
        log_debug(f"📂 从CSV读取图片路径: {TEST_CSV_PATH}")
        all_image_paths = get_image_paths_from_csv(TEST_CSV_PATH, IMAGE_DIR)
        if not all_image_paths:
            log_debug(f"❌ 从CSV未读取到任何图片路径，尝试扫描目录...")
            all_image_paths = get_image_paths(IMAGE_DIR, IMAGE_PATTERNS)
            if not all_image_paths:
                log_debug(f"❌ 目录扫描也未找到图片")
                return

    # 🆕 限制处理数量（调试用）
    if PROCESS_ONLY_N_IMAGES:
        all_image_paths = all_image_paths[:PROCESS_ONLY_N_IMAGES]
        log_debug(f"🔬 调试模式: 只处理前 {PROCESS_ONLY_N_IMAGES} 张图片")

    log_debug(f"✅ 找到 {len(all_image_paths)} 张图片")
    if all_image_paths:
        log_debug(f"📷 示例图片: {all_image_paths[0]}")
        # 显示前5张图片信息
        for i, path in enumerate(all_image_paths[:5]):
            info = get_image_info(path)
            log_debug(f"  [{i+1}] {os.path.basename(path)}: {info.get('width')}x{info.get('height')}, "
                     f"{info.get('size_mb', 0):.2f}MB, {info.get('format')}")

    # 加载已有进度
    existing_results, processed_paths = load_existing_progress(
        OUTPUT_JSON_PATH, 
        force_reprocess=FORCE_REPROCESS
    )
    
    if FORCE_REPROCESS:
        image_paths = all_image_paths
        csv_rows = []
        log_debug(f"🔄 强制重新处理所有 {len(image_paths)} 张图片")
    else:
        image_paths = [p for p in all_image_paths if p not in processed_paths]
        csv_rows = csv_rows_from_results(existing_results)
        if processed_paths:
            log_debug(f"🔄 检测到已有进度: {len(processed_paths)} 张已处理，跳过")
            log_debug(f"➡️  剩余待处理: {len(image_paths)} 张")

    if not image_paths:
        log_debug("✅ 所有图片均已处理完毕，无需继续推理")
        save_json(existing_results, IMAGE_DIR, OUTPUT_JSON_PATH)
        save_csv(csv_rows, OUTPUT_CSV_PATH)
        return

    log_debug("正在加载训练好的 LoRA 模型...")
    load_start = time.time()
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="qwen2_vl_specimen_lora",
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    load_time = time.time() - load_start
    log_debug(f"✅ 模型加载完成，耗时: {load_time:.2f}秒")
    
    # 显示模型信息
    if DEBUG_MODE:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_debug(f"📊 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    instruction = (
        "Extract 'verbatimDate' and 'verbatimLocality' from this specimen label as"
        " JSON. Preserve exact characters."
    )

    log_debug("开始批量推理抽取...")
    log_debug(f"📝 中间结果会实时写入: {OUTPUT_JSON_PATH} / {OUTPUT_CSV_PATH}")
    log_debug(f"📦 批次大小: {BATCH_SIZE}")
    
    # 显存使用
    if torch.cuda.is_available():
        log_debug(f"💾 显存使用 (加载后): {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    if FORCE_REPROCESS:
        results, csv_rows = process_batch(
            model, tokenizer, image_paths, instruction, BATCH_SIZE,
            json_path=OUTPUT_JSON_PATH, csv_path=OUTPUT_CSV_PATH,
            save_every=SAVE_EVERY_N_BATCHES, image_dir=IMAGE_DIR,
            results=[], csv_rows=[],
        )
    else:
        results, csv_rows = process_batch(
            model, tokenizer, image_paths, instruction, BATCH_SIZE,
            json_path=OUTPUT_JSON_PATH, csv_path=OUTPUT_CSV_PATH,
            save_every=SAVE_EVERY_N_BATCHES, image_dir=IMAGE_DIR,
            results=existing_results, csv_rows=csv_rows,
        )

    success_count = sum(1 for r in results if r.get("result") is not None)
    log_debug(f"\n✅ 累计成功提取: {success_count}/{len(results)}")
    log_debug(f"❌ 累计失败: {len(results) - success_count}/{len(results)}")

    save_json(results, IMAGE_DIR, OUTPUT_JSON_PATH)
    save_csv(csv_rows, OUTPUT_CSV_PATH)
    log_debug(f"\n✅ 结果已保存至: {OUTPUT_JSON_PATH} 和 {OUTPUT_CSV_PATH}")

    log_debug("\n" + "="*50)
    log_debug("📌 示例结果（最新3个）:")
    log_debug("="*50)
    for row in csv_rows[-3:]:
        log_debug(str(row))
    log_debug("="*50)
    
    # 最终显存使用
    if torch.cuda.is_available():
        log_debug(f"💾 显存使用 (最终): {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        log_debug(f"💾 显存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    log_debug("🏁 推理任务完成!")


if __name__ == "__main__":
    main()