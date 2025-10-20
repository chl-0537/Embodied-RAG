import os
import re
import json
from openai import OpenAI
import time

# Azure OpenAI 配置
key = "bc709d6234e04a80ab2d744eb2434086"
os.environ["AZURE_OPENAI_API_KEY"] = key
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://lechuang.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"

# DeepSeek 配置
base_url = "https://api.deepseek.com"
base_model = "deepseek-chat"
api_key = "sk-b1882ddd341f4871bbcf093efe68ddf5"  # 替换为您的API密钥
client = OpenAI(api_key=api_key, base_url=base_url)


def split_text_into_paragraphs(text):
    """
    将文本按空行分割为段落
    处理多种换行情况，包括\n\n、\r\n\r\n等
    """
    # 使用正则表达式按一个或多个空行分割段落
    paragraphs = re.split(r'\n\s*\n', text.strip())
    # 过滤掉空段落
    return [p.strip() for p in paragraphs if p.strip()]


def split_long_paragraph(paragraph, max_length=1000, overlap=100):
    """
    将过长的段落分割成较小的块，保留重叠部分
    仅在段落长度超过max_length时才分割
    """
    if len(paragraph) <= max_length:
        return [paragraph]
    
    chunks = []
    start = 0
    para_length = len(paragraph)
    
    while start < para_length:
        end = min(start + max_length, para_length)
        chunk = paragraph[start:end]
        chunks.append(chunk)
        if end == para_length:
            break
        start = end - overlap
    
    return chunks


def _extract_json_from_text(text: str):
    """
    尝试从模型返回文本中提取 JSON（数组优先）
    """
    if not text:
        return []

    s = text.strip()

    # 去除```json ... ```代码块
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.DOTALL).strip()

    # 1) 直接解析
    try:
        parsed = json.loads(s)
        # 允许返回单对象：转为数组
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # 2) 尝试抓取第一个 JSON 数组
    m = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # 3) 退一步：抓取多个对象拼数组
    objs = re.findall(r"\{.*?\}", s, flags=re.DOTALL)
    if objs:
        items = []
        for o in objs:
            try:
                items.append(json.loads(o))
            except Exception:
                continue
        if items:
            return items

    return []


def _rule_based_fallback(paragraphs):
    """
    当 LLM 解析失败时的本地回退：用正则尽力抽取
    """
    results = []
    # 人物（可选）+（正在/在/正等修饰）+(到处/四处)* + 找/寻找/搜寻/探寻 + 目标
    pattern = re.compile(
        r'([\u4e00-\u9fa5]{1,6})?(?:正在|在|正)?(?:到处|四处)?(?:苦苦|拼命|继续)?'
        r'(寻找|找|搜寻|探寻)'
        r'([\u4e00-\u9fa5A-Za-z0-9《》“”"\']{1,20})'
    )
    for para in paragraphs:
        m = pattern.search(para)
        if m:
            character = m.group(1) or "未知人物"
            target = m.group(3) or "未知"
            results.append({"character": character, "target": target, "paragraph": para})
    return results


def call_llm_extract(paragraphs, retries=1, sleep_between=0.8):
    """
    调用大模型，从段落中抽取“人物-目标-段落”的结构化信息（健壮版）
    """
    # 尽量简洁的 system + user，减少 token
    system_msg = (
        "你是信息抽取助手。只输出严格的 JSON 数组，不要解释、不要额外文字、不要代码块。"
        "若没有可抽取项，返回空数组 []。数组元素的字段固定为："
        "character（人物名称，缺省填“未知人物”）、target（寻找对象，缺省填“未知”）、paragraph（完整段落）。"
    )
    user_msg = (
        "从以下段落中抽取“人物正在寻找某个东西”的场景。"
        "一段落若能抽取多个对象请拆成多条。"
        f"\n段落列表：\n{json.dumps(paragraphs, ensure_ascii=False)}"
    )

    last_content = ""
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model=base_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg if attempt == 0
                 else "仅返回 JSON 数组（不要任何解释/代码块）。上次输出无法解析，请严格遵守格式。\n" + user_msg}
            ],
            temperature=0,
            timeout=60
        )
        content = getattr(resp.choices[0].message, "content", "") or ""
        last_content = content

        data = _extract_json_from_text(content)
        if isinstance(data, list):
            # 轻量清洗 + 架构兜底
            cleaned = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                char = str(item.get("character") or "未知人物")
                target = str(item.get("target") or "未知")
                para = str(item.get("paragraph") or "").strip()
                if not para:
                    continue
                cleaned.append({"character": char, "target": target, "paragraph": para})
            print("cleaned", cleaned)
            return cleaned

        time.sleep(sleep_between)

    # 两次都解析失败，做回退
    print("LLM 返回解析失败，使用本地正则回退。原始返回：", last_content[:200])
    return _rule_based_fallback(paragraphs)


def _batch_paragraphs(paragraphs, max_items=5, max_chars=3000):
    """
    按段落数量和总字符数分批，避免单次 prompt 过长
    """
    batch, total = [], 0
    for para in paragraphs:
        l = len(para)
        # 如果加入当前段落会超过限制，则先输出当前批次
        if batch and (len(batch) >= max_items or total + l > max_chars):
            yield batch
            batch, total = [], 0
        batch.append(para)
        total += l
    if batch:
        yield batch


def extract_search_scenes(text, max_items_per_batch=5, max_chars_per_batch=3000, max_para_length=1000, para_overlap=100):
    """
    按段落处理文本，提取寻找场景
    """
    scenes = []
    
    # 1. 将文本分割为段落
    paragraphs = split_text_into_paragraphs(text)
    print(f"共分割出 {len(paragraphs)} 个段落")
    
    # 2. 初筛包含寻找关键词的段落
    kw = re.compile(r'(寻找|找|搜寻|探寻|找不到|找回|寻找着)')
    candidates = [p for p in paragraphs if kw.search(p)]
    
    # 3. 去重
    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]
    print(f"筛选出 {len(candidates)} 个可能包含寻找场景的段落")
    
    # 4. 处理过长段落
    processed_paragraphs = []
    for para in candidates:
        # 对过长段落进行分割
        chunks = split_long_paragraph(para, max_length=max_para_length, overlap=para_overlap)
        processed_paragraphs.extend(chunks)
    
    # 5. 分批处理段落
    for batch in _batch_paragraphs(processed_paragraphs, max_items=max_items_per_batch, max_chars=max_chars_per_batch):
        extracted = call_llm_extract(batch, retries=1)
        scenes.extend(extracted)
    
    print(f"共提取到 {len(scenes)} 个寻找场景")
    return scenes


def process_novels(novel_dir, output_path="search_scenes_1.json"):
    all_scenes = []

    for filename in os.listdir(novel_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(novel_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                print(f"开始处理 {filename}")
                text = f.read()
                scenes = extract_search_scenes(text)
                if scenes:
                    all_scenes.append({
                        "novel": filename,
                        "scenes": scenes
                    })
                print(f"{filename} 处理完成\n")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_scenes, f, ensure_ascii=False, indent=2)
    print(f"所有文件处理完成，结果已保存到 {output_path}")


if __name__ == "__main__":
    process_novels("./dataset/detect")
