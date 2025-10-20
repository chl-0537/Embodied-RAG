# rag_chatglm_final_api_embedding_azure.py
import os
import json
import time
import uuid
import math
import random
import requests
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# ----------------- ChatGLM API Config -----------------
CHATGLM_TOKEN = "fb4968f5f4cc415dbadc110ca418675a.AlJqhOJYMkIi2X0Q"
CHATGLM_LLM_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
CHATGLM_VISION_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

LLM_MODEL = "glm-4.5-flash"
VISION_MODEL = "glm-4v-flash"

HEADERS = {
    "Authorization": f"Bearer {CHATGLM_TOKEN}",
    "Content-Type": "application/json"
}

# ----------------- Memory storage -----------------
MEMORY_VECS: List[np.ndarray] = []
ID_MAP: List[str] = []
METADATA: Dict[str, Dict[str, Any]] = {}

# Azure Embedding 配置
EMB_DIM = 1536  # text-embedding-ada-002 默认维度
AZURE_API_KEY = "bc709d6234e04a80ab2d744eb2434086"
AZURE_API_TYPE = "azure"
AZURE_API_VERSION = "2023-05-15"
AZURE_ENDPOINT = "https://lechuang.openai.azure.com/"
AZURE_DEPLOYMENT = "lechuang-embedding"

os.environ["AZURE_OPENAI_API_KEY"] = AZURE_API_KEY
os.environ["OPENAI_API_TYPE"] = AZURE_API_TYPE
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_ENDPOINT
os.environ["AZURE_OPENAI_API_VERSION"] = AZURE_API_VERSION

from langchain_openai import AzureOpenAIEmbeddings

def get_azure_embedding_instance():
    """获取 Azure Embedding 对象"""
    return AzureOpenAIEmbeddings(
        deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        model="text-embedding-ada-002"
    )

azure_embeddings = get_azure_embedding_instance()

# ----------------- Embedding via Azure -----------------
def get_embedding_from_api(text: str, dimensions: int = EMB_DIM) -> np.ndarray:
    """
    使用 Azure OpenAI Embedding 生成向量。
    返回 np.ndarray 类型，自动归一化。
    """
    try:
        emb = azure_embeddings.embed_query(text)
        emb = np.array(emb, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        return emb
    except Exception as e:
        print(f"[Error] Azure embedding 生成失败: {e}")
        return np.zeros(dimensions, dtype=np.float32)

# ----------------- Memory and retrieval -----------------
def add_memory_entry_simple(text_desc: str, meta: Dict[str, Any]):
    emb = get_embedding_from_api(text_desc)
    new_id = meta.get("id") or f"item_{meta.get('item_label','unk')}_{datetime.now(timezone.utc).isoformat()}_{uuid.uuid4().hex[:6]}"
    MEMORY_VECS.append(emb)
    ID_MAP.append(new_id)
    METADATA[new_id] = {**meta, "text_desc": text_desc, "id": new_id}
    return new_id

def retrieve_candidates_by_text(query_text: str, topk: int = 20) -> List[Dict[str, Any]]:
    if not MEMORY_VECS:
        return []
    q_emb = get_embedding_from_api(query_text)
    all_embs = np.stack(MEMORY_VECS)
    sims = np.dot(all_embs, q_emb)
    idxs = np.argsort(-sims)[:topk]
    results = [{"id": ID_MAP[i], "score": float(sims[i]), "meta": METADATA[ID_MAP[i]]} for i in idxs]
    return results

def rerank_with_time_space(cands: List[Dict], current_pose: Optional[Dict] = None, now_ts: Optional[float] = None) -> List[Dict]:
    if now_ts is None:
        now_ts = time.time()
    scored = []
    for c in cands:
        m = c["meta"]
        sim = c.get("score", 0.0)
        conf = m.get("confidence", 0.5)
        tstamp = m.get("temporal", {}).get("timestamp_utc")
        dt = 1e6
        if tstamp:
            try:
                dt = max(0.0, now_ts - datetime.fromisoformat(tstamp).timestamp())
            except Exception:
                dt = 1e6
        w_time = math.exp(-dt / 3600.0)
        w_space = 1.0
        if current_pose and m.get("spatial", {}).get("pose"):
            p = m["spatial"]["pose"]
            d = math.sqrt((p["x"] - current_pose.get("x", 0))**2 + (p["y"] - current_pose.get("y", 0))**2)
            w_space = math.exp(-d / 3.0)
        final = sim * 0.6 + conf * 0.25 + w_time * 0.1 + w_space * 0.05
        scored.append({**c, "final_score": final})
    scored.sort(key=lambda e: e["final_score"], reverse=True)
    return scored

# ----------------- JSON schema helper -----------------
def is_valid_json_structure(obj: Any, required_schema: Dict[str, type]) -> bool:
    if not isinstance(obj, dict):
        return False
    for k, typ in required_schema.items():
        if k not in obj:
            return False
        if typ is None:
            continue
        if typ == float or typ == int:
            if not isinstance(obj[k], (int, float)):
                return False
        elif not isinstance(obj[k], typ) and not (isinstance(typ, tuple) and isinstance(obj[k], typ)):
            return False
    return True

# ----------------- ChatGLM LLM call -----------------
def call_chatglm_llm_json(system_msg: str, user_msg: str, schema: Dict[str, type], retries: int = 3) -> Dict:
    for attempt in range(retries):
        try:
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.2,
                "max_tokens": 8192,
                "stream": False
            }
            resp = requests.post(CHATGLM_LLM_URL, json=payload, headers=HEADERS, timeout=60)
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            import re
            m = re.search(r"\{.*\}", raw_text, re.S)
            candidate = m.group(0) if m else raw_text
            parsed = json.loads(candidate)
            if is_valid_json_structure(parsed, schema):
                return parsed
            else:
                print(f"[Schema mismatch attempt {attempt+1}] {parsed}")
                last_raw = raw_text
        except Exception as e:
            print(f"[ChatGLM call error {attempt+1}]: {e}")
            last_raw = str(e)
        time.sleep(0.5)
    return {"raw_output": last_raw}

# ----------------- ChatGLM Vision call -----------------
def call_chatglm_vision_json(image_urls: List[str], question: str) -> Dict:
    content_list = [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
    content_list.append({"type": "text", "text": question})
    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": content_list}],
        "temperature": 0.3,
        "max_tokens": 8192,
        "stream": False
    }
    resp = requests.post(CHATGLM_VISION_URL, json=payload, headers=HEADERS, timeout=60)
    data = resp.json()
    raw_text = data["choices"][0]["message"]["content"]
    try:
        import re
        m = re.search(r"\{.*\}", raw_text, re.S)
        return json.loads(m.group(0)) if m else {"text": raw_text}
    except:
        return {"text": raw_text}

# ----------------- High-level LLM tasks -----------------
def parse_user_intent_llm(user_command: str) -> Dict:
    system_inst = (
        "你是机器人语义解析器。请严格根据用户自然语言命令返回 JSON。"
        "字段：object (string|null), action (string|null), location (string|null), time (string|null), constraints (list)。"
        "未提及的填 null 或空列表。"
    )
    user_prompt = f"命令: {user_command}\n请仅输出 JSON 对象。"
    schema = {"object": str, "action": str, "location": (str, type(None)), "time": (str, type(None)), "constraints": list}
    parsed = call_chatglm_llm_json(system_inst, user_prompt, schema)
    for k in ["object", "action", "location", "time", "constraints"]:
        if k not in parsed:
            parsed[k] = None if k != "constraints" else []
    return parsed

def generate_action_plan_llm(intent: Dict, evidences: List[Dict], current_status: Dict) -> Dict:
    system_inst = (
        "你是机器人高级规划器。输入：intent, evidences, current_status。输出 JSON:\n"
        "{ next_action: string, reasoning: string, expected_observation: string, fallback_plan: list, confidence: float }\n"
        "next_action 必须是可执行命令文本。"
    )
    user_prompt = f"intent={json.dumps(intent, ensure_ascii=False)}\nevi={json.dumps(evidences, ensure_ascii=False)}\nstatus={json.dumps(current_status, ensure_ascii=False)}"
    schema = {"next_action": str, "reasoning": str, "expected_observation": str, "fallback_plan": list, "confidence": float}
    return call_chatglm_llm_json(system_inst, user_prompt, schema)

# ----------------- Main flow -----------------
if __name__ == "__main__":
    random.seed(1)
    demo_texts = [
        {"item_label": "watercup", "text_desc": "blue watercup on study_desk left of laptop", "spatial": {"location_semantic": "study_desk"}, "temporal": {"timestamp_utc": "2024-10-01T16:30:00+00:00"}, "confidence": 0.92},
        {"item_label": "phone", "text_desc": "black smartphone on couch armrest", "spatial": {"location_semantic": "sofa_armrest"}, "temporal": {"timestamp_utc": "2024-09-30T10:00:00+00:00"}, "confidence": 0.9}
    ]
    for e in demo_texts:
        add_memory_entry_simple(e["text_desc"], e)

    user_cmd = "帮我找到蓝色水杯，应该在书房的书桌上。"
    robot_pose = {"x": 0.0, "y": 0.0, "semantic": "living_room_entrance", "battery": 0.95}

    intent = parse_user_intent_llm(user_cmd)
    print("Intent:", intent)

    cands = retrieve_candidates_by_text(intent.get("object") or "object")
    ranked = rerank_with_time_space(cands, current_pose=robot_pose)
    top_evidences = ranked[:5]

    evid_slim = [{"id": e["id"], "loc": e["meta"].get("spatial",{}).get("location_semantic"), "time": e["meta"].get("temporal",{}).get("timestamp_utc"), "conf": e["meta"].get("confidence")} for e in top_evidences]
    plan = generate_action_plan_llm(intent, evid_slim, {"robot_location": robot_pose.get("semantic","unknown")})
    print("Action plan:", plan)

    image_urls = ["https://known-black-9eebdukx4b.edgeone.app/%E7%BB%98%E5%88%B6%E4%B9%A6%E6%88%BF%203D%20%E5%9B%BE%E5%83%8F.png"]
    vision_res = call_chatglm_vision_json(image_urls, f"请检查图片中是否有{intent.get('object')}，并输出 JSON")
    print("Vision result:", vision_res)
