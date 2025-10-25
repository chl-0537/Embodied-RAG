# rag_chatglm_final_api_embedding_azure.py
import os
import json
import time
import uuid
import math
import random
import requests
import numpy as np
import base64
import logging
import glob
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
from enum import Enum

# 尝试导入Chroma，如果失败则提供安装提示
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("警告: ChromaDB未安装。请运行: pip install chromadb")

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

# ----------------- Vector Database Configuration -----------------
VECTOR_DB_DIR = "vector_db"
COLLECTION_NAME = "robot_memory"

# 初始化Chroma数据库
def init_chroma_db():
    """初始化Chroma向量数据库"""
    if not CHROMA_AVAILABLE:
        log_error("ChromaDB不可用，请先安装: pip install chromadb")
        return None
    
    try:
        # 创建向量数据库目录
        if not os.path.exists(VECTOR_DB_DIR):
            os.makedirs(VECTOR_DB_DIR)
        
        # 初始化Chroma客户端
        client = chromadb.PersistentClient(
            path=VECTOR_DB_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            log_info(f"加载现有集合: {COLLECTION_NAME}")
        except Exception:
            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "机器人记忆存储"}
            )
            log_info(f"创建新集合: {COLLECTION_NAME}")
        
        return client, collection
    except Exception as e:
        log_error("初始化Chroma数据库失败", e)
        return None

# 全局数据库对象
chroma_client = None
chroma_collection = None

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

# ----------------- Logging Configuration -----------------
def setup_logging():
    """设置日志配置"""
    # 创建logs目录
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成基于时间的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"rag_robot_{timestamp}.log")
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    # 获取logger
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，日志文件: {log_filename}")
    
    return logger, log_filename

def cleanup_old_logs(log_dir="logs", max_files=10):
    """清理旧的日志文件，保留最新的max_files个"""
    try:
        log_files = glob.glob(os.path.join(log_dir, "rag_robot_*.log"))
        if len(log_files) > max_files:
            # 按修改时间排序
            log_files.sort(key=os.path.getmtime)
            # 删除最旧的文件
            for old_file in log_files[:-max_files]:
                os.remove(old_file)
                logging.info(f"删除旧日志文件: {old_file}")
    except Exception as e:
        logging.warning(f"清理旧日志文件失败: {e}")

# 初始化日志系统
logger, current_log_file = setup_logging()
cleanup_old_logs()

# ----------------- Status Code Enum -----------------
class StatusCode(Enum):
    OK = "ok"
    VISION_ERROR = "vision_error"
    LLM_SCHEMA_ERROR = "llm_schema_error"
    LLM_API_ERROR = "llm_api_error"
    EMBEDDING_ERROR = "embedding_error"
    MEMORY_ERROR = "memory_error"
    UNKNOWN_ERROR = "unknown_error"

class Result:
    """统一的结果返回类"""
    def __init__(self, status: StatusCode, data: Any = None, error_msg: str = "", metadata: Dict = None):
        self.status = status
        self.data = data
        self.error_msg = error_msg
        self.metadata = metadata or {}
    
    def is_success(self) -> bool:
        return self.status == StatusCode.OK
    
    def to_dict(self) -> Dict:
        return {
            "status": self.status.value,
            "data": self.data,
            "error_msg": self.error_msg,
            "metadata": self.metadata
        }

# ----------------- Logging Helper Functions -----------------
def log_info(message: str, data: Any = None):
    """记录信息日志"""
    if data is not None:
        logger.info(f"{message} | Data: {json.dumps(data, ensure_ascii=False, default=str)}")
    else:
        logger.info(message)

def log_error(message: str, error: Exception = None, data: Any = None):
    """记录错误日志"""
    if error:
        logger.error(f"{message} | Error: {str(error)}")
    else:
        logger.error(message)
    
    if data is not None:
        logger.error(f"Error Data: {json.dumps(data, ensure_ascii=False, default=str)}")

def log_warning(message: str, data: Any = None):
    """记录警告日志"""
    if data is not None:
        logger.warning(f"{message} | Data: {json.dumps(data, ensure_ascii=False, default=str)}")
    else:
        logger.warning(message)

def log_debug(message: str, data: Any = None):
    """记录调试日志"""
    if data is not None:
        logger.debug(f"{message} | Data: {json.dumps(data, ensure_ascii=False, default=str)}")
    else:
        logger.debug(message)

def log_result(result: Result, operation: str):
    """记录Result对象的日志"""
    if result.is_success():
        log_info(f"{operation} 成功", result.data)
    else:
        log_error(f"{operation} 失败: {result.error_msg}", data=result.metadata)

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
        log_error("Azure embedding 生成失败", e)
        return np.zeros(dimensions, dtype=np.float32)

# ----------------- Memory and retrieval -----------------
def add_memory_entry_simple(text_desc: str, meta: Dict[str, Any]):
    """添加记忆条目到Chroma数据库"""
    global chroma_collection
    
    if chroma_collection is None:
        log_error("Chroma数据库未初始化")
        return None
    
    try:
        # 生成唯一ID
        new_id = meta.get("id") or f"item_{meta.get('item_label','unk')}_{datetime.now(timezone.utc).isoformat()}_{uuid.uuid4().hex[:6]}"
        
        # 准备元数据
        chroma_metadata = {
            "item_label": meta.get("item_label", "unknown"),
            "location_semantic": meta.get("spatial", {}).get("location_semantic", "unknown"),
            "timestamp_utc": meta.get("temporal", {}).get("timestamp_utc", datetime.now(timezone.utc).isoformat()),
            "confidence": meta.get("confidence", 0.5),
            "text_desc": text_desc,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # 添加条目到Chroma
        chroma_collection.add(
            documents=[text_desc],
            embeddings=[get_embedding_from_api(text_desc).tolist()],
            metadatas=[chroma_metadata],
            ids=[new_id]
        )
        
        log_info(f"Added memory entry to Chroma: {new_id}", chroma_metadata)
        return new_id
        
    except Exception as e:
        log_error("添加记忆条目失败", e, {"text_desc": text_desc, "meta": meta})
        return None

def retrieve_candidates_by_text(query_text: str, topk: int = 20, location_filter: str = None) -> List[Dict[str, Any]]:
    """从Chroma数据库检索候选条目"""
    global chroma_collection
    
    if chroma_collection is None:
        log_error("Chroma数据库未初始化")
        return []
    
    try:
        # 构建查询条件
        where_clause = None
        if location_filter:
            where_clause = {"location_semantic": location_filter}
        
        # 执行查询
        results = chroma_collection.query(
            query_texts=[query_text],
            n_results=topk,
            where=where_clause
        )
        
        # 转换结果格式
        candidates = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                candidate = {
                    "id": doc_id,
                    "score": results['distances'][0][i] if results['distances'] else 0.0,
                    "meta": {
                        "text_desc": results['documents'][0][i] if results['documents'] else "",
                        **(results['metadatas'][0][i] if results['metadatas'] else {})
                    }
                }
                candidates.append(candidate)
        
        log_info(f"Retrieved {len(candidates)} candidates from Chroma", {
            "query": query_text,
            "location_filter": location_filter,
            "topk": topk
        })
        
        return candidates
        
    except Exception as e:
        log_error("检索候选条目失败", e, {"query_text": query_text, "topk": topk})
        return []

def retrieve_candidates_by_intent(intent: Dict[str, Any], topk: int = 20) -> List[Dict[str, Any]]:
    """基于意图进行智能检索"""
    # 构建增强的查询文本
    object_name = intent.get("object", "")
    location = intent.get("location", "")
    
    if object_name and location:
        # 组合对象和位置信息
        query_text = f"{object_name} in {location}"
        location_filter = location
    elif object_name:
        # 仅使用对象信息
        query_text = object_name
        location_filter = None
    else:
        # 使用通用查询
        query_text = "object"
        location_filter = None
    
    log_info("Enhanced query construction", {
        "original_object": object_name,
        "original_location": location,
        "enhanced_query": query_text,
        "location_filter": location_filter
    })
    
    return retrieve_candidates_by_text(query_text, topk, location_filter)

def get_memory_stats():
    """获取记忆数据库统计信息"""
    global chroma_collection
    
    if chroma_collection is None:
        return {"error": "Database not initialized"}
    
    try:
        count = chroma_collection.count()
        return {
            "total_memories": count,
            "collection_name": COLLECTION_NAME,
            "database_path": VECTOR_DB_DIR
        }
    except Exception as e:
        log_error("获取记忆统计信息失败", e)
        return {"error": str(e)}

def search_by_location(location: str, topk: int = 10):
    """按位置搜索记忆"""
    global chroma_collection
    
    if chroma_collection is None:
        log_error("Chroma数据库未初始化")
        return []
    
    try:
        results = chroma_collection.query(
            query_texts=["objects in location"],
            n_results=topk,
            where={"location_semantic": location}
        )
        
        candidates = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                candidate = {
                    "id": doc_id,
                    "score": results['distances'][0][i] if results['distances'] else 0.0,
                    "meta": {
                        "text_desc": results['documents'][0][i] if results['documents'] else "",
                        **(results['metadatas'][0][i] if results['metadatas'] else {})
                    }
                }
                candidates.append(candidate)
        
        log_info(f"Found {len(candidates)} objects in location: {location}")
        return candidates
        
    except Exception as e:
        log_error("按位置搜索失败", e, {"location": location})
        return []

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
def call_chatglm_llm_json(system_msg: str, user_msg: str, schema: Dict[str, type], retries: int = 3) -> Result:
    last_raw = ""
    last_error = ""
    
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
            
            try:
                parsed = json.loads(candidate)
                if is_valid_json_structure(parsed, schema):
                            return Result(StatusCode.OK, parsed, metadata={"attempt": attempt + 1, "raw_text": raw_text})
                else:
                            log_warning(f"Schema validation failed on attempt {attempt+1}", parsed)
                            last_raw = raw_text
                            last_error = f"Schema validation failed: {parsed}"
                            
                            # 如果是最后一次尝试，使用format修正prompt
                            if attempt == retries - 1:
                                return call_chatglm_llm_with_format_correction(system_msg, user_msg, schema, raw_text)
                        
            except json.JSONDecodeError as je:
                log_warning(f"JSON parse error on attempt {attempt+1}", str(je))
                last_raw = raw_text
                last_error = f"JSON parse error: {je}"
                
                # 如果是最后一次尝试，使用format修正prompt
                if attempt == retries - 1:
                    return call_chatglm_llm_with_format_correction(system_msg, user_msg, schema, raw_text)
                    
        except requests.RequestException as re:
            log_error(f"API request error on attempt {attempt+1}", re)
            last_error = f"API request error: {re}"
        except Exception as e:
            log_error(f"ChatGLM call error on attempt {attempt+1}", e)
            last_error = f"Unexpected error: {e}"
        
        time.sleep(0.5)
    
    return Result(StatusCode.LLM_SCHEMA_ERROR, {"raw_output": last_raw}, last_error, 
                  {"attempts": retries, "last_raw": last_raw})

def call_chatglm_llm_with_format_correction(system_msg: str, user_msg: str, schema: Dict[str, type], failed_output: str) -> Result:
    """当schema校验失败时，使用format修正prompt重试"""
    try:
        format_correction_prompt = (
            f"之前的输出格式不正确：\n{failed_output}\n\n"
            f"请严格按照以下JSON schema重新输出：\n{json.dumps(schema, indent=2)}\n\n"
            f"确保输出是有效的JSON格式，包含所有必需字段。"
        )
        
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": failed_output},
                {"role": "user", "content": format_correction_prompt}
            ],
            "temperature": 0.1,  # 降低温度以获得更稳定的格式
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
            return Result(StatusCode.OK, parsed, metadata={"format_corrected": True, "raw_text": raw_text})
        else:
            return Result(StatusCode.LLM_SCHEMA_ERROR, {"raw_output": raw_text}, 
                         "Format correction failed", {"format_corrected": True})
            
    except Exception as e:
        return Result(StatusCode.LLM_API_ERROR, {"raw_output": failed_output}, 
                     f"Format correction error: {e}", {"format_corrected": False})

# ----------------- Enhanced Vision Detection -----------------
def call_chatglm_multi_object_detection(image_url: str, target_objects: List[str] = None) -> Result:
    """多对象检测和边界框验证"""
    try:
        # 构建多对象检测prompt
        if target_objects:
            target_str = ", ".join(target_objects)
            detection_prompt = f"""
请仔细分析图片中的所有对象，特别关注以下目标对象：{target_str}

请输出JSON格式，包含字段：
{{
  "objects": [
    {{
      "label": "对象名称",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "description": "详细描述"
    }}
  ],
  "scene_description": "整体场景描述",
  "detection_summary": "检测总结"
}}

要求：
1. bbox格式：[左上角x, 左上角y, 右下角x, 右下角y]
2. confidence范围：0.0-1.0
3. 只输出JSON，不要其他文字
4. 检测所有可见对象，不限于目标对象
"""
        else:
            detection_prompt = """
请仔细分析图片中的所有对象。

请输出JSON格式，包含字段：
{
  "objects": [
    {
      "label": "对象名称",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "description": "详细描述"
    }
  ],
  "scene_description": "整体场景描述",
  "detection_summary": "检测总结"
}

要求：
1. bbox格式：[左上角x, 左上角y, 右下角x, 右下角y]
2. confidence范围：0.0-1.0
3. 只输出JSON，不要其他文字
4. 检测所有可见对象
"""

        content_list = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": detection_prompt}
        ]
        
        payload = {
            "model": VISION_MODEL,
            "messages": [{"role": "user", "content": content_list}],
            "temperature": 0.1  # 降低温度以获得更稳定的JSON输出
        }
        
        resp = requests.post(CHATGLM_VISION_URL, json=payload, headers=HEADERS, timeout=60)
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        
        # 解析JSON结果
        try:
            import re
            m = re.search(r"\{.*\}", raw_text, re.S)
            if m:
                parsed_result = json.loads(m.group(0))
                
                # 验证JSON结构
                if validate_detection_result(parsed_result):
                    log_info("Multi-object detection successful", {
                        "objects_count": len(parsed_result.get("objects", [])),
                        "target_objects": target_objects
                    })
                    return Result(StatusCode.OK, parsed_result, metadata={
                        "raw_text": raw_text,
                        "detection_type": "multi_object"
                    })
                else:
                    log_warning("Invalid detection result structure", parsed_result)
                    return Result(StatusCode.VISION_ERROR, None, "Invalid detection result structure")
            else:
                log_warning("No JSON found in vision response", {"raw_text": raw_text})
                return Result(StatusCode.VISION_ERROR, None, "No JSON found in response")
                
        except json.JSONDecodeError as e:
            log_error("JSON parsing failed in vision detection", e, {"raw_text": raw_text})
            return Result(StatusCode.VISION_ERROR, None, f"JSON parsing failed: {e}")
            
    except requests.RequestException as e:
        log_error("Vision API request failed", e)
        return Result(StatusCode.VISION_ERROR, None, f"Vision API request failed: {e}")
    except Exception as e:
        log_error("Vision processing error", e)
        return Result(StatusCode.VISION_ERROR, None, f"Vision processing error: {e}")

def validate_detection_result(result: Dict) -> bool:
    """验证检测结果的结构"""
    try:
        if not isinstance(result, dict):
            return False
        
        if "objects" not in result:
            return False
        
        objects = result["objects"]
        if not isinstance(objects, list):
            return False
        
        for obj in objects:
            if not isinstance(obj, dict):
                return False
            
            required_fields = ["label", "bbox", "confidence"]
            for field in required_fields:
                if field not in obj:
                    return False
            
            # 验证bbox格式
            bbox = obj["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                return False
            
            # 验证confidence范围
            confidence = obj["confidence"]
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                return False
        
        return True
    except Exception:
        return False

def add_vision_memory(detection_result: Dict, image_path: str = None) -> List[str]:
    """将视觉检测结果写回记忆区，实现跨模态RAG"""
    global chroma_collection
    
    if chroma_collection is None:
        log_error("Chroma数据库未初始化，无法添加视觉记忆")
        return []
    
    added_memory_ids = []
    
    try:
        objects = detection_result.get("objects", [])
        scene_description = detection_result.get("scene_description", "")
        
        # 为每个检测到的对象添加记忆
        for obj in objects:
            label = obj.get("label", "unknown")
            confidence = obj.get("confidence", 0.0)
            description = obj.get("description", "")
            bbox = obj.get("bbox", [])
            
            # 构建视觉记忆的描述文本
            vision_desc = f"vision detected {label}"
            if description:
                vision_desc += f" - {description}"
            
            # 构建元数据
            vision_metadata = {
                "item_label": label,
                "source": "vision_detection",
                "detection_confidence": confidence,
                "bbox": bbox,
                "description": description,
                "spatial": {
                    "location_semantic": "current_scene",
                    "source": "visual_analysis"
                },
                "temporal": {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "observation_type": "real_time_vision"
                },
                "metadata": {
                    "image_path": image_path,
                    "detection_method": "chatglm_multi_object",
                    "scene_description": scene_description
                }
            }
            
            # 添加到Chroma数据库
            memory_id = add_memory_entry_simple(vision_desc, vision_metadata)
            if memory_id:
                added_memory_ids.append(memory_id)
                log_info(f"Added vision memory: {memory_id}", {
                    "label": label,
                    "confidence": confidence,
                    "description": description
                })
        
        # 添加整体场景记忆
        if scene_description:
            scene_desc = f"vision scene: {scene_description}"
            scene_metadata = {
                "item_label": "scene",
                "source": "vision_scene",
                "spatial": {
                    "location_semantic": "current_scene",
                    "source": "visual_analysis"
                },
                "temporal": {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "observation_type": "real_time_vision"
                },
                "metadata": {
                    "image_path": image_path,
                    "detection_method": "chatglm_scene_analysis",
                    "objects_count": len(objects)
                }
            }
            
            scene_memory_id = add_memory_entry_simple(scene_desc, scene_metadata)
            if scene_memory_id:
                added_memory_ids.append(scene_memory_id)
                log_info(f"Added scene memory: {scene_memory_id}")
        
        log_info(f"Vision memory integration completed", {
            "objects_added": len(objects),
            "total_memories_added": len(added_memory_ids),
            "memory_ids": added_memory_ids
        })
        
        return added_memory_ids
        
    except Exception as e:
        log_error("添加视觉记忆失败", e, {"detection_result": detection_result})
        return []

def cross_modal_rag_validation(intent: Dict, vision_result: Dict, memory_evidences: List[Dict]) -> Dict:
    """跨模态RAG验证：结合视觉检测和记忆检索进行一致性分析"""
    try:
        target_object = intent.get("object", "")
        target_location = intent.get("location", "")
        
        vision_objects = vision_result.get("objects", [])
        vision_scene = vision_result.get("scene_description", "")
        
        # 分析视觉检测结果
        vision_analysis = {
            "target_found_in_vision": False,
            "matching_objects": [],
            "confidence_scores": [],
            "scene_matches_location": False
        }
        
        # 检查目标对象是否在视觉检测中
        for obj in vision_objects:
            obj_label = obj.get("label", "").lower()
            obj_confidence = obj.get("confidence", 0.0)
            
            if target_object and target_object.lower() in obj_label:
                vision_analysis["target_found_in_vision"] = True
                vision_analysis["matching_objects"].append(obj)
                vision_analysis["confidence_scores"].append(obj_confidence)
        
        # 检查场景是否匹配位置
        if target_location and vision_scene:
            vision_analysis["scene_matches_location"] = target_location.lower() in vision_scene.lower()
        
        # 分析记忆证据
        memory_analysis = {
            "memory_matches": [],
            "location_consistency": 0.0,
            "confidence_consistency": 0.0
        }
        
        for evidence in memory_evidences:
            if evidence.get("type") == "memory":
                meta = evidence.get("meta", {})
                memory_label = meta.get("item_label", "").lower()
                memory_location = meta.get("spatial", {}).get("location_semantic", "").lower()
                memory_confidence = meta.get("confidence", 0.0)
                
                if target_object and target_object.lower() in memory_label:
                    memory_analysis["memory_matches"].append(evidence)
                    
                    # 计算位置一致性
                    if target_location and target_location.lower() in memory_location:
                        memory_analysis["location_consistency"] += 1.0
                    
                    # 计算置信度一致性
                    memory_analysis["confidence_consistency"] += memory_confidence
        
        # 计算总体一致性分数
        consistency_score = 0.0
        consistency_factors = []
        
        # 视觉-记忆一致性
        if vision_analysis["target_found_in_vision"] and memory_analysis["memory_matches"]:
            consistency_score += 0.4
            consistency_factors.append("vision_memory_match")
        
        # 位置一致性
        if vision_analysis["scene_matches_location"] and memory_analysis["location_consistency"] > 0:
            consistency_score += 0.3
            consistency_factors.append("location_consistency")
        
        # 置信度一致性
        if vision_analysis["confidence_scores"] and memory_analysis["confidence_consistency"] > 0:
            avg_vision_conf = sum(vision_analysis["confidence_scores"]) / len(vision_analysis["confidence_scores"])
            avg_memory_conf = memory_analysis["confidence_consistency"] / max(len(memory_analysis["memory_matches"]), 1)
            conf_diff = abs(avg_vision_conf - avg_memory_conf)
            if conf_diff < 0.3:  # 置信度差异小于0.3
                consistency_score += 0.3
                consistency_factors.append("confidence_consistency")
        
        cross_modal_result = {
            "consistency_score": consistency_score,
            "consistency_factors": consistency_factors,
            "vision_analysis": vision_analysis,
            "memory_analysis": memory_analysis,
            "recommendation": get_cross_modal_recommendation(consistency_score, vision_analysis, memory_analysis)
        }
        
        log_info("Cross-modal RAG validation completed", cross_modal_result)
        return cross_modal_result
        
    except Exception as e:
        log_error("跨模态RAG验证失败", e)
        return {"consistency_score": 0.0, "error": str(e)}

def get_cross_modal_recommendation(consistency_score: float, vision_analysis: Dict, memory_analysis: Dict) -> str:
    """基于一致性分析生成推荐"""
    if consistency_score >= 0.8:
        return "high_confidence: 视觉和记忆高度一致，可以执行任务"
    elif consistency_score >= 0.5:
        return "medium_confidence: 视觉和记忆基本一致，建议进一步验证"
    elif vision_analysis["target_found_in_vision"] and not memory_analysis["memory_matches"]:
        return "vision_only: 仅在视觉中检测到目标，可能是新物体"
    elif not vision_analysis["target_found_in_vision"] and memory_analysis["memory_matches"]:
        return "memory_only: 仅在记忆中找到目标，可能需要重新搜索"
    else:
        return "low_confidence: 视觉和记忆不一致，需要人工干预"

# ----------------- ChatGLM Vision call -----------------
def call_chatglm_vision_json(image_url: str, question: str) -> Result:
    try:
        content_list = []
        content_list.append({"type": "image_url", "image_url": {"url": image_url}})
        content_list.append({"type": "text", "text": question})
        payload = {
            "model": VISION_MODEL,
            "messages": [{"role": "user", "content": content_list}]
        }
        resp = requests.post(CHATGLM_VISION_URL, json=payload, headers=HEADERS, timeout=60)
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
            
        try:
            import re
            m = re.search(r"\{.*\}", raw_text, re.S)
            parsed_result = json.loads(m.group(0)) if m else {"text": raw_text}
            return Result(StatusCode.OK, parsed_result, metadata={"raw_text": raw_text})
        except json.JSONDecodeError:
            return Result(StatusCode.OK, {"text": raw_text}, metadata={"raw_text": raw_text, "json_parse_failed": True})
    except requests.RequestException as e:
        return Result(StatusCode.VISION_ERROR, None, f"Vision API request failed: {e}")
    except Exception as e:
        return Result(StatusCode.VISION_ERROR, None, f"Vision processing error: {e}")

def call_chatglm_vision_with_fallback(image_url: str, question: str, memory_evidences: List[Dict]) -> Result:
    """视觉检测，失败时fallback到memory-based reasoning"""
    vision_result = call_chatglm_vision_json(image_url, question)
    
    if vision_result.is_success():
        return vision_result
    
    # Vision失败，fallback到memory-based reasoning
    log_warning(f"Vision detection failed: {vision_result.error_msg}")
    log_info("Falling back to memory-based reasoning")
    
    # 基于记忆进行推理
    memory_reasoning_prompt = (
        f"视觉检测失败，请基于以下历史记忆进行推理：\n"
        f"问题：{question}\n"
        f"相关记忆：{json.dumps(memory_evidences, ensure_ascii=False, indent=2)}\n\n"
        f"请分析这些记忆信息，推断目标对象的位置和状态，并输出JSON格式的结果。"
    )
    
    fallback_result = call_chatglm_llm_json(
        "你是基于记忆进行推理的机器人助手。当视觉检测不可用时，请基于历史记忆进行合理推断。",
        memory_reasoning_prompt,
        {"detected": bool, "confidence": float, "reasoning": str, "location": str, "source": str}
    )
    
    if fallback_result.is_success():
        fallback_data = fallback_result.data
        fallback_data["source"] = "memory_fallback"
        fallback_data["vision_failed"] = True
        return Result(StatusCode.OK, fallback_data, 
                     f"Vision failed, used memory fallback: {vision_result.error_msg}",
                     {"fallback_used": True, "original_vision_error": vision_result.error_msg})
    else:
        return Result(StatusCode.VISION_ERROR, None, 
                     f"Both vision and memory fallback failed. Vision: {vision_result.error_msg}, Fallback: {fallback_result.error_msg}")

# ----------------- Vision evidence fusion -----------------
def structure_vision_evidence(vision_result: Dict, target_object: str, image_path: str = None) -> Dict:
    """
    将视觉检测结果结构化为标准证据格式
    """
    # 提取视觉检测的关键信息
    vision_text = vision_result.get("text", "")
    
    # 尝试从JSON中提取结构化信息
    detected_objects = []
    confidence_score = 0.5
    location_info = "unknown"
    
    if isinstance(vision_result, dict):
        # 如果vision_result包含结构化信息
        if "detected_objects" in vision_result:
            detected_objects = vision_result["detected_objects"]
        if "confidence" in vision_result:
            confidence_score = vision_result["confidence"]
        if "location" in vision_result:
            location_info = vision_result["location"]
    
    # 基于文本内容进行简单分析
    target_found = target_object.lower() in vision_text.lower() if target_object else False
    if target_found:
        confidence_score = max(confidence_score, 0.7)  # 如果文本中提到目标，提高置信度
    
    # 构建标准化的证据格式
    evidence = {
        "id": f"vision_{datetime.now(timezone.utc).isoformat()}_{uuid.uuid4().hex[:6]}",
        "type": "vision_observation",
        "source": "visual_detection",
        "target_object": target_object,
        "detected": target_found,
        "confidence": confidence_score,
        "raw_text": vision_text,
        "spatial": {
            "location_semantic": location_info,
            "source": "visual_analysis"
        },
        "temporal": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "observation_type": "real_time"
        },
        "metadata": {
            "image_path": image_path,
            "detection_method": "chatglm_vision",
            "structured_data": vision_result
        }
    }
    
    return evidence

def fuse_vision_with_memory(evidences: List[Dict], vision_evidence: Dict) -> List[Dict]:
    """
    将视觉证据与记忆证据进行融合，添加一致性分析
    """
    # 将视觉证据添加到证据列表中
    fused_evidences = evidences.copy()
    fused_evidences.append(vision_evidence)
    
    # 为每个证据添加一致性分析
    target_object = vision_evidence.get("target_object", "")
    vision_detected = vision_evidence.get("detected", False)
    vision_confidence = vision_evidence.get("confidence", 0.5)
    
    for evidence in fused_evidences:
        if evidence.get("type") == "memory":
            # 分析记忆证据与视觉观察的一致性
            memory_object = evidence.get("meta", {}).get("item_label", "")
            memory_location = evidence.get("meta", {}).get("spatial", {}).get("location_semantic", "")
            
            # 计算一致性分数
            object_match = target_object.lower() in memory_object.lower() if target_object and memory_object else False
            consistency_score = 0.0
            
            if object_match and vision_detected:
                consistency_score = 0.8  # 对象匹配且视觉检测到
            elif object_match and not vision_detected:
                consistency_score = 0.3  # 对象匹配但视觉未检测到
            elif not object_match and vision_detected:
                consistency_score = 0.2  # 对象不匹配但视觉检测到其他
            else:
                consistency_score = 0.1  # 都不匹配
            
            # 添加一致性信息
            evidence["consistency"] = {
                "with_vision": consistency_score,
                "object_match": object_match,
                "vision_detected": vision_detected,
                "analysis": f"Memory object '{memory_object}' vs Vision target '{target_object}'"
            }
        elif evidence.get("type") == "vision_observation":
            # 为视觉证据添加与记忆的一致性分析
            memory_matches = [e for e in evidences if e.get("meta", {}).get("item_label", "").lower() == target_object.lower()]
            memory_consistency = len(memory_matches) > 0
            
            evidence["consistency"] = {
                "with_memory": memory_consistency,
                "memory_matches": len(memory_matches),
                "analysis": f"Vision detected '{target_object}': {vision_detected}, Memory matches: {len(memory_matches)}"
            }
    
    return fused_evidences

# ----------------- High-level LLM tasks -----------------
def parse_user_intent_llm(user_command: str) -> Result:
    system_inst = (
        "你是机器人语义解析器。请严格根据用户自然语言命令返回 JSON。"
        "字段：object (string|null), action (string|null), location (string|null), time (string|null), constraints (list)。"
        "未提及的填 null 或空列表。"
    )
    user_prompt = f"命令: {user_command}\n请仅输出 JSON 对象。"
    schema = {"object": str, "action": str, "location": (str, type(None)), "time": (str, type(None)), "constraints": list}
    
    result = call_chatglm_llm_json(system_inst, user_prompt, schema)
    if not result.is_success():
        return result
    
    parsed = result.data
    for k in ["object", "action", "location", "time", "constraints"]:
        if k not in parsed:
            parsed[k] = None if k != "constraints" else []
    
    return Result(StatusCode.OK, parsed, metadata=result.metadata)

def generate_action_plan_llm(intent: Dict, evidences: List[Dict], current_status: Dict) -> Result:
    system_inst = (
        "你是机器人高级规划器。输入：intent, evidences, current_status。输出 JSON:\n"
        "{ next_action: string, reasoning: string, expected_observation: string, fallback_plan: list, confidence: float, consistency_analysis: object, cross_modal_insights: object }\n"
        "next_action 必须是可执行命令文本。\n"
        "evidences 包含两种类型：\n"
        "1. memory: 来自历史记忆的语义检索结果\n"
        "2. vision_observation: 来自实时视觉检测的结果\n"
        "current_status 包含跨模态验证结果，请特别关注：\n"
        "- cross_modal_validation.consistency_score: 视觉与记忆的一致性分数\n"
        "- cross_modal_validation.recommendation: 系统推荐\n"
        "请基于跨模态验证结果制定更智能的行动计划。"
    )
    user_prompt = f"intent={json.dumps(intent, ensure_ascii=False)}\nevi={json.dumps(evidences, ensure_ascii=False)}\nstatus={json.dumps(current_status, ensure_ascii=False)}\n\n请基于跨模态验证结果分析视觉观察与记忆证据的一致性，并制定智能行动计划。"
    schema = {"next_action": str, "reasoning": str, "expected_observation": str, "fallback_plan": list, "confidence": float, "consistency_analysis": dict, "cross_modal_insights": dict}
    return call_chatglm_llm_json(system_inst, user_prompt, schema)

# ----------------- Robot Navigation System -----------------
class RobotNavigator:
    """机器人导航系统"""
    
    def __init__(self):
        self.current_location = "living_room"
        self.available_rooms = {
            "living_room": {
                "image_path": "./picture/living_room.png",
                "exits": ["kitchen", "bedroom", "study_room"],
                "description": "客厅，有三个入口分别通往厨房、卧室、书房"
            },
            "kitchen": {
                "image_path": "./picture/kitchen.png", 
                "exits": ["living_room"],
                "description": "厨房"
            },
            "bedroom": {
                "image_path": "./picture/bedroom.png",
                "exits": ["living_room"], 
                "description": "卧室"
            },
            "study_room": {
                "image_path": "./picture/study_room.png",
                "exits": ["living_room"],
                "description": "书房"
            }
        }
        self.navigation_history = []
    
    def get_current_image_path(self):
        """获取当前房间的图片路径"""
        return self.available_rooms[self.current_location]["image_path"]
    
    def get_available_exits(self):
        """获取当前房间的可用出口"""
        return self.available_rooms[self.current_location]["exits"]
    
    def navigate_to_room(self, target_room):
        """导航到指定房间"""
        if target_room in self.available_rooms[self.current_location]["exits"]:
            self.navigation_history.append(self.current_location)
            self.current_location = target_room
            log_info(f"Robot navigated from {self.navigation_history[-1]} to {self.current_location}")
            return True
        else:
            log_warning(f"Cannot navigate to {target_room} from {self.current_location}")
            return False
    
    def get_location_description(self):
        """获取当前位置描述"""
        return self.available_rooms[self.current_location]["description"]

def simulate_robot_navigation_task():
    """模拟机器人导航任务"""
    log_info("=" * 60)
    log_info("ROBOT NAVIGATION TASK SIMULATION")
    log_info("=" * 60)
    
    # 初始化向量数据库
    log_info("Initializing vector database")
    db_result = init_chroma_db()
    if db_result is None:
        log_error("Failed to initialize vector database, exiting")
        return
    
    global chroma_client, chroma_collection
    chroma_client, chroma_collection = db_result
    log_info("Vector database initialized successfully")
    
    # 初始化机器人导航器
    navigator = RobotNavigator()
    log_info(f"Robot starting at: {navigator.current_location}")
    
    # 添加记忆数据
    random.seed(1)
    log_info("Initializing memory data")
    demo_texts = [
        {"item_label": "watercup", "text_desc": "blue watercup on study_desk left of laptop", "spatial": {"location_semantic": "study_desk"}, "temporal": {"timestamp_utc": "2024-10-01T16:30:00+00:00"}, "confidence": 0.92},
        {"item_label": "phone", "text_desc": "black smartphone on couch armrest", "spatial": {"location_semantic": "sofa_armrest"}, "temporal": {"timestamp_utc": "2024-09-30T10:00:00+00:00"}, "confidence": 0.9},
        {"item_label": "book", "text_desc": "red book on study_desk", "spatial": {"location_semantic": "study_desk"}, "temporal": {"timestamp_utc": "2024-10-01T15:00:00+00:00"}, "confidence": 0.85}
    ]
    
    for e in demo_texts:
        memory_id = add_memory_entry_simple(e["text_desc"], e)
        log_info(f"Added memory entry: {memory_id}", e)

    # 显示数据库统计信息
    stats = get_memory_stats()
    log_info("Memory database statistics", stats)

    # 用户指令
    user_cmd = "帮我找到蓝色水杯，应该在书房的书桌上。"
    log_info("User command received", {"command": user_cmd})
    
    # 解析用户意图
    intent_result = parse_user_intent_llm(user_cmd)
    if not intent_result.is_success():
        log_error("Intent parsing failed", data={"error_msg": intent_result.error_msg, "status": intent_result.status.value})
        return
    
    intent = intent_result.data
    log_info("Intent parsed successfully", intent)
    
    # 确定目标房间
    target_room = "study_room"  # 根据指令确定目标房间
    log_info(f"Target room determined: {target_room}")
    
    # 开始导航任务
    log_info("=" * 40)
    log_info("STARTING NAVIGATION TASK")
    log_info("=" * 40)
    
    # 步骤1: 在客厅进行初始检测
    log_info(f"Step 1: Initial detection in {navigator.current_location}")
    perform_vision_detection_in_room(navigator, intent, "initial_detection")
    
    # 步骤2: 导航到目标房间
    log_info(f"Step 2: Navigating to target room: {target_room}")
    if navigator.navigate_to_room(target_room):
        log_info(f"Successfully navigated to {target_room}")
        
        # 步骤3: 在目标房间进行详细检测
        log_info(f"Step 3: Detailed detection in {navigator.current_location}")
        perform_vision_detection_in_room(navigator, intent, "target_detection")
    else:
        log_error(f"Failed to navigate to {target_room}")
    
    # 步骤4: 生成最终报告
    log_info("=" * 40)
    log_info("TASK COMPLETION REPORT")
    log_info("=" * 40)
    log_info(f"Navigation path: {' -> '.join(navigator.navigation_history + [navigator.current_location])}")
    log_info(f"Final location: {navigator.current_location}")
    log_info("Task simulation completed")

def perform_vision_detection_in_room(navigator, intent, detection_type):
    """在指定房间进行视觉检测"""
    log_info(f"Performing {detection_type} in {navigator.current_location}")
    
    # 获取当前房间图片
    image_path = navigator.get_current_image_path()
    log_info(f"Using image: {image_path}")
    
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # 根据检测类型调整目标对象
        if detection_type == "initial_detection":
            # 初始检测：寻找所有可能的对象
            target_objects = None
            log_info("Initial detection: scanning for all objects")
        else:
            # 目标检测：专注于目标对象
            target_objects = [intent.get('object')] if intent.get('object') else None
            log_info(f"Target detection: looking for {target_objects}")
        
        # 执行多对象检测
        vision_result = call_chatglm_multi_object_detection(base64_image, target_objects)
        
        if vision_result.is_success():
            log_info(f"Vision detection successful in {navigator.current_location}", {
                "detection_type": detection_type,
                "objects_found": len(vision_result.data.get("objects", []))
            })
            
            # 将视觉结果写回记忆区
            if vision_result.metadata.get("detection_type") == "multi_object":
                log_info("Integrating vision results into memory database")
                vision_memory_ids = add_vision_memory(vision_result.data, image_path)
                log_info(f"Added {len(vision_memory_ids)} vision memories to database")
            
            # 分析检测结果
            analyze_detection_results(vision_result.data, intent, navigator.current_location, detection_type)
            
        else:
            log_warning(f"Vision detection failed in {navigator.current_location}: {vision_result.error_msg}")
            
    except Exception as e:
        log_error(f"Error during vision detection in {navigator.current_location}", e)

def analyze_detection_results(vision_data, intent, location, detection_type):
    """分析视觉检测结果"""
    objects = vision_data.get("objects", [])
    target_object = intent.get("object", "")
    
    log_info(f"Analyzing detection results in {location}", {
        "detection_type": detection_type,
        "total_objects": len(objects),
        "target_object": target_object
    })
    
    # 检查是否找到目标对象
    target_found = False
    for obj in objects:
        obj_label = obj.get("label", "").lower()
        if target_object and target_object.lower() in obj_label:
            target_found = True
            log_info(f"TARGET FOUND: {obj['label']} with confidence {obj['confidence']}", obj)
            break
    
    if not target_found and detection_type == "target_detection":
        log_warning(f"Target object '{target_object}' not found in {location}")
        
        # 显示所有检测到的对象
        log_info("All detected objects:", [obj['label'] for obj in objects])
    
    return target_found

# ----------------- Main flow -----------------
if __name__ == "__main__":
    simulate_robot_navigation_task()
