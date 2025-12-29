# rag_chatglm_framework.py
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
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
from pathlib import Path

# 导入 MemoryGraph
try:
    from memory_graph import MemoryGraph
    MEMORY_GRAPH_AVAILABLE = True
except ImportError:
    MEMORY_GRAPH_AVAILABLE = False
    print("警告: MemoryGraph 未导入，请确保 memory_graph.py 在项目根目录")

# 导入 WorldModel
try:
    from world_model import WorldModel
    WORLD_MODEL_AVAILABLE = True
except ImportError:
    WORLD_MODEL_AVAILABLE = False
    print("警告: WorldModel 未导入，请确保 world_model.py 在项目根目录")

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

# 全局变量初始化（在 init_chroma_db 调用前设置为 None）
chroma_client = None
chroma_collection = None
memory_graph = None
world_model = None

# 初始化Chroma数据库
def init_chroma_db():
    """初始化Chroma向量数据库"""
    global chroma_client, chroma_collection, memory_graph, world_model
    
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
            
            # 检查现有集合的维度是否匹配
            try:
                # 尝试获取一个测试向量来检查维度
                test_results = collection.peek(limit=1)
                if test_results.get('embeddings') and len(test_results['embeddings']) > 0:
                    existing_dim = len(test_results['embeddings'][0])
                    if existing_dim != EMB_DIM:
                        log_warning(f"现有集合维度不匹配 (现有: {existing_dim}, 需要: {EMB_DIM})，删除并重建")
                        client.delete_collection(name=COLLECTION_NAME)
                        collection = client.create_collection(
                            name=COLLECTION_NAME,
                            metadata={"description": "机器人记忆存储"}
                        )
                        log_info(f"重新创建集合: {COLLECTION_NAME}")
                    else:
                        log_info(f"加载现有集合: {COLLECTION_NAME} (维度: {existing_dim})")
                else:
                    log_info(f"加载现有集合: {COLLECTION_NAME}")
            except Exception as check_e:
                log_warning(f"检查集合维度时出错: {check_e}，尝试重新创建")
                try:
                    client.delete_collection(name=COLLECTION_NAME)
                except:
                    pass
                collection = client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"description": "机器人记忆存储"}
                )
                log_info(f"重新创建集合: {COLLECTION_NAME}")
        except Exception:
            # 集合不存在，创建新集合
            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "机器人记忆存储"}
            )
            log_info(f"创建新集合: {COLLECTION_NAME}")
        
        log_info(f"Chroma数据库初始化成功: {VECTOR_DB_DIR}")
        
        # 初始化 MemoryGraph
        if MEMORY_GRAPH_AVAILABLE:
            try:
                db_path = Path(VECTOR_DB_DIR) / "memory_graph.sqlite"
                memory_graph = MemoryGraph(
                    db_path=str(db_path),
                    chroma_client=client
                )
                log_info(f"MemoryGraph 初始化成功: {db_path}")
                
                # 启动周期性衰减线程（如果启用）
                if os.getenv("RAG_ENABLE_PERIODIC_DECAY") == "1":
                    _start_periodic_decay_thread(memory_graph)
                    log_info("周期性衰减线程已启动（每600秒执行一次）")
            except Exception as e:
                log_warning(f"MemoryGraph 初始化失败: {e}")
                memory_graph = None
        else:
            memory_graph = None
        
        # 初始化 WorldModel 单例
        world_model = None
        if WORLD_MODEL_AVAILABLE:
            try:
                # 使用简单的 logger 包装器
                def world_model_logger(msg, *args, **kwargs):
                    """WorldModel 日志记录器"""
                    if isinstance(msg, tuple) and len(msg) == 2:
                        # 处理 (function_name, data) 格式
                        func_name, data = msg
                        log_info(f"WorldModel.{func_name}: {data}")
                    else:
                        log_info(f"WorldModel: {msg}")
                
                db_path = Path(VECTOR_DB_DIR) / "world_model.sqlite"
                world_model = WorldModel(
                    db_path=str(db_path),
                    logger=world_model_logger
                )
                log_info(f"WorldModel 初始化成功: {db_path}")
            except Exception as e:
                log_warning(f"WorldModel 初始化失败: {e}")
                world_model = None
        else:
            world_model = None
        
        # 设置全局变量
        chroma_client = client
        chroma_collection = collection
        # memory_graph 和 world_model 已在上面设置
        
        # 初始化 WorldModel 单例（作为模块属性）
        # 使用 import sys 和 sys.modules 来设置模块属性
        import sys
        current_module = sys.modules[__name__]
        if not hasattr(current_module, "world_model") or current_module.world_model is None:
            current_module.world_model = world_model
        
        return client, collection
    except Exception as e:
        log_error("初始化Chroma数据库失败", e)
        return None



def world_model_ingest(obs: Dict) -> Optional[Dict]:
    """
    将观察添加到 WorldModel
    
    Args:
        obs: 观察字典，包含：
            - label: 对象标签
            - bbox: 边界框
            - position: 位置 [x, y, z]
            - rotation: 旋转
            - depth_stats: 深度统计
            - confidence: 置信度
            - image_path: 图像路径
            - ts: 时间戳
            - fused_3d_pos: 融合后的3D位置（可选）
            - memnode_id: MemoryGraph 节点 ID（可选）
    
    Returns:
        结果字典，包含 entity_id 和 action，如果 world_model 未初始化则返回 None
    """
    global world_model
    if world_model is None:
        log_warning("WorldModel 未初始化，无法添加观察")
        return None
    
    try:
        result = world_model.ingest_observation(obs)
        log_info(f"WorldModel 观察已添加: entity_id={result.get('entity_id')}, action={result.get('action')}")
        return result
    except Exception as e:
        log_error("WorldModel 添加观察失败", e)
        return None


def world_model_mark_no_detection(entity_id: str, evidence: Dict) -> bool:
    """
    记录一次 no-detect 证据
    
    Args:
        entity_id: 实体 ID
        evidence: 证据字典，包含 ts, agent_pose, vis_score, frame_id, depth_stats 等
    
    Returns:
        是否成功
    """
    global world_model
    if world_model is None:
        log_warning("WorldModel 未初始化，无法标记未检测")
        return False
    
    try:
        result = world_model.mark_no_detection(entity_id, evidence)
        if result:
            log_info(f"WorldModel 标记未检测: entity_id={entity_id}, vis_score={evidence.get('vis_score', 0)}")
        return result
    except Exception as e:
        log_error("WorldModel 标记未检测失败", e)
        return False


def world_model_evaluate_missing(entity_id: str, n_no_detects: int = 3, vis_th: float = 0.6) -> bool:
    """
    累积证据后判断是否进入 missing 状态
    
    Args:
        entity_id: 实体 ID
        n_no_detects: 需要多少次未检测到
        vis_th: 视觉分数阈值
    
    Returns:
        是否进入 missing 状态
    """
    global world_model
    if world_model is None:
        log_warning("WorldModel 未初始化，无法评估 missing 状态")
        return False
    
    try:
        result = world_model.evaluate_missing(entity_id, n_no_detects=n_no_detects, vis_th=vis_th)
        if result:
            log_info(f"WorldModel 实体进入 missing 状态: entity_id={entity_id}")
        return result
    except Exception as e:
        log_error("WorldModel 评估 missing 状态失败", e)
        return False


def world_model_confirm_removed(entity_id: str, removed_T: float = 30*60) -> bool:
    """
    确认实体已被移除（基于时间阈值）
    
    Args:
        entity_id: 实体 ID
        removed_T: 移除时间阈值（秒），默认30分钟
    
    Returns:
        是否确认移除
    """
    global world_model
    if world_model is None:
        log_warning("WorldModel 未初始化，无法确认移除")
        return False
    
    try:
        result = world_model.confirm_removed(entity_id, removed_T=removed_T)
        if result:
            log_info(f"WorldModel 确认实体移除: entity_id={entity_id}")
        return result
    except Exception as e:
        log_error("WorldModel 确认移除失败", e)
        return False


def world_model_relink(obs: Dict) -> Optional[str]:
    """
    尝试将观察重新链接到现有实体（基于空间接近度）
    
    Args:
        obs: 观察字典，包含 label, fused_3d_pos (optional), embedding (optional)
    
    Returns:
        匹配的 entity_id 或 None
    """
    global world_model
    if world_model is None:
        log_warning("WorldModel 未初始化，无法重新链接")
        return None
    
    try:
        result = world_model.relink_candidate(obs)
        if result:
            log_info(f"WorldModel 重新链接成功: entity_id={result}, label={obs.get('label', 'unknown')}")
        return result
    except Exception as e:
        log_error("WorldModel 重新链接失败", e)
        return None


def world_model_register_movement(entity_id: str, new_pos: List[float], method: str = "triangulate", confidence: float = 0.8) -> Optional[Dict]:
    """
    注册实体移动
    
    Args:
        entity_id: 实体 ID
        new_pos: 新位置 [x, y, z]
        method: 定位方法
        confidence: 置信度
    
    Returns:
        移动事件字典或 None
    """
    global world_model
    if world_model is None:
        log_warning("WorldModel 未初始化，无法注册移动")
        return None
    
    try:
        result = world_model.register_movement(entity_id, new_pos, method=method, confidence=confidence)
        if result:
            log_info(f"WorldModel 注册移动: entity_id={entity_id}, from={result.get('from')}, to={result.get('to')}")
        return result
    except Exception as e:
        log_error("WorldModel 注册移动失败", e)
        return None


def add_or_update_memory(fused_result: Dict) -> Optional[str]:
    """
    将融合结果添加到或更新到 MemoryGraph
    
    Args:
        fused_result: 融合结果字典，包含：
            - label: 对象标签
            - position: 3D位置 [x, y, z] (fused_3d_pos)
            - confidence: 置信度
            - bbox: 边界框（可选）
            - image_path: 图像路径（可选）
            - agent_position: 机器人位置（可选）
            - agent_rotation: 机器人旋转（可选）
            - ts: 时间戳（可选）
            - metadata: 其他元数据（可选）
    
    Returns:
        MemoryGraph 节点 ID，如果失败则返回 None
    """
    global memory_graph
    if memory_graph is None:
        log_warning("MemoryGraph 未初始化，无法添加记忆")
        return None
    
    try:
        # 构建 evidence 字典
        evidence = {
            "type": "fused_observation",
            "label": fused_result.get("label", "unknown"),
            "confidence": fused_result.get("confidence", 0.5),
            "ts": fused_result.get("ts", time.time())
        }
        
        # 添加位置信息
        position = fused_result.get("fused_3d_pos") or fused_result.get("position")
        if position:
            evidence["position"] = position
        
        # 添加元数据
        metadata = fused_result.get("metadata", {})
        if fused_result.get("bbox"):
            metadata["bbox"] = fused_result["bbox"]
        if fused_result.get("image_path"):
            metadata["image_path"] = fused_result["image_path"]
        if fused_result.get("agent_position"):
            metadata["agent_position"] = fused_result["agent_position"]
        if fused_result.get("agent_rotation"):
            metadata["agent_rotation"] = fused_result["agent_rotation"]
        
        if metadata:
            evidence["metadata"] = metadata
        
        # 添加到 MemoryGraph
        node_id = memory_graph.add_evidence(evidence)
        log_info(f"MemoryGraph 添加/更新记忆: node_id={node_id}, label={evidence['label']}")
        return node_id
    
    except Exception as e:
        log_error("MemoryGraph 添加记忆失败", e)
        return None


def process_fused_result(fused_result: Dict) -> Dict[str, Any]:
    """
    统一处理融合结果：写入 MemoryGraph 并同步到 WorldModel
    
    Args:
        fused_result: 融合结果字典，包含：
            - label: 对象标签
            - fused_3d_pos: 融合后的3D位置 [x, y, z]
            - confidence: 置信度
            - bbox: 边界框（可选）
            - image_path: 图像路径（可选）
            - agent_position: 机器人位置（可选）
            - agent_rotation: 机器人旋转（可选）
            - ts: 时间戳（可选）
            - metadata: 其他元数据（可选）
    
    Returns:
        处理结果字典，包含：
            - memory_node_id: MemoryGraph 节点 ID
            - entity_id: WorldModel 实体 ID
            - action: "created" | "updated" | "moved" | "relinked"
            - movement_event: 移动事件（如果有）
    """
    global memory_graph, world_model
    
    result = {
        "memory_node_id": None,
        "entity_id": None,
        "action": None,
        "movement_event": None
    }
    
    # 1. 写入 MemoryGraph
    memory_node_id = add_or_update_memory(fused_result)
    result["memory_node_id"] = memory_node_id
    
    # 2. 同步到 WorldModel
    if world_model is None:
        log_warning("WorldModel 未初始化，跳过同步")
        return result
    
    try:
        label = fused_result.get("label", "unknown")
        fused_3d_pos = fused_result.get("fused_3d_pos") or fused_result.get("position")
        confidence = fused_result.get("confidence", 0.5)
        ts = fused_result.get("ts", time.time())
        
        if not fused_3d_pos:
            log_warning(f"融合结果缺少位置信息，跳过 WorldModel 同步: label={label}")
            return result
        
        # 构建观察字典用于 relink
        obs = {
            "label": label,
            "fused_3d_pos": fused_3d_pos,
            "confidence": confidence,
            "ts": ts
        }
        
        # 尝试重新链接到现有实体（基于空间相似性）
        linked_entity_id = world_model.relink_candidate(obs, spatial_th=0.3)
        
        if linked_entity_id:
            # 已链接：检查移动并更新
            old_ent = world_model.get_entity(linked_entity_id)
            if old_ent and old_ent.get("position"):
                old_pos = np.array(old_ent["position"])
                new_pos = np.array(fused_3d_pos)
                dist = np.linalg.norm(new_pos - old_pos)
                
                if dist > 0.5:  # 移动阈值 0.5m
                    # 注册移动事件
                    movement_event = world_model.register_movement(
                        linked_entity_id,
                        fused_3d_pos,
                        method="triangulate",
                        confidence=confidence
                    )
                    result["movement_event"] = movement_event
                    result["action"] = "moved"
                    log_info(f"WorldModel: 实体移动 (entity_id={linked_entity_id}, distance={dist:.3f}m)")
                else:
                    # 更新位置历史（通过 ingest_observation）
                    obs_for_wm = {
                        "label": label,
                        "fused_3d_pos": fused_3d_pos,
                        "confidence": confidence,
                        "ts": ts,
                        "memnode_id": memory_node_id
                    }
                    world_model.ingest_observation(obs_for_wm)
                    result["action"] = "updated"
            else:
                result["action"] = "relinked"
            
            result["entity_id"] = linked_entity_id
        else:
            # 未链接：创建新实体
            new_entity = world_model.create_entity(
                label=label,
                position=fused_3d_pos,
                confidence=confidence,
                ts=ts
            )
            entity_id = new_entity["entity_id"]
            
            # 将 memory_node_id 关联到实体
            if memory_node_id:
                # 更新实体的 fused_node_id
                entity_data = world_model.get_entity(entity_id)
                if entity_data:
                    entity_data["fused_node_id"] = memory_node_id
                    world_model._save_entity(entity_data)
            
            result["entity_id"] = entity_id
            result["action"] = "created"
            log_info(f"WorldModel: 创建新实体 (entity_id={entity_id}, label={label})")
    
    except Exception as e:
        log_error("WorldModel 同步失败", e)
    
    return result


def print_memory_world_summary():
    """
    打印 MemoryGraph 和 WorldModel 的摘要信息
    """
    global memory_graph, world_model
    
    print("\n" + "=" * 60)
    print("MemoryGraph & WorldModel 摘要")
    print("=" * 60)
    
    # MemoryGraph 摘要
    if memory_graph:
        try:
            conn = memory_graph._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nodes")
            node_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM edges")
            edge_count = cursor.fetchone()[0]
            conn.close()
            
            print(f"\nMemoryGraph:")
            print(f"  节点数: {node_count}")
            print(f"  边数: {edge_count}")
            
            # 获取最近的节点
            conn = memory_graph._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, data_json, fused_conf FROM nodes ORDER BY updated_ts DESC LIMIT 5")
            recent_nodes = cursor.fetchall()
            conn.close()
            
            if recent_nodes:
                print(f"  最近5个节点:")
                for node_id, data_json, conf in recent_nodes:
                    try:
                        data = json.loads(data_json)
                        label = data.get("label", "unknown")
                        pos = data.get("position")
                        pos_str = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})" if pos else "N/A"
                        print(f"    - {node_id[:8]}...: {label} @ {pos_str} (conf: {conf:.3f})")
                    except:
                        print(f"    - {node_id[:8]}...: (解析失败)")
        except Exception as e:
            print(f"  MemoryGraph 摘要获取失败: {e}")
    else:
        print("\nMemoryGraph: 未初始化")
    
    # WorldModel 摘要
    if world_model:
        try:
            # 统计实体状态
            active_count = 0
            missing_count = 0
            removed_count = 0
            
            for eid, ent in world_model.entities.items():
                status = ent.get("status", "active")
                if status == "active":
                    active_count += 1
                elif status == "missing":
                    missing_count += 1
                elif status == "removed":
                    removed_count += 1
            
            print(f"\nWorldModel:")
            print(f"  活跃实体: {active_count}")
            print(f"  Missing 实体: {missing_count}")
            print(f"  已移除实体: {removed_count}")
            print(f"  总实体数: {len(world_model.entities)}")
            
            # 获取最近的实体
            if world_model.entities:
                print(f"  最近5个实体:")
                sorted_entities = sorted(
                    world_model.entities.items(),
                    key=lambda x: x[1].get("last_seen_ts", 0),
                    reverse=True
                )[:5]
                
                for eid, ent in sorted_entities:
                    label = ent.get("label", "unknown")
                    status = ent.get("status", "active")
                    pos = ent.get("position")
                    pos_str = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})" if pos else "N/A"
                    print(f"    - {eid[:8]}...: {label} [{status}] @ {pos_str}")
        except Exception as e:
            print(f"  WorldModel 摘要获取失败: {e}")
    else:
        print("\nWorldModel: 未初始化")
    
    print("=" * 60 + "\n")



def _start_periodic_decay_thread(memory_graph_instance):
    """
    启动周期性衰减线程
    
    Args:
        memory_graph_instance: MemoryGraph 实例
    """
    global _decay_thread
    
    def decay_worker():
        """衰减工作线程"""
        while True:
            try:
                time.sleep(600)  # 每600秒执行一次
                if memory_graph_instance:
                    updated_count = memory_graph_instance.decay_nodes()
                    if updated_count > 0:
                        log_info(f"周期性衰减完成，更新了 {updated_count} 个节点")
            except Exception as e:
                log_error("周期性衰减线程出错", e)
    
    if _decay_thread is None or not _decay_thread.is_alive():
        _decay_thread = threading.Thread(target=decay_worker, daemon=True)
        _decay_thread.start()

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

# 尝试导入 langchain_openai，如果失败则使用替代方案
try:
    from langchain_openai import AzureOpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("警告: langchain_openai 未安装，某些功能可能不可用。请运行: pip install langchain-openai")
    # 定义一个占位类以避免后续错误
    class AzureOpenAIEmbeddings:
        pass

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

# ----------------- Time Context Extraction -----------------
def extract_time_context(timestamp: float) -> Dict[str, Any]:
    """
    从 unix timestamp 提取时间语义上下文
    
    Args:
        timestamp: Unix 时间戳（秒）
    
    Returns:
        {
            "hour": int,          # 0-23
            "weekday": int,       # 0=Monday, 6=Sunday
            "period": str,        # "dawn"|"morning"|"afternoon"|"evening"|"night"
            "is_weekend": bool    # True if Saturday or Sunday
        }
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    hour = dt.hour
    weekday = dt.weekday()  # 0=Monday, 6=Sunday
    
    # 时间段定义
    if 0 <= hour < 6:
        period = "dawn"
    elif 6 <= hour < 12:
        period = "morning"
    elif 12 <= hour < 18:
        period = "afternoon"
    elif 18 <= hour < 22:
        period = "evening"
    else:  # 22 <= hour < 24
        period = "night"
    
    is_weekend = weekday >= 5  # Saturday=5, Sunday=6
    
    return {
        "hour": hour,
        "weekday": weekday,
        "period": period,
        "is_weekend": is_weekend
    }

def get_azure_embedding_instance():
    """获取 Azure Embedding 对象"""
    if not LANGCHAIN_AVAILABLE:
        log_warning("langchain_openai 未安装，无法使用 AzureOpenAIEmbeddings")
        return None
    try:
        return AzureOpenAIEmbeddings(
            deployment=AZURE_DEPLOYMENT,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            model="text-embedding-ada-002"
        )
    except Exception as e:
        log_warning(f"创建 AzureOpenAIEmbeddings 失败: {e}")
        return None

# 延迟初始化，避免导入时错误
azure_embeddings = None
def _init_azure_embeddings():
    global azure_embeddings
    if azure_embeddings is None:
        azure_embeddings = get_azure_embedding_instance()
    return azure_embeddings

# ----------------- Embedding via Azure -----------------
def get_embedding_from_api(text: str, dimensions: int = EMB_DIM) -> np.ndarray:
    """
    从 Azure API 获取文本嵌入
    
    注意：如果 langchain_openai 未安装，此函数会返回零向量
    返回 np.ndarray 类型，自动归一化。
    """
    try:
        # 延迟初始化 azure_embeddings
        _init_azure_embeddings()
        if azure_embeddings is None:
            log_warning("azure_embeddings 未初始化（langchain_openai 未安装），返回零向量")
            return np.zeros(dimensions, dtype=np.float32)
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
        
        # 提取时间上下文（如果存在）
        time_context = meta.get("time_context")
        if time_context is None:
            # 如果没有提供 time_context，从 timestamp 提取
            timestamp_utc = meta.get("temporal", {}).get("timestamp_utc")
            if timestamp_utc:
                try:
                    if isinstance(timestamp_utc, str):
                        ts = datetime.fromisoformat(timestamp_utc).timestamp()
                    else:
                        ts = float(timestamp_utc)
                    time_context = extract_time_context(ts)
                except:
                    # 如果解析失败，使用当前时间
                    time_context = extract_time_context(time.time())
            else:
                # 如果没有 timestamp，使用当前时间
                time_context = extract_time_context(time.time())
        
        # 准备元数据
        chroma_metadata = {
            "item_label": meta.get("item_label", "unknown"),
            "location_semantic": meta.get("spatial", {}).get("location_semantic", "unknown"),
            "timestamp_utc": meta.get("temporal", {}).get("timestamp_utc", datetime.now(timezone.utc).isoformat()),
            "confidence": meta.get("confidence", 0.5),
            "text_desc": text_desc,
            "created_at": datetime.now(timezone.utc).isoformat(),
            # 添加时间上下文到 metadata（作为字符串存储，便于查询）
            "time_context_hour": str(time_context.get("hour", 0)),
            "time_context_period": time_context.get("period", "unknown"),
            "time_context_weekday": str(time_context.get("weekday", 0)),
            "time_context_is_weekend": str(time_context.get("is_weekend", False))
        }
        
        # 将完整的 time_context 作为 JSON 字符串存储（用于后续分析）
        import json
        chroma_metadata["time_context_json"] = json.dumps(time_context)
        
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
        
        # 使用Azure embedding生成查询向量
        query_embedding = get_embedding_from_api(query_text).tolist()
        
        # 执行查询 - 使用query_embeddings而不是query_texts以确保维度一致
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
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
    """
    基于意图进行智能检索，使用 Chroma 向量检索 + MemoryGraph 重排序
    
    Args:
        intent: 意图字典，包含 object, location 等字段
        topk: 返回的候选数量
    
    Returns:
        排序后的候选列表，每个元素包含:
        - id: 节点 ID（兼容旧格式）
        - node_id: 节点 ID
        - score: 最终评分
        - fused_conf: 融合置信度
        - meta: 元数据（包含 spatial, temporal 等信息）
    """
    global chroma_collection, memory_graph
    
    # 构建查询文本
    object_name = intent.get("object", "")
    location = intent.get("location", "")
    query_text = object_name or "object"
    
    # 如果 MemoryGraph 可用，使用新的检索流程
    if memory_graph is not None:
        try:
            # 1. 调用 Chroma 向量检索获取 topN (N >= topk*3)
            if chroma_collection is None:
                log_warning("Chroma collection 未初始化，回退到旧方法")
                return _retrieve_candidates_by_intent_fallback(intent, topk)
            
            # 生成查询向量
            query_embedding = get_embedding_from_api(query_text).tolist()
            
            # 查询 Chroma（获取更多候选以便重排序）
            n_results = max(topk * 3, 30)
            chroma_results = chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            if not chroma_results.get("ids") or not chroma_results["ids"][0]:
                log_warning("Chroma 查询无结果，回退到旧方法")
                return _retrieve_candidates_by_intent_fallback(intent, topk)
            
            chroma_ids = chroma_results["ids"][0]
            chroma_distances = chroma_results.get("distances", [[]])[0] if chroma_results.get("distances") else []
            chroma_metadatas = chroma_results.get("metadatas", [[]])[0] if chroma_results.get("metadatas") else []
            
            # 2. 将 Chroma hits 映射到 MemoryGraph node ids
            candidates = []
            for i, chroma_id in enumerate(chroma_ids):
                # 尝试从 metadata 获取 node_id，或直接使用 chroma_id
                metadata = chroma_metadatas[i] if i < len(chroma_metadatas) else {}
                node_id = metadata.get("node_id") or chroma_id
                
                # 计算余弦相似度（从距离转换）
                distance = chroma_distances[i] if i < len(chroma_distances) else 1.0
                cos_sim = 1.0 - distance  # Chroma 返回的是距离，转换为相似度
                
                candidates.append({
                    "node_id": node_id,
                    "chroma_id": chroma_id,
                    "cos_sim": cos_sim,
                    "metadata": metadata
                })
            
            # 3. 调用 MemoryGraph.query_by_intent 进行重排序
            # 获取 location_hint（如果有）
            location_hint = None
            if location:
                # 尝试从 intent 中获取位置坐标
                location_hint = intent.get("location_hint") or intent.get("position")
            
            ranked_nodes = memory_graph.query_by_intent(
                intent=intent,
                topk=topk,
                candidates=candidates,
                location_hint=location_hint
            )
            
            # 4. 转换为兼容格式
            results = []
            for node_result in ranked_nodes:
                node_id = node_result.get("node_id")
                node_data = node_result.get("data", {})
                fused_conf = node_result.get("fused_conf", 0.5)
                
                # 构建兼容格式
                result = {
                    "id": node_id,  # 兼容旧格式
                    "node_id": node_id,
                    "score": node_result.get("score", 0.5),
                    "fused_conf": fused_conf,
                    "meta": {
                        "text_desc": node_data.get("label", ""),
                        "item_label": node_data.get("label", ""),
                        "confidence": fused_conf,
                        "spatial": node_result.get("spatial", {}),
                        "temporal": node_result.get("temporal", {}),
                        "node_data": node_data  # 包含完整节点数据
                    }
                }
                results.append(result)
            
            log_info(f"MemoryGraph 重排序完成，返回 {len(results)} 个结果", {
                "query": query_text,
                "topk": topk,
                "fused_conf_range": [r["fused_conf"] for r in results[:3]] if results else []
            })
            
            return results
            
        except Exception as e:
            log_error("MemoryGraph 检索失败，回退到旧方法", e)
            return _retrieve_candidates_by_intent_fallback(intent, topk)
    else:
        # MemoryGraph 不可用，使用旧方法
        return _retrieve_candidates_by_intent_fallback(intent, topk)


def _retrieve_candidates_by_intent_fallback(intent: Dict[str, Any], topk: int = 20) -> List[Dict[str, Any]]:
    """回退方法：使用旧的检索逻辑"""
    # 构建增强的查询文本
    object_name = intent.get("object", "")
    location = intent.get("location", "")
    
    if object_name and location:
        # 组合对象和位置信息
        query_text = f"{object_name} in {location}"
        location_filter = location
        
        log_info("Enhanced query construction", {
            "original_object": object_name,
            "original_location": location,
            "enhanced_query": query_text,
            "location_filter": location_filter
        })
        
        # 尝试带位置过滤的查询
        results = retrieve_candidates_by_text(query_text, topk, location_filter)
        
        # 如果没有找到结果，放宽条件：不使用位置过滤
        if len(results) == 0:
            log_warning(f"No results found with location filter '{location}', trying without location filter")
            query_text = object_name
            location_filter = None
            results = retrieve_candidates_by_text(query_text, topk, location_filter)
        
        return results
    elif object_name:
        # 仅使用对象信息
        query_text = object_name
        location_filter = None
        
        log_info("Enhanced query construction", {
            "original_object": object_name,
            "original_location": location,
            "enhanced_query": query_text,
            "location_filter": location_filter
        })
        
        return retrieve_candidates_by_text(query_text, topk, location_filter)
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
        # 使用Azure embedding生成查询向量
        query_embedding = get_embedding_from_api("objects in location").tolist()
        
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
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

def time_prior_score(memory_item: Dict, current_time_context: Dict) -> float:
    """
    计算时间优先级分数
    
    Args:
        memory_item: 记忆项，包含 time_context 字段（可能在 meta 中）
        current_time_context: 当前时间上下文，由 extract_time_context 生成
    
    Returns:
        时间优先级分数 (0.8 - 1.8)
    """
    # 尝试从不同位置获取 time_context
    mc = memory_item.get("time_context")
    if not mc:
        mc = memory_item.get("meta", {}).get("time_context")
    if not mc:
        # 尝试从 metadata 中解析
        try:
            time_context_json = memory_item.get("meta", {}).get("time_context_json")
            if time_context_json:
                mc = json.loads(time_context_json)
        except:
            pass
    
    if not mc:
        return 1.0  # 如果没有时间上下文，返回默认分数
    
    # 如果时间段完全一致
    if mc.get("period") == current_time_context.get("period"):
        return 1.8
    
    # 如果小时差在 2 小时内
    hour_diff = abs(mc.get("hour", 0) - current_time_context.get("hour", 0))
    if hour_diff <= 2:
        return 1.2
    
    # 否则常规衰减
    return 0.8

def rerank_with_time_space(cands: List[Dict], current_pose: Optional[Dict] = None, now_ts: Optional[float] = None, current_time_context: Optional[Dict] = None) -> List[Dict]:
    """
    重新排序候选结果，考虑时间和空间因素
    
    Args:
        cands: 候选结果列表
        current_pose: 当前位置姿态
        now_ts: 当前时间戳
        current_time_context: 当前时间上下文（如果提供，避免重复计算）
    """
    if now_ts is None:
        now_ts = time.time()
    if current_time_context is None:
        current_time_context = extract_time_context(now_ts)
    
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
        
        # 计算时间优先级分数
        time_prior = time_prior_score(c, current_time_context)
        
        # 整合所有评分因子：视觉相似度、置信度、时间衰减、空间距离、时间优先级
        final = sim * 0.5 + conf * 0.2 + w_time * 0.1 + w_space * 0.05 + (time_prior - 1.0) * 0.15
        scored.append({**c, "final_score": final, "time_prior_score": time_prior})
    scored.sort(key=lambda e: e["final_score"], reverse=True)
    return scored

def analyze_time_patterns(object_name: str) -> Dict[str, Dict[str, float]]:
    """
    从 Chroma memory 中读取该物品的所有观察记录，按 period 统计出现频率
    
    Args:
        object_name: 物品名称
    
    Returns:
        {
            "morning": {"desk": 0.82, "kitchen": 0.12, ...},
            "afternoon": {...},
            "evening": {...},
            "night": {...},
            "dawn": {...}
        }
    """
    global chroma_collection
    
    if chroma_collection is None:
        log_error("Chroma数据库未初始化")
        return {}
    
    try:
        # 获取所有包含该物品的记忆
        # 使用向量检索找到相关记忆
        query_embedding = get_embedding_from_api(object_name).tolist()
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=1000,  # 获取足够多的结果
            where={"item_label": object_name}  # 如果支持精确匹配
        )
        
        # 统计每个时间段和位置的频率
        period_location_counts = {
            "dawn": {},
            "morning": {},
            "afternoon": {},
            "evening": {},
            "night": {}
        }
        
        total_by_period = {
            "dawn": 0,
            "morning": 0,
            "afternoon": 0,
            "evening": 0,
            "night": 0
        }
        
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                
                # 获取时间上下文
                time_context = None
                try:
                    time_context_json = metadata.get("time_context_json")
                    if time_context_json:
                        time_context = json.loads(time_context_json)
                    else:
                        # 尝试从其他字段构建
                        period = metadata.get("time_context_period")
                        if period:
                            time_context = {"period": period}
                except:
                    pass
                
                if not time_context:
                    continue
                
                period = time_context.get("period", "unknown")
                if period not in period_location_counts:
                    continue
                
                # 获取位置信息
                location = metadata.get("location_semantic", "unknown")
                
                # 统计
                if location not in period_location_counts[period]:
                    period_location_counts[period][location] = 0
                period_location_counts[period][location] += 1
                total_by_period[period] += 1
        
        # 转换为频率（概率）
        result = {}
        for period, location_counts in period_location_counts.items():
            if total_by_period[period] > 0:
                result[period] = {
                    loc: count / total_by_period[period]
                    for loc, count in location_counts.items()
                }
            else:
                result[period] = {}
        
        log_info(f"分析物品时间规律: {object_name}", {
            "total_observations": sum(total_by_period.values()),
            "periods": {p: total_by_period[p] for p in period_location_counts.keys()}
        })
        
        return result
        
    except Exception as e:
        log_error("分析时间规律失败", e, {"object_name": object_name})
        return {}

def retrieve_candidates_with_time_prior(intent: Dict[str, Any], current_time_context: Optional[Dict] = None, topk: int = 20) -> List[Dict[str, Any]]:
    """
    基于意图进行检索，并应用时间优先级重新排序
    
    Args:
        intent: 意图字典，包含 object, location 等字段
        current_time_context: 当前时间上下文（如果为 None，则自动提取）
        topk: 返回的候选数量
    
    Returns:
        按时间优先级重新排序的候选列表（已包含 fused_conf 和 MemoryGraph 重排序）
    """
    if current_time_context is None:
        current_time_context = extract_time_context(time.time())
    
    # 使用新的 retrieve_candidates_by_intent（已集成 MemoryGraph 重排序）
    candidates = retrieve_candidates_by_intent(intent, topk=topk)
    
    # 如果结果已包含 fused_conf，直接返回（MemoryGraph 已处理重排序）
    # 否则应用时间优先级重新排序（兼容旧格式）
    if candidates and "fused_conf" in candidates[0]:
        # 新格式：已包含 MemoryGraph 重排序，直接返回
        return candidates
    else:
        # 旧格式：应用时间优先级重新排序
        reranked = rerank_with_time_space(
            candidates,
            current_pose=None,
            now_ts=None,
            current_time_context=current_time_context
        )
        return reranked[:topk]

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
def _classify_vision_error(error: Exception, response: Optional[requests.Response] = None) -> str:
    """
    分类视觉API错误
    
    Returns:
        错误类型: "network", "timeout", "5xx", "4xx", "malformed_json", "unknown"
    """
    if isinstance(error, requests.Timeout):
        return "timeout"
    elif isinstance(error, requests.ConnectionError):
        return "network"
    elif isinstance(error, requests.RequestException):
        if response is not None:
            status_code = response.status_code
            if 500 <= status_code < 600:
                return "5xx"
            elif 400 <= status_code < 500:
                return "4xx"
        return "network"
    elif isinstance(error, (json.JSONDecodeError, ValueError)):
        return "malformed_json"
    else:
        return "unknown"


def _safe_parse_json(raw_text: str, max_retries: int = 3) -> Tuple[Optional[Dict], str]:
    """
    安全解析JSON，支持多种格式和清理
    
    Args:
        raw_text: 原始文本
        max_retries: 最大重试次数
    
    Returns:
        (parsed_dict, error_message)
    """
    import re
    
    # 尝试1: 直接解析
    try:
        return json.loads(raw_text), ""
    except json.JSONDecodeError:
        pass
    
    # 尝试2: 提取JSON块
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # 匹配嵌套JSON
        r'\{.*?\}',  # 简单匹配
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, raw_text, re.DOTALL)
        for match in matches:
            try:
                # 清理可能的markdown代码块标记
                cleaned = match.strip()
                if cleaned.startswith('```'):
                    # 移除markdown代码块标记
                    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
                    cleaned = re.sub(r'\s*```\s*$', '', cleaned)
                
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return parsed, ""
            except json.JSONDecodeError:
                continue
    
    # 尝试3: 修复常见的JSON错误
    try:
        # 移除注释
        cleaned = re.sub(r'//.*?$', '', raw_text, flags=re.MULTILINE)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        
        # 修复单引号
        cleaned = cleaned.replace("'", '"')
        
        # 修复尾随逗号
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        
        # 提取JSON
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0)), ""
    except (json.JSONDecodeError, AttributeError):
        pass
    
    return None, "Failed to parse JSON after multiple attempts"


def _call_chatglm_vision_robust(image_url: str, prompt: str, 
                                max_retries: int = 4, 
                                backoff_base: float = 0.5) -> Dict[str, Any]:
    """
    增强的视觉API调用，包含错误分类和指数退避
    
    Args:
        image_url: 图像URL
        prompt: 提示词
        max_retries: 最大重试次数
        backoff_base: 退避基础时间（秒）
    
    Returns:
        结构化输出字典:
        {
            "success": bool,
            "error": str or None,
            "error_type": str or None,
            "raw": str or None,
            "boxes": List[Dict],
            "labels": List[str],
            "confidence": float,
            "data": Dict or None,
            "retry_count": int
        }
    """
    content_list = [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": prompt}
    ]
    
    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": content_list}],
        "temperature": 0.1
    }
    
    backoff_times = [backoff_base * (2 ** i) for i in range(max_retries)]
    last_error = None
    last_response = None
    
    for attempt in range(max_retries):
        try:
            # 如果不是第一次尝试，执行退避
            if attempt > 0:
                sleep_time = backoff_times[attempt - 1]
                log_info(f"视觉API重试 {attempt}/{max_retries - 1}，等待 {sleep_time:.2f} 秒...")
                time.sleep(sleep_time)
            
            # 发送请求
            resp = requests.post(
                CHATGLM_VISION_URL, 
                json=payload, 
                headers=HEADERS, 
                timeout=60
            )
            last_response = resp
            
            # 检查HTTP状态码
            if resp.status_code >= 500:
                # 5xx错误，重试
                error_type = "5xx"
                error_msg = f"Server error {resp.status_code}: {resp.text[:200]}"
                log_warning("vision_error", {
                    "error_type": error_type,
                    "status_code": resp.status_code,
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "error": error_msg
                })
                
                if attempt < max_retries - 1:
                    continue
                else:
                    return {
                        "success": False,
                        "error": error_msg,
                        "error_type": error_type,
                        "raw": None,
                        "boxes": [],
                        "labels": [],
                        "confidence": 0.0,
                        "data": None,
                        "retry_count": attempt + 1
                    }
            
            elif resp.status_code >= 400:
                # 4xx错误，不重试
                error_type = "4xx"
                error_msg = f"Client error {resp.status_code}: {resp.text[:200]}"
                log_warning("vision_error", {
                    "error_type": error_type,
                    "status_code": resp.status_code,
                    "attempt": attempt + 1,
                    "error": error_msg
                })
                
                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": error_type,
                    "raw": None,
                    "boxes": [],
                    "labels": [],
                    "confidence": 0.0,
                    "data": None,
                    "retry_count": attempt + 1
                }
            
            # 成功响应，解析JSON
            data = resp.json()
            raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not raw_text:
                error_msg = "Empty response from vision API"
                log_warning("vision_error", {
                    "error_type": "empty_response",
                    "attempt": attempt + 1,
                    "error": error_msg
                })
                
                if attempt < max_retries - 1:
                    continue
                else:
                    return {
                        "success": False,
                        "error": error_msg,
                        "error_type": "empty_response",
                        "raw": None,
                        "boxes": [],
                        "labels": [],
                        "confidence": 0.0,
                        "data": None,
                        "retry_count": attempt + 1
                    }
            
            # 安全解析JSON
            parsed_result, parse_error = _safe_parse_json(raw_text)
            
            if parsed_result is None:
                # JSON解析失败
                if attempt < max_retries - 1:
                    # malformed_json只重试1次
                    if attempt == 0:
                        log_warning("vision_error", {
                            "error_type": "malformed_json",
                            "attempt": attempt + 1,
                            "error": parse_error,
                            "raw_text_preview": raw_text[:200]
                        })
                        continue
                    else:
                        # 已经重试过1次，不再重试
                        break
                else:
                    # 最后一次尝试，返回fallback结果
                    return {
                        "success": False,
                        "error": f"JSON parsing failed: {parse_error}",
                        "error_type": "malformed_json",
                        "raw": raw_text,
                        "boxes": [],
                        "labels": [],
                        "confidence": 0.0,
                        "data": None,
                        "retry_count": attempt + 1
                    }
            
            # 成功解析，提取结构化信息
            objects = parsed_result.get("objects", [])
            boxes = []
            labels = []
            confidences = []
            
            for obj in objects:
                if isinstance(obj, dict):
                    bbox = obj.get("bbox", [])
                    label = obj.get("label", "")
                    confidence = obj.get("confidence", 0.0)
                    
                    if bbox and label:
                        boxes.append(bbox)
                        labels.append(label)
                        confidences.append(float(confidence))
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            log_info("视觉API调用成功", {
                "attempt": attempt + 1,
                "objects_count": len(objects),
                "boxes_count": len(boxes)
            })
            
            return {
                "success": True,
                "error": None,
                "error_type": None,
                "raw": raw_text,
                "boxes": boxes,
                "labels": labels,
                "confidence": avg_confidence,
                "data": parsed_result,
                "retry_count": attempt + 1
            }
            
        except requests.Timeout as e:
            last_error = e
            error_type = "timeout"
            error_msg = f"Request timeout: {e}"
            log_warning("vision_error", {
                "error_type": error_type,
                "attempt": attempt + 1,
                "max_retries": max_retries,
                "error": error_msg
            })
            
            if attempt < max_retries - 1:
                continue
            else:
                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": error_type,
                    "raw": None,
                    "boxes": [],
                    "labels": [],
                    "confidence": 0.0,
                    "data": None,
                    "retry_count": attempt + 1
                }
        
        except requests.ConnectionError as e:
            last_error = e
            error_type = "network"
            error_msg = f"Network error: {e}"
            log_warning("vision_error", {
                "error_type": error_type,
                "attempt": attempt + 1,
                "max_retries": max_retries,
                "error": error_msg
            })
            
            if attempt < max_retries - 1:
                continue
            else:
                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": error_type,
                    "raw": None,
                    "boxes": [],
                    "labels": [],
                    "confidence": 0.0,
                    "data": None,
                    "retry_count": attempt + 1
                }
        
        except requests.RequestException as e:
            last_error = e
            error_type = _classify_vision_error(e, last_response)
            error_msg = f"Request error: {e}"
            log_warning("vision_error", {
                "error_type": error_type,
                "attempt": attempt + 1,
                "max_retries": max_retries,
                "error": error_msg
            })
            
            # 根据错误类型决定是否重试
            if error_type in ["5xx", "timeout", "network"] and attempt < max_retries - 1:
                continue
            else:
                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": error_type,
                    "raw": None,
                    "boxes": [],
                    "labels": [],
                    "confidence": 0.0,
                    "data": None,
                    "retry_count": attempt + 1
                }
        
        except Exception as e:
            last_error = e
            error_type = "unknown"
            error_msg = f"Unexpected error: {e}"
            log_warning("vision_error", {
                "error_type": error_type,
                "attempt": attempt + 1,
                "error": error_msg,
                "error_class": type(e).__name__
            })
            
            return {
                "success": False,
                "error": error_msg,
                "error_type": error_type,
                "raw": None,
                "boxes": [],
                "labels": [],
                "confidence": 0.0,
                "data": None,
                "retry_count": attempt + 1
            }
    
    # 所有重试都失败
    final_error_type = _classify_vision_error(last_error, last_response) if last_error else "unknown"
    return {
        "success": False,
        "error": f"All retries failed. Last error: {last_error}",
        "error_type": final_error_type,
        "raw": None,
        "boxes": [],
        "labels": [],
        "confidence": 0.0,
        "data": None,
        "retry_count": max_retries
    }


def call_chatglm_multi_object_detection(image_url: str, target_objects: List[str] = None) -> Result:
    """
    多对象检测和边界框验证（增强版，包含错误处理和fallback）
    """
    # 构建多对象检测prompt
    if target_objects:
        target_str = ", ".join(target_objects)
        detection_prompt = (
            f"请仔细分析图片中的所有对象，特别关注以下目标对象：{target_str}\n\n"
            "重要提示：\n"
            "- 目标对象必须作为独立的objects列表项输出\n"
            "- 即使目标对象作为其他对象的属性描述（如\"书桌上的水杯\"），也必须单独提取为独立对象\n"
            "- 如果一个物体是由多个部分组成的（如书桌和水杯），每个部分都应单独列出\n\n"
            "附加要求：\n"
            "1) 优先对目标对象输出中英双语标签（例如：\"钢琴/piano\"、\"三角钢琴/grand piano\"、\"立式钢琴/upright piano\"）。\n"
            "2) 若不确定但高度相关，也请输出该候选，同时降低confidence。\n\n"
            "请输出JSON格式，包含字段：\n"
            "{\n"
            "  \"objects\": [\n"
            "    {\n"
            "      \"label\": \"对象名称（建议中英双语，如 钢琴/piano）\",\n"
            "      \"bbox\": [x1, y1, x2, y2],\n"
            "      \"confidence\": 0.95,\n"
            "      \"description\": \"详细描述\"\n"
            "    }\n"
            "  ],\n"
            "  \"scene_description\": \"整体场景描述\",\n"
            "  \"detection_summary\": \"检测总结\"\n"
            "}\n\n"
            "要求：\n"
            "1. objects数组中每个元素必须是一个独立完整的物体（如\"水杯\"、\"书\"、\"电脑\"等）\n"
            "2. bbox格式：[左上角x, 左上角y, 右下角x, 右下角y]\n"
            "3. confidence范围：0.0-1.0\n"
            "4. 只输出JSON，不要其他文字\n"
            f"5. 对于目标对象\"{target_str}\"及其同义词，必须单独列出，不要只作为其他对象的描述\n"
            "6. 检测所有可见对象，包括目标对象\n"
        )
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

    # 使用增强的视觉API包装器
    result = _call_chatglm_vision_robust(image_url, detection_prompt, max_retries=4, backoff_base=0.5)
    
    # 如果成功，验证并返回
    if result["success"] and result["data"] is not None:
        parsed_result = result["data"]
                
        # 验证JSON结构
        if validate_detection_result(parsed_result):
            log_info("Multi-object detection successful", {
                "objects_count": len(parsed_result.get("objects", [])),
                "target_objects": target_objects,
                "retry_count": result["retry_count"]
            })
            return Result(StatusCode.OK, parsed_result, metadata={
                "raw_text": result["raw"],
                "detection_type": "multi_object",
                "retry_count": result["retry_count"]
            })
        else:
            log_warning("Invalid detection result structure", parsed_result)
            # 结构无效，尝试fallback
            return _fallback_vision_json_rectification(image_url, detection_prompt, result["raw"], target_objects)
    
    # 如果失败，尝试fallback
    if not result["success"]:
        log_warning("Vision API call failed, attempting fallback", {
            "error_type": result.get("error_type"),
            "error": result.get("error"),
            "retry_count": result.get("retry_count", 0)
        })
        return _fallback_vision_json_rectification(image_url, detection_prompt, result.get("raw"), target_objects)
    
    # 如果数据为空但成功，也尝试fallback
    return _fallback_vision_json_rectification(image_url, detection_prompt, result.get("raw"), target_objects)


def _fallback_vision_json_rectification(image_url: str, original_prompt: str, 
                                        raw_text: Optional[str], target_objects: List[str] = None) -> Result:
    """
    Fallback机制：使用LLM文本修复JSON
    
    Args:
        image_url: 图像URL
        original_prompt: 原始提示词
        raw_text: 原始响应文本（如果有）
        target_objects: 目标对象列表
    
    Returns:
        Result对象
    """
    try:
        # 构建fallback提示词
        if raw_text:
            fallback_prompt = (
                f"视觉API返回了以下文本，但JSON解析失败。请严格输出JSON格式，不要包含任何其他文字：\n\n"
                f"原始响应：\n{raw_text[:1000]}\n\n"
                f"原始要求：\n{original_prompt}\n\n"
                f"请严格输出JSON格式，确保：\n"
                f"1. 只输出JSON，不要markdown代码块标记\n"
                f"2. 确保所有字符串使用双引号\n"
                f"3. 确保没有尾随逗号\n"
                f"4. 确保所有字段都符合要求\n"
            )
        else:
            fallback_prompt = (
                f"视觉API调用失败，请基于以下要求直接输出JSON格式：\n\n"
                f"{original_prompt}\n\n"
                f"请严格输出JSON格式，不要包含任何其他文字。"
            )
        
        # 使用LLM进行JSON修复
        rectification_result = call_chatglm_llm_json(
            "你是一个JSON格式修复专家。请严格输出JSON格式，不要包含任何markdown标记或其他文字。",
            fallback_prompt,
            {
                "objects": list,
                "scene_description": str,
                "detection_summary": str
            }
        )
        
        if rectification_result.is_success():
            rectified_data = rectification_result.data
            
            # 验证修复后的JSON结构
            if validate_detection_result(rectified_data):
                log_info("JSON rectification successful via LLM fallback", {
                    "objects_count": len(rectified_data.get("objects", []))
                })
                return Result(StatusCode.OK, rectified_data, 
                            "Vision API failed, used LLM JSON rectification",
                            {
                                "fallback_used": True,
                                "fallback_type": "llm_json_rectification",
                                "raw_text": raw_text
                            })
            else:
                log_warning("Rectified JSON structure still invalid", rectified_data)
                return Result(StatusCode.VISION_ERROR, None, 
                            "Both vision API and JSON rectification failed",
                            {
                                "fallback_used": True,
                                "fallback_failed": True,
                                "raw_text": raw_text
                            })
        else:
            log_warning("LLM JSON rectification failed", {
                "error": rectification_result.error_msg
            })
            return Result(StatusCode.VISION_ERROR, None, 
                        f"Both vision API and JSON rectification failed. Vision: {raw_text[:200] if raw_text else 'N/A'}, Rectification: {rectification_result.error_msg}",
                        {
                            "fallback_used": True,
                            "fallback_failed": True,
                            "raw_text": raw_text
                        })
    
    except Exception as e:
        log_error("Fallback JSON rectification error", e)
        return Result(StatusCode.VISION_ERROR, None, 
                    f"Fallback error: {e}",
                    {
                        "fallback_used": True,
                        "fallback_error": str(e),
                        "raw_text": raw_text
                    })

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

def add_vision_memory(detection_result: Dict, image_path: str = None, 
                     agent_position: List[float] = None, agent_rotation: List[float] = None,
                     object_crop_paths: Dict[str, str] = None) -> List[str]:
    """
    将视觉检测结果写回记忆区，实现跨模态RAG
    
    Args:
        detection_result: 视觉检测结果字典
        image_path: 完整场景图像路径
        agent_position: 机器人当前位置 [x, y, z]
        agent_rotation: 机器人当前旋转 [x, y, z, w] (四元数)
        object_crop_paths: 物体标签到截图路径的映射 {label: crop_path}
    """
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
            
            # 获取该物体的截图路径
            object_crop_path = None
            if object_crop_paths and label in object_crop_paths:
                object_crop_path = object_crop_paths[label]
            
            # 构建增强的视觉记忆描述文本（包含位置信息）
            vision_desc = f"vision detected {label}"
            if description:
                vision_desc += f" - {description}"
            
            # 添加位置信息到描述中（用于语义检索）
            if agent_position:
                pos_str = f"at position ({agent_position[0]:.2f}, {agent_position[1]:.2f}, {agent_position[2]:.2f})"
                vision_desc += f" {pos_str}"
            
            # 获取当前时间戳并提取时间上下文
            ts = time.time()
            time_context = extract_time_context(ts)
            
            # 构建增强的元数据
            vision_metadata = {
                "item_label": label,
                "source": "vision_detection",
                "detection_confidence": confidence,
                "bbox": bbox,
                "description": description,
                "spatial": {
                    "location_semantic": "current_scene",
                    "source": "visual_analysis",
                    # 添加世界坐标位置
                    "world_position": agent_position if agent_position else None,
                    "agent_rotation": agent_rotation if agent_rotation else None,
                },
                "temporal": {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "observation_type": "real_time_vision"
                },
                "time_context": time_context,  # 添加时间语义上下文
                "metadata": {
                    "image_path": image_path,  # 完整场景图
                    "object_crop_path": object_crop_path,  # 物体截图路径
                    "detection_method": "chatglm_multi_object",
                    "scene_description": scene_description,
                    "has_position": agent_position is not None,
                    "has_crop": object_crop_path is not None
                }
            }
            
            # 添加到Chroma数据库
            memory_id = add_memory_entry_simple(vision_desc, vision_metadata)
            if memory_id:
                added_memory_ids.append(memory_id)
                log_info(f"Added enhanced vision memory: {memory_id}", {
                    "label": label,
                    "confidence": confidence,
                    "description": description,
                    "position": agent_position,
                    "crop_path": object_crop_path
                })
        
        # 添加整体场景记忆
        if scene_description:
            # 使用相同的时间上下文
            ts = time.time()
            time_context = extract_time_context(ts)
            
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
                "time_context": time_context,  # 添加时间语义上下文
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

def query_object_location(query_text: str, object_label: str = None, topk: int = 5) -> Dict[str, Any]:
    """
    查询物体的位置信息
    
    Args:
        query_text: 查询文本，例如："刚刚找到的那个杯子在哪个桌子上？" 或 "水杯在哪里？"
        object_label: 可选的物体标签，用于精确匹配
        topk: 返回前k个结果
    
    Returns:
        包含位置信息的字典
    """
    global chroma_collection
    
    if chroma_collection is None:
        log_error("Chroma数据库未初始化，无法查询位置")
        return {"error": "数据库未初始化"}
    
    try:
        # 检索相关记忆
        results = retrieve_candidates_by_text(query_text, topk=topk)
        
        location_results = []
        for result in results:
            meta = result.get("meta", {})
            spatial = meta.get("spatial", {})
            metadata = meta.get("metadata", {})
            
            # 检查是否有位置信息
            world_position = spatial.get("world_position")
            location_semantic = spatial.get("location_semantic", "unknown")
            object_crop_path = metadata.get("object_crop_path")
            image_path = metadata.get("image_path")
            
            # 如果指定了object_label，进行过滤
            if object_label:
                item_label = meta.get("item_label", "").lower()
                if object_label.lower() not in item_label:
                    continue
            
            location_info = {
                "item_label": meta.get("item_label", "unknown"),
                "description": meta.get("description", ""),
                "confidence": meta.get("detection_confidence", 0.0),
                "world_position": world_position,
                "location_semantic": location_semantic,
                "object_crop_path": object_crop_path,
                "scene_image_path": image_path,
                "timestamp": meta.get("temporal", {}).get("timestamp_utc", ""),
                "distance": result.get("distance", 0.0)
            }
            
            location_results.append(location_info)
        
        # 格式化位置信息用于回答
        if location_results:
            best_match = location_results[0]  # 最相关的结果
            position_str = ""
            if best_match["world_position"]:
                pos = best_match["world_position"]
                position_str = f"世界坐标: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            
            answer = f"找到物体 '{best_match['item_label']}'"
            if best_match["description"]:
                answer += f"，描述：{best_match['description']}"
            if position_str:
                answer += f"，{position_str}"
            if best_match["location_semantic"] and best_match["location_semantic"] != "unknown":
                answer += f"，语义位置：{best_match['location_semantic']}"
            if best_match["object_crop_path"]:
                answer += f"，截图路径：{best_match['object_crop_path']}"
            
            return {
                "found": True,
                "answer": answer,
                "best_match": best_match,
                "all_matches": location_results,
                "count": len(location_results)
            }
        else:
            return {
                "found": False,
                "answer": f"未找到物体 '{object_label or '目标物体'}' 的位置信息",
                "all_matches": [],
                "count": 0
            }
            
    except Exception as e:
        log_error("查询物体位置失败", e)
        return {"error": str(e), "found": False}

# ----------------- ChatGLM Vision call -----------------
def call_chatglm_vision_json(image_url: str, question: str) -> Result:
    """
    视觉API调用（增强版，包含错误处理和fallback）
    """
    # 使用增强的视觉API包装器
    result = _call_chatglm_vision_robust(image_url, question, max_retries=4, backoff_base=0.5)
    
    # 如果成功，返回结果
    if result["success"] and result["data"] is not None:
        return Result(StatusCode.OK, result["data"], metadata={
            "raw_text": result["raw"],
            "retry_count": result["retry_count"]
        })
    
    # 如果失败，尝试fallback JSON修复
    if not result["success"]:
        log_warning("Vision API call failed, attempting JSON rectification", {
            "error_type": result.get("error_type"),
            "error": result.get("error"),
            "retry_count": result.get("retry_count", 0)
        })
        
        # 如果有原始文本，尝试修复
        if result.get("raw"):
            parsed_result, parse_error = _safe_parse_json(result["raw"])
            if parsed_result is not None:
                return Result(StatusCode.OK, parsed_result, 
                            f"Vision API failed but JSON recovered: {result.get('error')}",
                            {
                                "raw_text": result["raw"],
                                "retry_count": result.get("retry_count", 0),
                                "json_recovered": True
                            })
        
        # 如果无法修复，返回错误
        return Result(StatusCode.VISION_ERROR, None, 
                    f"Vision API failed: {result.get('error')}",
                    {
                        "error_type": result.get("error_type"),
                        "retry_count": result.get("retry_count", 0),
                        "raw_text": result.get("raw")
                    })
    
    # 如果数据为空，返回原始文本
    if result.get("raw"):
        return Result(StatusCode.OK, {"text": result["raw"]}, 
                    metadata={"raw_text": result["raw"], "json_parse_failed": True})
    
    return Result(StatusCode.VISION_ERROR, None, "Vision API returned empty response")

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
        "\n关键指令：\n"
        "1. 首先检查 current_status 中的 target_found 和 visual_detection_result 字段。\n"
        "2. 如果 target_found=true 且 visual_detection_result 包含 'SUCCESS'，说明目标已经找到，下一步应该是操作步骤（如：移动到目标前、准备抓取、报告位置等），而不是搜索。\n"
        "3. 如果目标未找到，才需要生成搜索策略。\n"
        "4. 请根据目标是否已找到来制定合适的行动计划。"
    )
    user_prompt = f"intent={json.dumps(intent, ensure_ascii=False)}\nevi={json.dumps(evidences, ensure_ascii=False)}\nstatus={json.dumps(current_status, ensure_ascii=False)}\n\n请分析当前状态：\n1. 目标是否已找到？（检查 status 中的 target_found 和 visual_detection_result）\n2. 如果已找到，生成后续操作计划（如移动到目标、抓取、报告等）。\n3. 如果未找到，生成搜索策略。"
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
    
    # 开始导航任务
    log_info("=" * 40)
    log_info("STARTING NAVIGATION TASK")
    log_info("=" * 40)
    
    # 步骤1: 在客厅进行初始检测
    log_info(f"Step 1: Initial detection in {navigator.current_location}")
    perform_vision_detection_in_room(navigator, intent, "initial_detection")
    
    # 步骤2: 循环搜索所有可能的房间
    log_info("=" * 40)
    log_info("STARTING ROOM SEARCH CYCLE")
    log_info("=" * 40)
    
    # 获取所有可搜索的房间
    rooms_to_search = ["kitchen", "bedroom", "study_room"]  # 客厅的三个出口
    rooms_searched = []
    target_found = False
    found_location = None
    
    for room in rooms_to_search:
        log_info(f"\n{'='*60}")
        log_info(f"Attempting to search room: {room}")
        log_info(f"{'='*60}")
        
        # 导航到目标房间
        if navigator.navigate_to_room(room):
            log_info(f"Successfully navigated to {room}")
            rooms_searched.append(room)
            
            # 在目标房间进行检测
            log_info(f"Performing detection in {navigator.current_location}")
            target_found = perform_vision_detection_in_room(navigator, intent, "target_detection")
            
            if target_found:
                found_location = room
                log_info(f"\n{'='*60}")
                log_info(f"TARGET OBJECT FOUND in {room}!")
                log_info(f"{'='*60}\n")
                break
            else:
                log_info(f"Target object not found in {room}. Returning to living room.")
                # 返回客厅
                navigator.navigate_to_room("living_room")
                log_info(f"Returned to living room")
        else:
            log_error(f"Failed to navigate to {room}")
    
    # 步骤3: 生成最终报告和任务规划
    log_info("=" * 40)
    log_info("TASK COMPLETION REPORT")
    log_info("=" * 40)
    log_info(f"Navigation path: {' -> '.join(navigator.navigation_history + [navigator.current_location])}")
    log_info(f"Final location: {navigator.current_location}")
    log_info(f"Rooms searched: {rooms_searched}")
    
    if target_found:
        log_info(f"✓ TASK SUCCESS: Target object found in {found_location}")
        
        # 生成任务总结和行动计划
        log_info("\n" + "=" * 40)
        log_info("GENERATING TASK SUMMARY AND ACTION PLAN")
        log_info("=" * 40)
        
        # 检索记忆证据
        target_object = intent.get("object", "")
        memory_evidences = retrieve_candidates_by_intent(intent, topk=5)
        
        # 构建当前状态 - 明确标注视觉检测已成功找到目标
        current_status = {
            "status": "SUCCESS",
            "target_object": target_object,
            "target_found": True,
            "found_location": found_location,
            "current_location": navigator.current_location,
            "search_path": navigator.navigation_history + [navigator.current_location],
            "total_rooms_searched": len(rooms_searched),
            "rooms_searched": rooms_searched,
            "visual_detection_result": "SUCCESS - Target object visually confirmed in current location",
            "detection_confidence": "HIGH - Visual detection successfully identified the target object"
        }
        
        # 构建视觉证据 - 明确标注已成功检测到
        vision_evidence = structure_vision_evidence(
            {
                "detected": True,
                "detection_status": "SUCCESS",
                "location": found_location,
                "object": target_object,
                "confidence": "high",
                "message": "Visual detection successfully confirmed the presence of target object"
            },
            target_object
        )
        
        # 融合视觉和记忆证据
        if memory_evidences:
            fused_evidences = fuse_vision_with_memory(memory_evidences, vision_evidence)
            
            # 生成行动计划
            plan_result = generate_action_plan_llm(intent, fused_evidences, current_status)
            
            if plan_result.is_success():
                log_info("Action plan generated successfully", plan_result.data)
                
                # 输出最终结果总结
                log_info("\n" + "=" * 60)
                log_info("FINAL TASK RESULT")
                log_info("=" * 60)
                log_info(f"目标对象: {target_object}")
                log_info(f"找到位置: {found_location}")
                log_info(f"搜索路径: {' -> '.join(navigator.navigation_history + [navigator.current_location])}")
                log_info(f"共搜索房间数: {len(rooms_searched)}")
                log_info(f"搜索房间列表: {', '.join(rooms_searched)}")
                
                # 输出行动计划
                if plan_result.data:
                    log_info("\n--- ACTION PLAN ---")
                    log_info(f"Next Action: {plan_result.data.get('next_action', 'N/A')}")
                    log_info(f"Confidence: {plan_result.data.get('confidence', 'N/A')}")
                    log_info(f"Reasoning: {plan_result.data.get('reasoning', 'N/A')}")
                
                log_info("=" * 60)
            else:
                log_warning("Failed to generate action plan", {"error": plan_result.error_msg})
        else:
            log_warning("No memory evidences found for action planning")
            
    else:
        log_info("✗ TASK FAILED: Target object not found in any searched room")
    
    log_info("Task simulation completed")

def perform_vision_detection_in_room(navigator, intent, detection_type):
    """在指定房间进行视觉检测，返回是否找到目标对象"""
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
        vision_result = call_chatglm_multi_object_detection(f"data:image/png;base64,{base64_image}", target_objects)
        
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
            target_found = analyze_detection_results(vision_result.data, intent, navigator.current_location, detection_type)
            return target_found
            
        else:
            log_warning(f"Vision detection failed in {navigator.current_location}: {vision_result.error_msg}")
            return False
            
    except Exception as e:
        log_error(f"Error during vision detection in {navigator.current_location}", e)
        return False

def analyze_detection_results(vision_data, intent, location, detection_type):
    """分析视觉检测结果"""
    objects = vision_data.get("objects", [])
    target_object = intent.get("object", "")
    
    log_info(f"Analyzing detection results in {location}", {
        "detection_type": detection_type,
        "total_objects": len(objects),
        "target_object": target_object
    })
    
    # 定义关键词映射（中文到英文）
    keyword_mapping = {
        "水杯": "watercup",
        "杯子": "cup",
        "手机": "phone",
        "书": "book",
        "电脑": "laptop",
        "笔记本": "laptop",
        "钢琴": "piano",
        "三角钢琴": "grand piano",
        "立式钢琴": "upright piano"
    }
    
    # 检查是否找到目标对象
    target_found = False
    matching_obj = None
    
    for obj in objects:
        obj_label = obj.get("label", "").lower()
        obj_desc = obj.get("description", "").lower()
        
        # 检查直接匹配
        if target_object and target_object.lower() in obj_label:
            target_found = True
            matching_obj = obj
            log_info(f"TARGET FOUND (direct match): {obj['label']} with confidence {obj['confidence']}", obj)
            break
        
        # 检查关键词映射匹配
        if target_object:
            target_lower = target_object.lower()
            # 检查中文关键词
            for chinese_keyword, english_keyword in keyword_mapping.items():
                if chinese_keyword in target_lower:
                    # 在label或description中查找匹配
                    if english_keyword in obj_label or english_keyword in obj_desc:
                        target_found = True
                        matching_obj = obj
                        log_info(f"TARGET FOUND (keyword mapping in label/desc): {obj['label']} with confidence {obj['confidence']}", obj)
                        break
                    # 也检查中文关键词
                    if chinese_keyword in obj_label or chinese_keyword in obj_desc:
                        target_found = True
                        matching_obj = obj
                        log_info(f"TARGET FOUND (keyword mapping in label/desc): {obj['label']} with confidence {obj['confidence']}", obj)
                        break
                # 检查英文关键词
                if english_keyword in target_lower and (english_keyword in obj_label or english_keyword in obj_desc):
                    target_found = True
                    matching_obj = obj
                    log_info(f"TARGET FOUND (keyword mapping): {obj['label']} with confidence {obj['confidence']}", obj)
                    break
            
            if target_found:
                break
    
    # Fallback: 如果在objects中没找到，检查description中是否提到了目标对象
    if not target_found and detection_type == "target_detection":
        log_warning(f"Target object '{target_object}' not found in objects list")
        
        # 在所有对象的description中搜索目标对象
        for obj in objects:
            obj_desc = obj.get("description", "").lower()
            obj_label = obj.get("label", "").lower()
            
            # 检查description中是否包含目标对象
            if target_object:
                target_lower = target_object.lower()
                
                # 直接检查目标对象是否在description中
                if target_lower in obj_desc:
                    target_found = True
                    matching_obj = obj
                    log_info(f"TARGET FOUND (in description): {obj['label']} - description mentions '{target_object}'", {
                        "label": obj['label'],
                        "description": obj.get('description', ''),
                        "confidence": obj.get('confidence', 0)
                    })
                    break
                
                # 检查关键词映射
                for chinese_keyword, english_keyword in keyword_mapping.items():
                    if chinese_keyword in target_lower:
                        if english_keyword in obj_desc or chinese_keyword in obj_desc:
                            target_found = True
                            matching_obj = obj
                            log_info(f"TARGET FOUND (keyword in description): {obj['label']} - {chinese_keyword} mentioned in description", {
                                "label": obj['label'],
                                "description": obj.get('description', ''),
                                "confidence": obj.get('confidence', 0)
                            })
                            break
                
                if target_found:
                    break
        
        if not target_found:
            log_warning(f"Target object '{target_object}' critically not found in {location}")
            
        # 显示所有检测到的对象
        log_info("All detected objects:", [{"label": obj['label'], "confidence": obj.get('confidence', 0)} for obj in objects])
    
    return target_found

# ----------------- Main flow -----------------
if __name__ == "__main__":
    simulate_robot_navigation_task()
