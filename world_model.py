#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WorldModel: 世界状态模型，管理实体、关系和事件
支持动态场景：missing detection, movement tracking, removal confirmation
使用 SQLite 存储结构化世界状态
"""
import sqlite3
import json
import time
import uuid
import os
import math
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

DB_PATH_DEFAULT = "world_model.sqlite"


class WorldModel:
    """世界状态模型，管理实体、关系和事件"""
    
    def __init__(self, db_path: str = DB_PATH_DEFAULT, logger=None, llm_client=None):
        """
        初始化 WorldModel
        
        Args:
            db_path: SQLite 数据库路径
            logger: 日志记录器（可选）
            llm_client: LLM 客户端（可选），用于高级功能
        """
        self.db_path = db_path
        self.logger = logger
        self.llm_client = llm_client
        
        # 确保目录存在
        db_dir = os.path.dirname(self.db_path)
        if db_dir:  # 如果有目录部分
            os.makedirs(db_dir, exist_ok=True)
        
        # 初始化数据库
        self._init_db()
        
        # in-memory cache for speed (simple)
        self.entities = {}  # entity_id -> dict (load lazily)
        self._load_entities_into_cache(limit=100)
    
    def _init_db(self):
        """初始化 SQLite 数据库表"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # 创建 entities 表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                data TEXT
            )
        """)
        
        # 创建 events 表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                data TEXT
            )
        """)
        
        # 为了兼容性，保留原有的表结构（如果存在）
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_entity_id TEXT NOT NULL,
                    to_entity_id TEXT NOT NULL,
                    rel_type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    updated_ts REAL NOT NULL
                )
            """)
        except:
            pass
        
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS provenance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    observation_json TEXT,
                    ts REAL NOT NULL
                )
            """)
        except:
            pass
        
        conn.commit()
        conn.close()
    
    def _save_entity(self, entity: Dict):
        """保存实体到数据库和缓存"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("REPLACE INTO entities(entity_id, data) VALUES (?, ?)", 
                   (entity["entity_id"], json.dumps(entity, ensure_ascii=False)))
        conn.commit()
        conn.close()
        self.entities[entity["entity_id"]] = entity
    
    def _load_entities_into_cache(self, limit=100):
        """加载实体到内存缓存"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT entity_id, data FROM entities LIMIT ?", (limit,))
        rows = cur.fetchall()
        for eid, data in rows:
            try:
                self.entities[eid] = json.loads(data)
            except:
                continue
        conn.close()
    
    def create_entity(self, label: str, position: List[float], confidence: float, ts: Optional[float] = None) -> Dict:
        """
        创建新实体
        
        Args:
            label: 实体标签
            position: 位置 [x, y, z]
            confidence: 置信度
            ts: 时间戳（可选）
        
        Returns:
            创建的实体字典
        """
        ts = ts or time.time()
        eid = str(uuid.uuid4())
        ent = {
            "entity_id": eid,
            "label": label,
            "position": position,
            "pos_history": [{"ts": ts, "pos": position, "conf": confidence}],
            "status": "active",
            "last_seen_ts": ts,
            "missing_since": None,
            "missing_evidence": [],
            "movement_events": [],
            "remove_event": None,
            "decay_score": confidence,
            "provenance": [],
            "fused_node_id": None
        }
        self._save_entity(ent)
        return ent
    
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """获取实体（从缓存或数据库）"""
        if entity_id in self.entities:
            return self.entities[entity_id]
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT data FROM entities WHERE entity_id=?", (entity_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            ent = json.loads(row[0])
            self.entities[entity_id] = ent
            return ent
        return None
    
    def save_event(self, event: Dict):
        """保存事件到数据库"""
        event_id = event.get("event_id", str(uuid.uuid4()))
        event["event_id"] = event_id
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("REPLACE INTO events(event_id, data) VALUES (?, ?)", 
                   (event_id, json.dumps(event, ensure_ascii=False)))
        conn.commit()
        conn.close()
    
    def ingest_observation(self, obs: Dict) -> Dict:
        """
        处理观察数据，更新或创建实体
        
        Args:
            obs: 观察字典，包含：
                - label: 对象标签
                - fused_3d_pos: 融合后的3D位置 [x, y, z]
                - confidence: 置信度
                - ts: 时间戳（可选）
                - memnode_id: MemoryGraph 节点 ID（可选）
                - bbox: 边界框（可选）
                - image_path: 图像路径（可选）
        
        Returns:
            结果字典，包含：
                - entity_id: 实体 ID
                - action: "created" | "updated" | "relinked"
        """
        label = obs.get("label", "unknown")
        fused_3d_pos = obs.get("fused_3d_pos")
        confidence = obs.get("confidence", 0.5)
        ts = obs.get("ts", time.time())
        memnode_id = obs.get("memnode_id")
        
        if not fused_3d_pos:
            if self.logger:
                self.logger("ingest_observation", {"error": "missing position", "label": label})
            return {"entity_id": None, "action": "skipped"}
        
        # 尝试重新链接到现有实体
        linked_entity_id = self.relink_candidate(obs, spatial_th=0.3)
        
        if linked_entity_id:
            # 更新现有实体
            ent = self.get_entity(linked_entity_id)
            if ent:
                # 更新位置历史
                ent.setdefault("pos_history", []).append({
                    "ts": ts,
                    "pos": fused_3d_pos,
                    "conf": confidence
                })
                # 更新位置（加权平均）
                old_pos = ent.get("position")
                if old_pos:
                    # 简单更新：使用新位置（可以改为加权平均）
                    ent["position"] = fused_3d_pos
                else:
                    ent["position"] = fused_3d_pos
                
                # 更新 last_seen_ts
                ent["last_seen_ts"] = ts
                
                # 更新 fused_node_id（如果提供）
                if memnode_id:
                    ent["fused_node_id"] = memnode_id
                
                # 更新状态为 active（如果之前是 missing）
                if ent.get("status") in ["missing", "moved"]:
                    ent["status"] = "active"
                
                self._save_entity(ent)
                
                if self.logger:
                    self.logger("ingest_observation", {
                        "entity_id": linked_entity_id,
                        "action": "updated",
                        "label": label
                    })
                
                return {"entity_id": linked_entity_id, "action": "updated"}
        
        # 创建新实体
        new_entity = self.create_entity(
            label=label,
            position=fused_3d_pos,
            confidence=confidence,
            ts=ts
        )
        entity_id = new_entity["entity_id"]
        
        # 关联 MemoryGraph 节点 ID
        if memnode_id:
            new_entity["fused_node_id"] = memnode_id
            self._save_entity(new_entity)
        
        if self.logger:
            self.logger("ingest_observation", {
                "entity_id": entity_id,
                "action": "created",
                "label": label
            })
        
        return {"entity_id": entity_id, "action": "created"}
    
    # --- dynamic scene methods ---
    
    def mark_no_detection(self, entity_id: str, evidence: Dict):
        """
        记录一次 no-detect 证据
        
        Args:
            entity_id: 实体 ID
            evidence: 证据字典，example:
                {
                  "ts": 169..., "agent_pose": [...], "vis_score": 0.8,
                  "frame_id": "...", "depth_stats": {...}
                }
        
        Returns:
            是否成功
        """
        ent = self.get_entity(entity_id)
        if not ent:
            return False
        ent.setdefault("missing_evidence", []).append(evidence)
        self._save_entity(ent)
        if self.logger:
            self.logger("mark_no_detection", {"entity_id": entity_id, "evidence": evidence})
        return True
    
    def evaluate_missing(self, entity_id: str, n_no_detects: int = 3, vis_th: float = 0.6):
        """
        累积证据后判断是否进入 missing 状态
        
        Args:
            entity_id: 实体 ID
            n_no_detects: 需要多少次未检测到
            vis_th: 视觉分数阈值
        
        Returns:
            是否进入 missing 状态
        """
        ent = self.get_entity(entity_id)
        if not ent:
            return False
        evs = ent.get("missing_evidence", [])[-10:]  # 最近最多10条
        if len(evs) < n_no_detects:
            return False
        # 考虑最近 n_no_detects 的 vis_score 平均
        recent = evs[-n_no_detects:]
        vis_avg = sum(e.get("vis_score", 0.0) for e in recent) / n_no_detects
        if vis_avg >= vis_th:
            # set missing
            ent["status"] = "missing"
            ent["missing_since"] = recent[0].get("ts", time.time())
            self._save_entity(ent)
            self.save_event({
                "type": "disappearance_candidate",
                "entities": [entity_id],
                "ts": time.time(),
                "confidence": vis_avg,
                "evidence": recent
            })
            return True
        return False
    
    def confirm_removed(self, entity_id: str, removed_T: float = 30*60):
        """
        确认实体已被移除（基于时间阈值）
        
        Args:
            entity_id: 实体 ID
            removed_T: 移除时间阈值（秒），默认30分钟
        
        Returns:
            是否确认移除
        """
        ent = self.get_entity(entity_id)
        if not ent or ent.get("status") != "missing":
            return False
        if ent.get("missing_since") is None:
            return False
        if time.time() - ent["missing_since"] > removed_T:
            ent["status"] = "removed"
            ev = {
                "type": "disappearance",
                "entities": [entity_id],
                "ts": time.time(),
                "method": "time_threshold",
                "confidence": 0.9,
                "from": ent.get("position"),
                "to": None
            }
            self.save_event(ev)
            ent["remove_event"] = ev
            self._save_entity(ent)
            return True
        return False
    
    def register_movement(self, entity_id: str, new_pos: List[float], method: str = "triangulate", confidence: float = 0.8):
        """
        注册实体移动
        
        Args:
            entity_id: 实体 ID
            new_pos: 新位置 [x, y, z]
            method: 定位方法
            confidence: 置信度
        
        Returns:
            移动事件字典
        """
        ent = self.get_entity(entity_id)
        if not ent:
            return None
        prev_pos = ent.get("position")
        # 计算距离（兼容 Python 3.6，math.dist 在 3.8+ 才有）
        if prev_pos:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(prev_pos, new_pos)))
        else:
            dist = None
        event = {
            "type": "movement",
            "entities": [entity_id],
            "from": prev_pos,
            "to": new_pos,
            "ts": time.time(),
            "method": method,
            "confidence": confidence,
            "distance": dist
        }
        self.save_event(event)
        ent.setdefault("movement_events", []).append(event)
        ent["position"] = new_pos
        ent.setdefault("pos_history", []).append({"ts": time.time(), "pos": new_pos, "conf": confidence})
        ent["status"] = "moved"
        ent["last_seen_ts"] = time.time()
        self._save_entity(ent)
        return event
    
    def relink_candidate(self, obs: Dict, spatial_th: float = 0.3):
        """
        尝试将观察重新链接到现有实体（基于空间接近度）
        
        Args:
            obs: 观察字典，包含 label, fused_3d_pos (optional), embedding (optional)
            spatial_th: 空间距离阈值（米）
        
        Returns:
            匹配的 entity_id 或 None
        """
        if obs.get("fused_3d_pos") is not None:
            p = obs["fused_3d_pos"]
            for eid, ent in list(self.entities.items()):
                ent_pos = ent.get("position")
                if ent_pos:
                    # 计算距离（兼容 Python 3.6）
                    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(ent_pos, p)))
                    if dist < spatial_th:
                        return eid
        # fallback: None (embedding-based match can be added later)
        return None
    
    def query_entity_state(self, entity_id: str):
        """查询实体状态"""
        return self.get_entity(entity_id)
    
    def export_snapshot(self, out_path: str):
        """
        导出快照到 JSON 文件
        
        Args:
            out_path: 输出文件路径
        """
        # dump all entities + events to json
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT data FROM entities")
        ents = [json.loads(r[0]) for r in cur.fetchall()]
        cur.execute("SELECT data FROM events")
        evs = [json.loads(r[0]) for r in cur.fetchall()]
        conn.close()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"entities": ents, "events": evs}, f, ensure_ascii=False, indent=2)
