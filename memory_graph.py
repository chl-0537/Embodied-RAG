"""
MemoryGraph: 轻量级记忆图模块，使用 SQLite 作为后端
实现证据合并、衰减和基于图的检索重排序
"""
import sqlite3
import json
import time
import uuid
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import numpy as np


class MemoryGraph:
    """轻量级记忆图，使用 SQLite 存储节点、边和来源信息"""
    
    def __init__(self, db_path: str, chroma_client=None, decay_defaults: Optional[Dict] = None):
        """
        初始化 MemoryGraph
        
        Args:
            db_path: SQLite 数据库路径
            chroma_client: ChromaDB 客户端（可选）
            decay_defaults: 默认衰减参数 {"halflife": 86400} (24小时)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chroma_client
        # 默认衰减参数：支持 halflife_hours（小时）或 halflife（秒）
        if decay_defaults is None:
            self.decay_defaults = {"halflife_hours": 24}  # 默认24小时半衰期
        else:
            self.decay_defaults = decay_defaults.copy()
            # 兼容旧的 halflife 格式（秒）
            if "halflife" in self.decay_defaults and "halflife_hours" not in self.decay_defaults:
                self.decay_defaults["halflife_hours"] = self.decay_defaults["halflife"] / 3600.0
        
        # 初始化数据库表
        self._init_db()
    
    def _init_db(self):
        """初始化 SQLite 数据库表"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 创建 nodes 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                data_json TEXT NOT NULL,
                fused_conf REAL DEFAULT 0.5,
                updated_ts REAL NOT NULL
            )
        """)
        
        # 创建 edges 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                rel_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                updated_ts REAL NOT NULL,
                FOREIGN KEY (from_id) REFERENCES nodes(id),
                FOREIGN KEY (to_id) REFERENCES nodes(id)
            )
        """)
        
        # 创建 provenance 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS provenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                ts REAL NOT NULL,
                meta_json TEXT,
                FOREIGN KEY (node_id) REFERENCES nodes(id)
            )
        """)
        
        # 创建索引以提高查询性能
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_updated_ts ON nodes(updated_ts)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_from_id ON edges(from_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_to_id ON edges(to_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_provenance_node_id ON provenance(node_id)")
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(str(self.db_path))
    
    def _apply_decay_to_node(self, node_data: Dict, node_id: str, now_ts: Optional[float] = None, 
                             current_fused_conf: Optional[float] = None, 
                             last_update_ts: Optional[float] = None) -> Tuple[float, float]:
        """
        对节点应用衰减并返回新的置信度和更新时间戳
        
        Args:
            node_data: 节点数据字典
            node_id: 节点 ID
            now_ts: 当前时间戳（如果为 None，使用当前时间）
            current_fused_conf: 当前融合置信度（如果为 None，从 node_data 获取）
            last_update_ts: 最后更新时间戳（如果为 None，从 node_data 获取）
        
        Returns:
            (new_fused_conf, new_updated_ts) 元组
        """
        if now_ts is None:
            now_ts = time.time()
        
        # 获取当前置信度和更新时间
        if current_fused_conf is None:
            # 尝试从数据库获取
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT fused_conf, updated_ts FROM nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                current_fused_conf, last_update_ts = row
            else:
                current_fused_conf = node_data.get("confidence", 0.5)
                last_update_ts = node_data.get("updated_ts", now_ts)
        
        if last_update_ts is None:
            last_update_ts = node_data.get("updated_ts", now_ts)
        
        # 获取衰减参数
        decay_params = node_data.get("decay_params", {})
        halflife_hours = decay_params.get("halflife_hours")
        
        if halflife_hours is None:
            # 使用默认值（24小时）
            halflife_hours = self.decay_defaults.get("halflife_hours", 24)
            if halflife_hours is None:
                # 兼容旧的 halflife 格式（秒）
                halflife_seconds = self.decay_defaults.get("halflife", 86400)
                halflife_hours = halflife_seconds / 3600.0
        
        halflife_seconds = halflife_hours * 3600.0
        
        # 计算时间差
        dt = now_ts - last_update_ts
        
        # 应用衰减公式: C_new = C_old * 0.5 ** (dt / halflife_seconds)
        if halflife_seconds > 0 and dt >= 0:
            if dt > 0:
                decay_factor = 0.5 ** (dt / halflife_seconds)
                new_fused_conf = current_fused_conf * decay_factor
            else:
                # dt == 0，没有时间差，不衰减
                new_fused_conf = current_fused_conf
        else:
            new_fused_conf = current_fused_conf
        
        return new_fused_conf, now_ts
    
    def _update_time_histogram(self, node_data: Dict, ts: float):
        """
        更新节点的时间直方图
        
        Args:
            node_data: 节点数据字典（会被修改）
            ts: 时间戳
        """
        # 初始化 time_histogram（如果不存在）
        if "time_histogram" not in node_data:
            node_data["time_histogram"] = [0] * 24  # 24小时直方图
        
        # 初始化 weekday_hour_histogram（如果不存在）
        if "weekday_hour_histogram" not in node_data:
            node_data["weekday_hour_histogram"] = [[0] * 24 for _ in range(7)]  # 7天 * 24小时
        
        # 将时间戳转换为 datetime
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour  # 0-23
        weekday = dt.weekday()  # 0=Monday, 6=Sunday
        
        # 更新 24小时直方图
        node_data["time_histogram"][hour] += 1
        
        # 更新 weekday-hour 直方图
        node_data["weekday_hour_histogram"][weekday][hour] += 1
    
    def _parse_time_hint(self, time_hint: str) -> List[int]:
        """
        解析时间提示字符串，返回小时列表
        
        Args:
            time_hint: 时间提示，例如:
                - "16:00" -> [16]
                - "15:00-17:00" -> [15, 16, 17]
                - "下午" -> [12, 13, 14, 15, 16, 17]
                - "16" -> [16]
        
        Returns:
            小时列表（0-23）
        """
        hours = []
        
        # 处理时间段范围
        if "-" in time_hint:
            parts = time_hint.split("-")
            if len(parts) == 2:
                try:
                    start_hour = int(parts[0].split(":")[0])
                    end_hour = int(parts[1].split(":")[0])
                    hours = list(range(start_hour, end_hour + 1))
                except:
                    pass
        
        # 处理单个时间点
        elif ":" in time_hint:
            try:
                hour = int(time_hint.split(":")[0])
                hours = [hour]
            except:
                pass
        
        # 处理时间段描述
        elif "上午" in time_hint or "morning" in time_hint.lower():
            hours = list(range(6, 12))
        elif "下午" in time_hint or "afternoon" in time_hint.lower():
            hours = list(range(12, 18))
        elif "晚上" in time_hint or "evening" in time_hint.lower():
            hours = list(range(18, 22))
        elif "夜晚" in time_hint or "night" in time_hint.lower():
            hours = list(range(22, 24)) + list(range(0, 6))
        elif "黎明" in time_hint or "dawn" in time_hint.lower():
            hours = list(range(0, 6))
        
        # 处理纯数字（小时）
        else:
            try:
                hour = int(time_hint)
                if 0 <= hour <= 23:
                    hours = [hour]
            except:
                pass
        
        return hours
    
    def _compute_time_score(self, node_data: Dict, time_hint: str) -> float:
        """
        计算时间分数
        
        Args:
            node_data: 节点数据
            time_hint: 时间提示字符串
        
        Returns:
            时间分数（0-1）
        """
        time_histogram = node_data.get("time_histogram", [0] * 24)
        
        # 解析时间提示
        hint_hours = self._parse_time_hint(time_hint)
        if not hint_hours:
            return 0.0
        
        # 计算总计数
        total_count = sum(time_histogram)
        if total_count == 0:
            return 0.0
        
        # 计算提示时间段内的计数
        hint_count = sum(time_histogram[h] for h in hint_hours if 0 <= h < 24)
        
        # 归一化分数：hint_count / total_count
        time_score = hint_count / total_count if total_count > 0 else 0.0
        
        return time_score
    
    def find_merge_candidates(self, evidence: Dict) -> List[Tuple[str, float]]:
        """
        查找可以合并的候选节点
        
        Args:
            evidence: 证据字典，必须包含 type, label, confidence
        
        Returns:
            List of (node_id, similarity_score) tuples
        """
        candidates = []
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 获取所有现有节点
        cursor.execute("SELECT id, data_json FROM nodes")
        rows = cursor.fetchall()
        
        evidence_embedding_id = evidence.get("embedding_id")
        evidence_embedding = evidence.get("embedding")
        evidence_position = evidence.get("position")
        
        for node_id, data_json_str in rows:
            try:
                node_data = json.loads(data_json_str)
            except:
                continue
            
            similarity = 0.0
            merge_reason = None
            
            # 1. 检查 embedding_id 是否相同
            if evidence_embedding_id and node_data.get("embedding_id"):
                if evidence_embedding_id == node_data.get("embedding_id"):
                    similarity = 1.0
                    merge_reason = "embedding_id_match"
            
            # 2. 检查 embedding cosine 相似度 > 0.85
            if similarity < 0.85 and evidence_embedding is not None:
                node_embedding = node_data.get("embedding")
                if node_embedding is not None:
                    try:
                        # 转换为 numpy 数组并计算余弦相似度
                        ev_emb = np.array(evidence_embedding, dtype=np.float32)
                        node_emb = np.array(node_embedding, dtype=np.float32)
                        
                        # 归一化
                        ev_emb = ev_emb / (np.linalg.norm(ev_emb) + 1e-8)
                        node_emb = node_emb / (np.linalg.norm(node_emb) + 1e-8)
                        
                        cos_sim = np.dot(ev_emb, node_emb)
                        if cos_sim > 0.85:
                            similarity = max(similarity, cos_sim)
                            merge_reason = "embedding_similarity"
                    except:
                        pass
            
            # 3. 检查空间距离 < 0.2
            if similarity < 0.85 and evidence_position is not None:
                node_position = node_data.get("position")
                if node_position is not None:
                    try:
                        ev_pos = np.array(evidence_position)
                        node_pos = np.array(node_position)
                        spatial_dist = np.linalg.norm(ev_pos - node_pos)
                        if spatial_dist < 0.2:
                            # 空间距离 < 0.2 时，直接设置相似度为 0.85（满足合并条件）
                            similarity = max(similarity, 0.85)
                            merge_reason = "spatial_proximity"
                    except:
                        pass
            
            if similarity >= 0.85:
                candidates.append((node_id, similarity))
        
        conn.close()
        
        # 按相似度排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def get_node(self, node_id: str, apply_decay: bool = True) -> Optional[Dict]:
        """
        获取节点数据，可选择性地应用衰减
        
        Args:
            node_id: 节点 ID
            apply_decay: 是否应用衰减
        
        Returns:
            节点数据字典，如果节点不存在则返回 None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT data_json, fused_conf, updated_ts FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        data_json_str, fused_conf, updated_ts = row
        node_data = json.loads(data_json_str)
        node_data["fused_conf"] = fused_conf
        node_data["updated_ts"] = updated_ts
        
        # 应用衰减
        if apply_decay:
            new_conf, new_ts = self._apply_decay_to_node(
                node_data, node_id, 
                current_fused_conf=fused_conf,
                last_update_ts=updated_ts
            )
            
            # 如果置信度发生变化，更新数据库
            if abs(new_conf - fused_conf) > 1e-6:
                node_data["fused_conf"] = new_conf
                node_data["updated_ts"] = new_ts
                cursor.execute("""
                    UPDATE nodes 
                    SET fused_conf = ?, updated_ts = ?
                    WHERE id = ?
                """, (new_conf, new_ts, node_id))
                conn.commit()
        
        conn.close()
        return node_data
    
    def compute_evidence_score(self, evidence: Dict, node: Dict, lambda_spatial: float = 1.0) -> float:
        """
        计算证据评分
        
        Args:
            evidence: 证据字典
            node: 节点数据字典
            lambda_spatial: 空间一致性衰减参数（默认1.0米）
        
        Returns:
            证据评分 (0.0 - 1.0)
        """
        # a) visual_confidence
        v = evidence.get("confidence", 0.0)
        
        # b) embedding similarity
        s = 0.0
        evidence_embedding_id = evidence.get("embedding_id")
        node_embedding_id = node.get("embedding_id")
        
        if evidence_embedding_id and node_embedding_id:
            if evidence_embedding_id == node_embedding_id:
                s = 1.0
            elif self.chroma_client:
                try:
                    # 尝试从 Chroma 获取相似度
                    collection = self.chroma_client.get_or_create_collection(name="robot_memory")
                    # 获取两个 embedding 的向量
                    evidence_emb = evidence.get("embedding")
                    node_emb = node.get("embedding")
                    
                    if evidence_emb and node_emb:
                        # 计算余弦相似度
                        ev_emb = np.array(evidence_emb, dtype=np.float32)
                        node_emb_arr = np.array(node_emb, dtype=np.float32)
                        
                        # 归一化
                        ev_emb = ev_emb / (np.linalg.norm(ev_emb) + 1e-8)
                        node_emb_arr = node_emb_arr / (np.linalg.norm(node_emb_arr) + 1e-8)
                        
                        s = float(np.dot(ev_emb, node_emb_arr))
                except Exception as e:
                    # 如果获取失败，使用默认值 0
                    pass
        
        # c) spatial consistency
        d = 0.0
        evidence_position = evidence.get("position")
        node_position = node.get("position")
        
        if evidence_position and node_position:
            try:
                ev_pos = np.array(evidence_position)
                node_pos = np.array(node_position)
                distance = np.linalg.norm(ev_pos - node_pos)
                # d = exp(-distance / lambda_spatial)
                d = math.exp(-distance / lambda_spatial)
            except:
                pass
        
        # 综合评分: score = 0.6*v + 0.2*s + 0.2*d
        score = 0.6 * v + 0.2 * s + 0.2 * d
        
        return max(0.0, min(1.0, score))  # 限制在 [0, 1] 范围内
    
    def merge_candidate_update(self, node_id: str, evidence: Dict, alpha: float = 0.7, 
                              lambda_spatial: float = 1.0) -> str:
        """
        将新证据合并到现有节点
        
        Args:
            node_id: 目标节点 ID
            evidence: 新证据
            alpha: 融合权重（默认0.7，表示旧置信度权重）
            lambda_spatial: 空间一致性衰减参数
        
        Returns:
            node_id
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 获取现有节点数据（应用衰减）
        cursor.execute("SELECT data_json, fused_conf, updated_ts FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return self.add_evidence(evidence)  # 如果节点不存在，创建新节点
        
        old_data_json, old_fused_conf, old_updated_ts = row
        old_data = json.loads(old_data_json)
        
        # 应用衰减到当前时间
        now_ts = time.time()
        new_fused_conf, _ = self._apply_decay_to_node(
            old_data, node_id,
            now_ts=now_ts,
            current_fused_conf=old_fused_conf,
            last_update_ts=old_updated_ts
        )
        C_old = new_fused_conf
        
        # 计算证据评分
        E = self.compute_evidence_score(evidence, old_data, lambda_spatial=lambda_spatial)
        
        # 合并置信度: C_new = clamp(alpha * C_old + (1-alpha) * E, 0, 1)
        C_new = max(0.0, min(1.0, alpha * C_old + (1.0 - alpha) * E))
        
        # 更新节点数据
        merged_data = old_data.copy()
        
        # 更新 last_seen 和 detections_count
        merged_data["last_seen"] = now_ts
        merged_data["detections_count"] = merged_data.get("detections_count", 0) + 1
        
        # 更新时间直方图
        evidence_ts = evidence.get("ts", now_ts)
        self._update_time_histogram(merged_data, evidence_ts)
        
        # 更新 position: weighted average by confidences
        evidence_position = evidence.get("position")
        old_position = old_data.get("position")
        old_pos_for_prov = None
        
        if evidence_position and old_position:
            try:
                # 保存旧位置到 provenance
                if isinstance(old_position, list):
                    old_pos_for_prov = old_position.copy()
                else:
                    old_pos_for_prov = old_position
                
                # 加权平均: new_pos = (C_old * old_pos + E * new_pos) / (C_old + E)
                ev_pos = np.array(evidence_position)
                old_pos = np.array(old_position)
                
                total_weight = C_old + E
                if total_weight > 0:
                    new_position = (C_old * old_pos + E * ev_pos) / total_weight
                    merged_data["position"] = new_position.tolist()
                else:
                    merged_data["position"] = evidence_position
            except:
                merged_data["position"] = evidence_position
        elif evidence_position:
            merged_data["position"] = evidence_position
        
        # 更新 embedding（如果新证据有 embedding）
        if evidence.get("embedding"):
            merged_data["embedding"] = evidence.get("embedding")
            merged_data["embedding_id"] = evidence.get("embedding_id")
        
        # 更新其他字段
        merged_data["updated_ts"] = now_ts
        if evidence.get("label"):
            merged_data["label"] = evidence.get("label")
        if evidence.get("type"):
            merged_data["type"] = evidence.get("type")
        
        # 更新节点
        cursor.execute("""
            UPDATE nodes 
            SET data_json = ?, fused_conf = ?, updated_ts = ?
            WHERE id = ?
        """, (json.dumps(merged_data, ensure_ascii=False), C_new, now_ts, node_id))
        
        # 添加来源记录（包含位置信息）
        provenance_meta = evidence.get("metadata", {}).copy()
        if old_position and evidence_position:
            provenance_meta["previous_position"] = old_pos_for_prov if old_pos_for_prov is not None else old_position
            provenance_meta["new_position"] = evidence_position
        
        evidence_with_meta = evidence.copy()
        evidence_with_meta["metadata"] = provenance_meta
        self._add_provenance(node_id, evidence_with_meta, cursor)
        
        conn.commit()
        conn.close()
        
        return node_id
    
    def _add_provenance(self, node_id: str, evidence: Dict, cursor=None):
        """添加来源记录"""
        if cursor is None:
            conn = self._get_connection()
            cursor = conn.cursor()
            should_close = True
        else:
            should_close = False
        
        cursor.execute("""
            INSERT INTO provenance (node_id, source, confidence, ts, meta_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            node_id,
            evidence.get("type", "unknown"),
            evidence.get("confidence", 0.5),
            evidence.get("ts", time.time()),
            json.dumps(evidence.get("metadata", {}), ensure_ascii=False)
        ))
        
        if should_close:
            conn.commit()
            conn.close()
    
    def _add_edge(self, from_id: str, to_id: str, rel_type: str, weight: float = 1.0, cursor=None):
        """
        添加边到图中
        
        Args:
            from_id: 源节点 ID
            to_id: 目标节点 ID
            rel_type: 关系类型
            weight: 边权重
            cursor: 数据库游标（如果为 None，会创建新连接）
        """
        if cursor is None:
            conn = self._get_connection()
            cursor = conn.cursor()
            should_close = True
        else:
            should_close = False
        
        # 检查边是否已存在
        cursor.execute("""
            SELECT id FROM edges 
            WHERE from_id = ? AND to_id = ? AND rel_type = ?
        """, (from_id, to_id, rel_type))
        existing = cursor.fetchone()
        
        if existing:
            # 更新现有边
            cursor.execute("""
                UPDATE edges 
                SET weight = ?, updated_ts = ?
                WHERE from_id = ? AND to_id = ? AND rel_type = ?
            """, (weight, time.time(), from_id, to_id, rel_type))
        else:
            # 创建新边
            cursor.execute("""
                INSERT INTO edges (from_id, to_id, rel_type, weight, updated_ts)
                VALUES (?, ?, ?, ?, ?)
            """, (from_id, to_id, rel_type, weight, time.time()))
        
        if should_close:
            conn.commit()
            conn.close()
    
    def add_evidence(self, evidence: Dict) -> str:
        """
        添加新证据到记忆图
        
        Args:
            evidence: 证据字典，必须包含:
                - type: 证据类型
                - label: 标签
                - confidence: 置信度
                - embedding_id (optional): Chroma embedding ID
                - embedding (optional): 向量嵌入
                - position (optional): 位置 [x, y, z]
                - ts (optional): 时间戳
        
        Returns:
            node_id: 节点 ID
        """
        # 验证必需字段
        required_fields = ["type", "label", "confidence"]
        for field in required_fields:
            if field not in evidence:
                raise ValueError(f"Evidence must contain '{field}' field")
        
        # 查找合并候选
        candidates = self.find_merge_candidates(evidence)
        
        if candidates:
            # 获取最佳候选节点
            best_candidate_id, similarity = candidates[0]
            
            # 获取节点数据以检查冲突
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT data_json FROM nodes WHERE id = ?", (best_candidate_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                node_data = json.loads(row[0])
                evidence_label = evidence.get("label", "")
                node_label = node_data.get("label", "")
                
                # 检查冲突：label 相同但位置距离 > 2.0m
                evidence_position = evidence.get("position")
                node_position = node_data.get("position")
                
                if evidence_label == node_label and evidence_position and node_position:
                    try:
                        ev_pos = np.array(evidence_position)
                        node_pos = np.array(node_position)
                        distance = np.linalg.norm(ev_pos - node_pos)
                        
                        if distance > 2.0:
                            # 冲突：创建新节点并添加 "possible_move" 边
                            new_node_id = self._create_new_node(evidence, cursor=None)
                            
                            conn = self._get_connection()
                            cursor = conn.cursor()
                            self._add_edge(best_candidate_id, new_node_id, "possible_move", weight=0.5, cursor=cursor)
                            self._add_edge(new_node_id, best_candidate_id, "possible_move", weight=0.5, cursor=cursor)
                            conn.commit()
                            conn.close()
                            
                            return new_node_id
                    except:
                        pass
                
                # 检查 embedding 相似度高但位置远的情况
                evidence_embedding_id = evidence.get("embedding_id")
                node_embedding_id = node_data.get("embedding_id")
                
                if similarity > 0.9 and evidence_position and node_position:
                    try:
                        ev_pos = np.array(evidence_position)
                        node_pos = np.array(node_position)
                        distance = np.linalg.norm(ev_pos - node_pos)
                        
                        if distance > 1.0:  # 位置距离 > 1.0m
                            # 仍然合并，但添加 "same_instance_possible" 边（低权重）
                            merged_node_id = self.merge_candidate_update(best_candidate_id, evidence)
                            
                            # 如果创建了新节点（由于其他原因），添加边
                            if merged_node_id != best_candidate_id:
                                conn = self._get_connection()
                                cursor = conn.cursor()
                                self._add_edge(best_candidate_id, merged_node_id, "same_instance_possible", weight=0.3, cursor=cursor)
                                conn.commit()
                                conn.close()
                            
                            return merged_node_id
                    except:
                        pass
            
            # 正常合并
            return self.merge_candidate_update(best_candidate_id, evidence)
        
        # 检查是否有相同 label 但位置距离 > 2.0m 的节点（冲突处理）
        evidence_label = evidence.get("label", "")
        evidence_position = evidence.get("position")
        
        if evidence_label and evidence_position:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, data_json FROM nodes")
            rows = cursor.fetchall()
            conn.close()
            
            for node_id, data_json_str in rows:
                try:
                    node_data = json.loads(data_json_str)
                    node_label = node_data.get("label", "")
                    node_position = node_data.get("position")
                    
                    if node_label == evidence_label and node_position:
                        try:
                            ev_pos = np.array(evidence_position)
                            node_pos = np.array(node_position)
                            distance = np.linalg.norm(ev_pos - node_pos)
                            
                            if distance > 2.0:
                                # 冲突：创建新节点并添加 "possible_move" 边
                                new_node_id = self._create_new_node(evidence, cursor=None)
                                
                                conn = self._get_connection()
                                cursor = conn.cursor()
                                self._add_edge(node_id, new_node_id, "possible_move", weight=0.5, cursor=cursor)
                                self._add_edge(new_node_id, node_id, "possible_move", weight=0.5, cursor=cursor)
                                conn.commit()
                                conn.close()
                                
                                return new_node_id
                        except:
                            pass
                except:
                    continue
        
        # 创建新节点
        return self._create_new_node(evidence, cursor=None)
    
    def _create_new_node(self, evidence: Dict, cursor=None) -> str:
        """
        创建新节点的内部方法
        
        Args:
            evidence: 证据字典
            cursor: 数据库游标（如果为 None，会创建新连接）
        
        Returns:
            node_id: 新节点 ID
        """
        node_id = f"node_{uuid.uuid4().hex[:12]}"
        
        # 准备节点数据
        evidence_ts = evidence.get("ts", time.time())
        node_data = {
            "type": evidence.get("type"),
            "label": evidence.get("label"),
            "confidence": evidence.get("confidence"),
            "embedding_id": evidence.get("embedding_id"),
            "embedding": evidence.get("embedding"),
            "position": evidence.get("position"),
            "metadata": evidence.get("metadata", {}),
            "created_ts": evidence_ts,
            "updated_ts": time.time(),
            "last_seen": time.time(),
            "detections_count": 1
        }
        
        # 初始化并更新时间直方图
        self._update_time_histogram(node_data, evidence_ts)
        
        # 写入 Chroma（如果有 embedding）
        if self.chroma_client and evidence.get("embedding"):
            try:
                collection = self.chroma_client.get_or_create_collection(name="robot_memory")
                collection.add(
                    ids=[node_id],
                    embeddings=[evidence["embedding"]],
                    metadatas=[{
                        "node_id": node_id,
                        "label": evidence.get("label"),
                        "type": evidence.get("type")
                    }],
                    documents=[evidence.get("label", "")]
                )
            except Exception as e:
                print(f"Warning: Failed to add to Chroma: {e}")
        
        # 插入节点到 SQLite
        if cursor is None:
            conn = self._get_connection()
            cursor = conn.cursor()
            should_close = True
        else:
            should_close = False
        
        cursor.execute("""
            INSERT INTO nodes (id, data_json, fused_conf, updated_ts)
            VALUES (?, ?, ?, ?)
        """, (
            node_id,
            json.dumps(node_data, ensure_ascii=False),
            evidence.get("confidence", 0.5),
            time.time()
        ))
        
        # 添加来源记录
        self._add_provenance(node_id, evidence, cursor)
        
        if should_close:
            conn.commit()
            conn.close()
        
        return node_id
    
    def decay_nodes(self, now_ts: Optional[float] = None) -> int:
        """
        对所有节点计算指数衰减的 fused_conf（使用事务保证线程安全）
        
        Args:
            now_ts: 当前时间戳（如果为 None，使用当前时间）
        
        Returns:
            更新的节点数量
        """
        if now_ts is None:
            now_ts = time.time()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 使用事务保证线程安全
        conn.execute("BEGIN TRANSACTION")
        
        try:
            # 获取所有节点
            cursor.execute("SELECT id, data_json, fused_conf, updated_ts FROM nodes")
            rows = cursor.fetchall()
            
            updated_count = 0
            for node_id, data_json_str, current_conf, updated_ts in rows:
                try:
                    node_data = json.loads(data_json_str)
                except:
                    continue
                
                # 使用统一的衰减方法
                new_conf, new_ts = self._apply_decay_to_node(
                    node_data, node_id,
                    now_ts=now_ts,
                    current_fused_conf=current_conf,
                    last_update_ts=updated_ts
                )
                
                # 如果置信度发生变化，更新节点
                if abs(new_conf - current_conf) > 1e-6:
                    cursor.execute("""
                        UPDATE nodes 
                        SET fused_conf = ?, updated_ts = ?
                        WHERE id = ?
                    """, (new_conf, new_ts, node_id))
                    updated_count += 1
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
        
        return updated_count
    
    def query_by_intent(self, intent: Dict, topk: int = 10, candidates: Optional[List[Dict]] = None,
                       beta1: float = 0.3, beta2: float = 0.4, beta3: float = 0.2, beta4: float = 0.1,
                       beta_time: float = 0.15, location_hint: Optional[List[float]] = None) -> List[Dict]:
        """
        基于意图查询节点，并进行图重排序
        
        Args:
            intent: 意图字典，可能包含:
                - text: 查询文本
                - object: 对象名称
                - location: 位置
            topk: 返回的节点数量
            candidates: 预选的候选列表（如果提供，将使用这些候选进行重排序）
            beta1: cosine_sim 权重（默认 0.3）
            beta2: fused_confidence 权重（默认 0.4）
            beta3: freshness_score 权重（默认 0.2）
            beta4: spatial_score 权重（默认 0.1）
            beta_time: time_score 权重（默认 0.15）
            location_hint: 位置提示 [x, y, z]（如果提供，将计算空间分数）
        
        Returns:
            排序后的节点列表，每个元素包含:
            - node_id: 节点 ID
            - fused_conf: 融合置信度
            - data: 完整节点数据
            - spatial: 空间信息
            - temporal: 时间信息
            - score: 最终评分
        """
        now_ts = time.time()
        
        # 如果提供了候选列表，需要从 MemoryGraph 获取节点数据
        if candidates is not None and len(candidates) > 0:
            # 从 MemoryGraph 获取候选节点的完整数据
            conn = self._get_connection()
            cursor = conn.cursor()
            
            processed_candidates = []
            for cand in candidates:
                node_id = cand.get("node_id")
                if not node_id:
                    continue
                
                # 从数据库获取节点数据
                cursor.execute("SELECT id, data_json, fused_conf, updated_ts FROM nodes WHERE id = ?", (node_id,))
                row = cursor.fetchone()
                
                if row:
                    db_node_id, data_json_str, fused_conf, updated_ts = row
                    node_data = json.loads(data_json_str)
                    
                    # 应用衰减
                    new_conf, new_ts = self._apply_decay_to_node(
                        node_data, db_node_id,
                        current_fused_conf=fused_conf,
                        last_update_ts=updated_ts
                    )
                    
                    # 如果置信度发生变化，更新数据库
                    if abs(new_conf - fused_conf) > 1e-6:
                        cursor.execute("""
                            UPDATE nodes 
                            SET fused_conf = ?, updated_ts = ?
                            WHERE id = ?
                        """, (new_conf, new_ts, db_node_id))
                        fused_conf = new_conf
                        updated_ts = new_ts
                    
                    processed_candidates.append({
                        "node_id": db_node_id,
                        "data": node_data,
                        "fused_conf": fused_conf,
                        "updated_ts": updated_ts,
                        "cos_sim": cand.get("cos_sim", 0.5)
                    })
            
            conn.close()
            candidates = processed_candidates
        
        # 如果没有提供候选列表或候选列表为空，从 Chroma 或 SQLite 查询
        if candidates is None or len(candidates) == 0:
            candidates = []
            query_text = intent.get("text") or intent.get("object")
            if query_text and self.chroma_client:
                try:
                    collection = self.chroma_client.get_or_create_collection(name="robot_memory")
                    results = collection.query(
                        query_texts=[query_text],
                        n_results=topk * 3  # 获取更多候选以便重排序
                    )
                    
                    # 从 Chroma 结果中提取 node_id
                    if results.get("ids") and results["ids"][0]:
                        chroma_ids = results["ids"][0]
                        chroma_distances = results.get("distances", [[]])[0] if results.get("distances") else []
                        
                        # 从 SQLite 获取节点数据
                        conn = self._get_connection()
                        cursor = conn.cursor()
                        
                        for i, chroma_id in enumerate(chroma_ids):
                            # 尝试直接使用 chroma_id 作为 node_id，或从 metadata 中获取
                            cursor.execute("SELECT id, data_json, fused_conf, updated_ts FROM nodes WHERE id = ?", (chroma_id,))
                            row = cursor.fetchone()
                            
                            if row:
                                node_id, data_json_str, fused_conf, updated_ts = row
                                cos_sim = 1.0 - chroma_distances[i] if i < len(chroma_distances) else 0.5
                                candidates.append({
                                    "node_id": node_id,
                                    "data": json.loads(data_json_str),
                                    "fused_conf": fused_conf,
                                    "updated_ts": updated_ts,
                                    "cos_sim": cos_sim
                                })
                        
                        conn.close()
                except Exception as e:
                    print(f"Warning: Chroma query failed: {e}")
            
            # 如果没有 Chroma 结果，从 SQLite 直接查询
            if not candidates:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                query_label = intent.get("object") or intent.get("label")
                if query_label:
                    cursor.execute("SELECT id, data_json, fused_conf, updated_ts FROM nodes")
                    rows = cursor.fetchall()
                    for node_id, data_json_str, fused_conf, updated_ts in rows:
                        try:
                            node_data = json.loads(data_json_str)
                            if query_label.lower() in node_data.get("label", "").lower():
                                # 应用衰减
                                new_conf, new_ts = self._apply_decay_to_node(
                                    node_data, node_id,
                                    current_fused_conf=fused_conf,
                                    last_update_ts=updated_ts
                                )
                                
                                # 如果置信度发生变化，更新数据库
                                if abs(new_conf - fused_conf) > 1e-6:
                                    cursor.execute("""
                                        UPDATE nodes 
                                        SET fused_conf = ?, updated_ts = ?
                                        WHERE id = ?
                                    """, (new_conf, new_ts, node_id))
                                    fused_conf = new_conf
                                    updated_ts = new_ts
                                
                                candidates.append({
                                    "node_id": node_id,
                                    "data": node_data,
                                    "fused_conf": fused_conf,
                                    "updated_ts": updated_ts,
                                    "cos_sim": 0.5  # 默认相似度
                                })
                        except:
                            continue
                
                conn.close()
        
        # 获取 location_hint（从 intent 或参数）
        if location_hint is None:
            location_hint = intent.get("location_hint") or intent.get("position")
        
        # 获取 time_hint（从 intent）
        time_hint = intent.get("time_hint")
        
        # 重排序: score = beta1*cos_sim + beta2*fused_conf + beta3*freshness + beta4*spatial + beta_time*time_score
        scored_candidates = []
        for cand in candidates:
            cos_sim = cand.get("cos_sim", 0.5)
            fused_conf = cand.get("fused_conf", 0.5)
            node_data = cand.get("data", {})
            
            # 计算新鲜度: freshness_score = exp(-(now - last_seen)/T) where T=24h
            last_seen = node_data.get("last_seen", cand.get("updated_ts", now_ts))
            dt = now_ts - last_seen
            T = 86400.0  # 24小时
            freshness_score = math.exp(-dt / T)
            
            # 计算空间分数: if intent contains location_hint compute 1 - normalized_distance (clamped)
            spatial_score = 0.0
            if location_hint:
                node_position = node_data.get("position")
                if node_position:
                    try:
                        hint_pos = np.array(location_hint)
                        node_pos = np.array(node_position)
                        distance = np.linalg.norm(hint_pos - node_pos)
                        # 归一化距离（假设最大距离为 10m）
                        max_distance = 10.0
                        normalized_distance = min(distance / max_distance, 1.0)
                        spatial_score = max(0.0, 1.0 - normalized_distance)
                    except:
                        pass
            
            # 计算时间分数: if intent contains time_hint
            time_score = 0.0
            if time_hint:
                time_score = self._compute_time_score(node_data, time_hint)
            
            # 综合评分
            final_score = (beta1 * cos_sim + 
                          beta2 * fused_conf + 
                          beta3 * freshness_score + 
                          beta4 * spatial_score +
                          beta_time * time_score)
            
            # 构建返回格式
            result = {
                "node_id": cand.get("node_id"),
                "fused_conf": fused_conf,
                "data": node_data,
                "spatial": {
                    "position": node_data.get("position"),
                    "location_semantic": node_data.get("metadata", {}).get("location_semantic")
                },
                "temporal": {
                    "last_seen": last_seen,
                    "updated_ts": cand.get("updated_ts", now_ts),
                    "created_ts": node_data.get("created_ts", now_ts)
                },
                "score": final_score,
                "cos_sim": cos_sim,
                "freshness_score": freshness_score,
                "spatial_score": spatial_score,
                "time_score": time_score
            }
            
            scored_candidates.append(result)
        
        # 按分数排序
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # 返回 topk 个结果
        return scored_candidates[:topk]
    
    def predict_time_probability(self, node_id: str, ts: float) -> float:
        """
        预测在给定时间戳下节点出现的概率
        
        Args:
            node_id: 节点 ID
            ts: 时间戳
        
        Returns:
            概率值（0-1）
        """
        node = self.get_node(node_id, apply_decay=False)
        if not node:
            return 0.0
        
        time_histogram = node.get("time_histogram", [0] * 24)
        total_count = sum(time_histogram)
        
        if total_count == 0:
            return 0.0
        
        # 将时间戳转换为小时
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        
        # 计算该小时的概率
        hour_count = time_histogram[hour]
        probability = hour_count / total_count if total_count > 0 else 0.0
        
        return probability
    
    def export_graph_snapshot(self, out_dir: str) -> str:
        """
        导出图快照到指定目录
        
        Args:
            out_dir: 输出目录路径
        
        Returns:
            生成的 HTML 文件路径
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 导出节点数据
        cursor.execute("SELECT id, data_json, fused_conf, updated_ts FROM nodes")
        nodes_data = []
        for row in cursor.fetchall():
            node_id, data_json_str, fused_conf, updated_ts = row
            try:
                node_data = json.loads(data_json_str)
                nodes_data.append({
                    "id": node_id,
                    "fused_conf": fused_conf,
                    "updated_ts": updated_ts,
                    "data": node_data
                })
            except:
                pass
        
        # 导出边数据
        cursor.execute("SELECT from_id, to_id, rel_type, weight, updated_ts FROM edges")
        edges_data = []
        for row in cursor.fetchall():
            from_id, to_id, rel_type, weight, updated_ts = row
            edges_data.append({
                "from_id": from_id,
                "to_id": to_id,
                "rel_type": rel_type,
                "weight": weight,
                "updated_ts": updated_ts
            })
        
        conn.close()
        
        # 保存 JSON 文件
        nodes_file = out_path / "nodes.json"
        edges_file = out_path / "edges.json"
        
        with open(nodes_file, 'w', encoding='utf-8') as f:
            json.dump(nodes_data, f, ensure_ascii=False, indent=2)
        
        with open(edges_file, 'w', encoding='utf-8') as f:
            json.dump(edges_data, f, ensure_ascii=False, indent=2)
        
        # 生成 HTML 索引
        html_file = out_path / "index.html"
        html_content = self._generate_html_index(nodes_data, edges_data)
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_file)
    
    def _generate_html_index(self, nodes_data: List[Dict], edges_data: List[Dict]) -> str:
        """生成 HTML 索引页面"""
        html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MemoryGraph Snapshot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }
        .stat-box {
            flex: 1;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 4px;
            border-left: 4px solid #4CAF50;
        }
        .stat-box h3 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }
        .stat-box .value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .node-id {
            font-family: monospace;
            color: #2196F3;
        }
        .conf-high { color: #4CAF50; font-weight: bold; }
        .conf-medium { color: #FF9800; }
        .conf-low { color: #f44336; }
        .rel-type {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            display: inline-block;
        }
        .rel-possible_move { background-color: #FFE0B2; }
        .rel-same_instance { background-color: #C8E6C9; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MemoryGraph Snapshot</h1>
        <div class="stats">
            <div class="stat-box">
                <h3>节点数量</h3>
                <div class="value">{node_count}</div>
            </div>
            <div class="stat-box">
                <h3>边数量</h3>
                <div class="value">{edge_count}</div>
            </div>
            <div class="stat-box">
                <h3>平均置信度</h3>
                <div class="value">{avg_conf:.3f}</div>
            </div>
        </div>
        
        <h2>节点列表</h2>
        <table>
            <thead>
                <tr>
                    <th>节点 ID</th>
                    <th>标签</th>
                    <th>置信度</th>
                    <th>检测次数</th>
                    <th>最后看到</th>
                    <th>位置</th>
                </tr>
            </thead>
            <tbody>
                {nodes_rows}
            </tbody>
        </table>
        
        <h2>边列表</h2>
        <table>
            <thead>
                <tr>
                    <th>源节点</th>
                    <th>目标节点</th>
                    <th>关系类型</th>
                    <th>权重</th>
                    <th>更新时间</th>
                </tr>
            </thead>
            <tbody>
                {edges_rows}
            </tbody>
        </table>
    </div>
</body>
</html>"""
        
        # 计算统计信息
        node_count = len(nodes_data)
        edge_count = len(edges_data)
        avg_conf = sum(n.get("fused_conf", 0.0) for n in nodes_data) / node_count if node_count > 0 else 0.0
        
        # 生成节点行
        nodes_rows = []
        for node in nodes_data:
            node_id = node["id"]
            data = node.get("data", {})
            label = data.get("label", "N/A")
            fused_conf = node.get("fused_conf", 0.0)
            detections_count = data.get("detections_count", 0)
            last_seen = data.get("last_seen", 0)
            position = data.get("position", [])
            
            # 格式化时间
            if last_seen:
                last_seen_str = datetime.fromtimestamp(last_seen, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            else:
                last_seen_str = "N/A"
            
            # 格式化位置
            if position:
                pos_str = f"[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]"
            else:
                pos_str = "N/A"
            
            # 置信度样式
            if fused_conf >= 0.7:
                conf_class = "conf-high"
            elif fused_conf >= 0.4:
                conf_class = "conf-medium"
            else:
                conf_class = "conf-low"
            
            nodes_rows.append(f"""
                <tr>
                    <td class="node-id">{node_id}</td>
                    <td>{label}</td>
                    <td class="{conf_class}">{fused_conf:.3f}</td>
                    <td>{detections_count}</td>
                    <td>{last_seen_str}</td>
                    <td>{pos_str}</td>
                </tr>
            """)
        
        # 生成边行
        edges_rows = []
        for edge in edges_data:
            from_id = edge["from_id"]
            to_id = edge["to_id"]
            rel_type = edge["rel_type"]
            weight = edge["weight"]
            updated_ts = edge["updated_ts"]
            
            updated_str = datetime.fromtimestamp(updated_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            
            # 关系类型样式
            if "move" in rel_type.lower():
                rel_class = "rel-possible_move"
            else:
                rel_class = "rel-same_instance"
            
            edges_rows.append(f"""
                <tr>
                    <td class="node-id">{from_id}</td>
                    <td class="node-id">{to_id}</td>
                    <td><span class="rel-type {rel_class}">{rel_type}</span></td>
                    <td>{weight:.3f}</td>
                    <td>{updated_str}</td>
                </tr>
            """)
        
        # 转义 HTML 特殊字符
        def escape_html(text):
            if text is None:
                return ""
            return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        
        # 转义所有文本内容
        nodes_rows_escaped = []
        for row in nodes_rows:
            # 已经格式化的 HTML，不需要再次转义
            nodes_rows_escaped.append(row)
        
        edges_rows_escaped = []
        for row in edges_rows:
            edges_rows_escaped.append(row)
        
        # 使用字符串替换而不是 format（避免 CSS 中的花括号问题）
        html = html_template.replace("{node_count}", str(node_count))
        html = html.replace("{edge_count}", str(edge_count))
        html = html.replace("{avg_conf:.3f}", f"{avg_conf:.3f}")
        html = html.replace("{nodes_rows}", "".join(nodes_rows_escaped))
        html = html.replace("{edges_rows}", "".join(edges_rows_escaped))
        
        return html

