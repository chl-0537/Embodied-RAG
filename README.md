# Embodied-RAG
探索具身智能RAG技术

# Embodied-RAG 项目架构与代码解释

## 项目概述

这是一个**机器人感知 + RAG + 世界模型**的完整系统，用于在3D场景中搜索目标物体。系统集成了：
- **Habitat-Sim**：3D场景模拟器
- **视觉检测**：基于ChatGLM的多物体检测
- **3D空间融合**：多视角几何三角化定位
- **记忆图（MemoryGraph）**：基于图的记忆存储与检索
- **世界模型（WorldModel）**：动态场景状态管理
- **RAG框架**：检索增强生成，结合记忆与视觉

---

## 核心模块架构

```
┌─────────────────────────────────────────────────────────────┐
│                    habitat_hm3d_rag_search.py               │
│     (主执行脚本：Habitat环境 + 探索策略 + 3D融合)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  rag_robot_framework.py                     │
│  (RAG核心：LLM调用 + 记忆检索 + MemoryGraph/WorldModel接口)   │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────┐    ┌──────────────────────────┐
│   memory_graph.py    │    │    world_model.py        │
│  (记忆图：证据合并、   │    │  (世界模型：实体管理、    │
│   衰减、图检索)       │    │   动态场景跟踪)           │
└──────────────────────┘    └──────────────────────────┘
```

---

## 1. rag_robot_framework.py - RAG核心框架

### 1.1 核心功能

这是整个系统的**中央协调模块**，负责：

#### **初始化系统组件**
```python
init_chroma_db()
```
- 初始化 **ChromaDB** 向量数据库（用于语义检索）
- 初始化 **MemoryGraph**（SQLite + 图结构）
- 初始化 **WorldModel**（SQLite + 实体管理）

#### **记忆管理**
- `add_vision_memory()`: 将视觉检测结果写入记忆
- `add_or_update_memory()`: 将融合结果写入MemoryGraph
- `retrieve_candidates_with_time_prior()`: 基于时间优先级的记忆检索
- `retrieve_candidates_by_intent()`: 基于意图的记忆检索

#### **LLM集成**
- `call_chatglm_multi_object_detection()`: 多物体视觉检测
- `parse_user_intent_llm()`: 解析用户意图
- `generate_action_plan_llm()`: 生成行动计划

#### **融合结果处理**
```python
process_fused_result(fused_result)
```
**核心函数**：统一处理3D融合结果
1. 写入 MemoryGraph（通过 `add_or_update_memory()`）
2. 同步到 WorldModel（通过 `world_model.ingest_observation()`）
3. 处理实体重链接、移动检测等

### 1.2 关键数据结构

```python
# 融合结果格式
fused_result = {
    "label": "水杯",
    "fused_3d_pos": [x, y, z],  # 融合后的3D位置
    "confidence": 0.85,
    "bbox": [x1, y1, x2, y2],
    "image_path": "...",
    "agent_position": [x, y, z],
    "agent_rotation": [x, y, z, w],
    "ts": timestamp,
    "metadata": {...}
}
```

---

## 2. memory_graph.py - 记忆图模块

### 2.1 设计理念

**MemoryGraph** 是一个轻量级图数据库，使用 SQLite 存储：
- **节点（nodes）**：记忆实体（物体、位置、事件）
- **边（edges）**：节点间关系（相同实例、可能移动等）
- **来源（provenance）**：证据来源追踪

### 2.2 核心功能

#### **证据合并（Evidence Merging）**
```python
add_evidence(evidence)
```
- **查找合并候选**：基于 embedding 相似度（>0.85）或空间距离（<0.2m）
- **冲突检测**：相同 label 但位置距离 >2.0m → 创建新节点 + "possible_move" 边
- **自动合并**：相似证据自动合并到同一节点，更新置信度

#### **置信度衰减（Confidence Decay）**
```python
decay_nodes(now_ts)
```
- **指数衰减模型**：`new_conf = old_conf * exp(-dt / halflife)`
- **默认半衰期**：24小时
- **自动更新**：每次检索时自动应用衰减

#### **图检索重排序**
```python
query_by_intent(intent, topk=10)
```
- **多因子评分**：
  - `beta1`: 语义相似度（embedding）
  - `beta2`: 空间距离
  - `beta3`: 时间相关性
  - `beta4`: 图结构（邻居节点）
  - `beta_time`: 时间语义（如"早上"匹配"morning"记忆）

### 2.3 数据库结构

```sql
-- nodes 表
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    data_json TEXT,        -- JSON格式的节点数据
    fused_conf REAL,       -- 融合置信度
    updated_ts REAL        -- 最后更新时间
);

-- edges 表
CREATE TABLE edges (
    from_id TEXT,
    to_id TEXT,
    rel_type TEXT,        -- "same_instance", "possible_move", etc.
    weight REAL
);

-- provenance 表
CREATE TABLE provenance (
    node_id TEXT,
    source TEXT,          -- "vision", "fused_observation", etc.
    confidence REAL,
    ts REAL,
    meta_json TEXT
);
```

---

## 3. world_model.py - 世界模型模块

### 3.1 设计理念

**WorldModel** 管理**动态场景状态**：
- **实体（Entities）**：场景中的物体
- **事件（Events）**：移动、消失、出现等
- **状态跟踪**：active → missing → removed

### 3.2 核心功能

#### **实体管理**
```python
create_entity(label, position, confidence, ts)
```
- 创建新实体，分配唯一 `entity_id`
- 记录位置历史 `pos_history`
- 初始状态：`"active"`

#### **观察处理**
```python
ingest_observation(obs)
```
**核心函数**：处理新的观察数据
1. **重链接**：通过 `relink_candidate()` 查找空间接近的现有实体（距离 < 0.3m）
2. **更新**：如果链接成功，更新位置历史、`last_seen_ts`
3. **创建**：如果未链接，创建新实体

#### **动态场景跟踪**
```python
mark_no_detection(entity_id, evidence)
```
- 记录"未检测到"的证据（即使实体在视野内）
- 用于评估 missing 状态

```python
evaluate_missing(entity_id, n_no_detects=3, vis_th=0.6)
```
- 如果连续 `n_no_detects` 次未检测到，且可见性分数 > `vis_th`
- 状态转换：`active` → `missing`

```python
confirm_removed(entity_id, removed_T=30*60)
```
- 如果 missing 状态持续超过 `removed_T` 秒（默认30分钟）
- 状态转换：`missing` → `removed`

#### **移动检测**
```python
register_movement(entity_id, new_pos, method, confidence)
```
- 检测实体位置变化（距离 > 0.5m）
- 记录移动事件到 `movement_events`
- 状态更新：`active` → `moved` → `active`

### 3.3 实体数据结构

```python
entity = {
    "entity_id": "uuid",
    "label": "水杯",
    "position": [x, y, z],
    "pos_history": [
        {"ts": t1, "pos": [x1, y1, z1], "conf": 0.8},
        {"ts": t2, "pos": [x2, y2, z2], "conf": 0.9}
    ],
    "status": "active",  # "active" | "missing" | "moved" | "removed"
    "last_seen_ts": timestamp,
    "missing_since": None,
    "missing_evidence": [],
    "movement_events": [],
    "fused_node_id": "memory_graph_node_id"  # 关联到 MemoryGraph
}
```

---

## 4. habitat_hm3d_rag_search.py - 主执行脚本

### 4.1 整体流程

```python
habitat_rag_search(scene_path, target_object, max_steps=1000)
```

#### **阶段1：初始化**
1. 初始化 ChromaDB + MemoryGraph + WorldModel
2. 解析用户意图（"找到水杯"）
3. 初始化 Habitat 场景和机器人代理
4. 选择探索策略（`frontier` 或 `simple`）

#### **阶段2：初始环扫（Sweep）**
```python
for i in range(rotate_sweep):  # 默认12次，每次旋转30度
    rgb = agent.get_rgb()
    depth = agent.get_depth()
    vision_result = call_chatglm_multi_object_detection(rgb)
    
    # 如果检测到目标
    if strict_obj:
        # 添加到空间融合模块
        fusion_module.add_observation(
            position, rotation, bbox, confidence, label,
            img_shape, depth
        )
        
        # 估计3D位置
        estimated_3d_pos = fusion_module.estimate_object_position()
        
        # 处理融合结果（写入MemoryGraph + WorldModel）
        process_result = rag.process_fused_result(fused_result)
```

#### **阶段3：探索循环（Explore）**
```python
for step in range(1, max_steps + 1):
    # 1. 获取观测
    rgb, depth = agent.get_observations()
    
    # 2. 视觉检测
    vision_result = call_chatglm_multi_object_detection(rgb)
    rag.add_vision_memory(vision_result, ...)
    
    # 3. 如果检测到目标
    if strict_obj:
        # 3D融合
        estimated_3d_pos = fusion_module.estimate_object_position()
        
        # 处理融合结果
        rag.process_fused_result(fused_result)
        
        # 多角度验证
        if verify_ok:
            return SUCCESS
    
    # 4. 探索策略决定下一步动作
    action = policy.plan(current_pos, depth, ...)
    agent.step(action)
```

#### **阶段4：后处理**
```python
_post_process_memory_world()
```
- `memory_graph.decay_nodes()`: 应用置信度衰减
- `world_model.evaluate_missing()`: 评估 missing 状态
- `world_model.confirm_removed()`: 确认 removed 状态

### 4.2 空间语义融合模块

```python
class SpatialSemanticFusionModule:
```

**功能**：通过多视角几何三角化实现精确3D定位

#### **融合模式**
1. **`triangulate`**（默认）：RANSAC三角化
   - 从多个视角提取射线
   - 使用RANSAC找到最佳交点
   - 内点比例 > 30% 才接受

2. **`depth`**：深度反投影
   - 直接从深度图反投影到3D
   - 适用于有深度信息的场景

3. **`voxel_vote`**：微体素投票
   - 将所有候选点（深度+三角化）量化到体素网格
   - 投票选择最高投票的体素中心

#### **异常检测**
```python
_check_observation_anomaly(bbox, depth, ...)
```
- bbox 太小/太大（<0.05% 或 >80%）
- bbox 中心太靠近边缘（<5%边距）
- 深度值无效（<0.1m 或 >10m）
- 深度区域内无有效值

#### **观测添加**
```python
add_observation(position, rotation, bbox, confidence, label, depth)
```
1. 归一化 bbox（自动检测归一化/像素坐标）
2. 检查异常（拒绝异常观测）
3. 深度反投影（如果有深度图）
4. 添加到缓冲区（最多 `max_buffer` 帧）

#### **位置估计**
```python
estimate_object_position()
```
1. **优先使用深度**：如果有深度观测，使用加权平均
2. **RANSAC三角化**：提取射线，RANSAC找交点
3. **一致性检查**：与上一个可靠估计距离 >2m → 降低置信度
4. **记录追踪日志**：保存到 `fusion_trace.json`

### 4.3 探索策略

#### **FrontierBasedExplorer**（前沿探索）
- **前沿检测**：基于深度图检测已探索/未探索边界
- **A*路径规划**：规划到前沿点的路径
- **知识引导**：结合RAG记忆推断目标可能位置

#### **SimpleExplorePolicy**（简单策略）
- 网格量化位置
- 避免重复访问
- 随机转向探索

---

## 5. 数据流与交互

### 5.1 完整数据流

```
视觉检测 (ChatGLM)
    ↓
add_vision_memory() → ChromaDB (向量检索)
    ↓
空间融合 (SpatialSemanticFusionModule)
    ↓
estimated_3d_pos
    ↓
process_fused_result()
    ├─→ add_or_update_memory() → MemoryGraph
    │       ├─→ 查找合并候选 (embedding/空间相似度)
    │       ├─→ 合并或创建新节点
    │       └─→ 更新置信度
    │
    └─→ world_model.ingest_observation() → WorldModel
            ├─→ relink_candidate() (空间重链接)
            ├─→ 更新或创建实体
            └─→ 记录位置历史
```

### 5.2 记忆检索流程

```
用户意图 ("找到水杯")
    ↓
retrieve_candidates_with_time_prior()
    ├─→ ChromaDB 向量检索 (语义相似度)
    ├─→ MemoryGraph 图检索重排序
    │       ├─→ 语义相似度 (beta1)
    │       ├─→ 空间距离 (beta2)
    │       ├─→ 时间相关性 (beta3)
    │       ├─→ 图结构 (beta4)
    │       └─→ 时间语义 (beta_time)
    └─→ 时间优先级排序
    ↓
返回 top-k 记忆证据
    ↓
用于探索策略的知识引导
```

---

## 6. 关键技术点

### 6.1 多模态融合
- **视觉 + 空间**：bbox + 深度图 → 3D位置
- **语义 + 空间**：embedding相似度 + 空间距离 → 记忆合并
- **时间 + 语义**：时间上下文（早上/晚上）影响记忆检索

### 6.2 动态场景处理
- **Missing Detection**：连续未检测到 → missing 状态
- **Movement Tracking**：位置变化 >0.5m → 移动事件
- **Removal Confirmation**：missing 持续30分钟 → removed 状态

### 6.3 鲁棒性设计
- **异常检测**：拒绝异常观测（bbox太小/太大、深度无效）
- **RANSAC**：鲁棒三角化，处理噪声和异常值
- **置信度衰减**：自动降低旧记忆的置信度
- **重链接机制**：空间接近的观察自动链接到同一实体

---

## 7. 配置与参数

### 7.1 关键参数

```python
# 空间融合
max_buffer = 10          # 缓冲区最大帧数
min_views = 3            # 三角化最少视角数
spatial_th = 0.3         # 空间重链接阈值（米）
move_th = 0.5           # 移动检测阈值（米）

# 记忆合并
embedding_sim_th = 0.85  # embedding相似度阈值
spatial_merge_th = 0.2   # 空间合并阈值（米）
spatial_conflict_th = 2.0 # 空间冲突阈值（米）

# 动态场景
n_no_detects = 3        # missing检测所需连续未检测次数
vis_th = 0.6            # 可见性阈值
removed_T = 30*60       # removed确认时间（30分钟）

# 置信度衰减
halflife_hours = 24     # 半衰期（24小时）
```

---

## 8. 总结

这是一个**完整的机器人感知与记忆系统**，实现了：

1. **多视角3D定位**：通过几何三角化精确估计物体位置
2. **长期记忆管理**：MemoryGraph 支持证据合并、衰减、图检索
3. **动态场景跟踪**：WorldModel 跟踪实体状态变化（出现、移动、消失）
4. **RAG增强探索**：结合记忆检索和知识引导的智能探索
5. **鲁棒性设计**：异常检测、RANSAC、置信度衰减等机制

系统设计遵循**模块化、可扩展**的原则，各模块职责清晰，便于维护和扩展。


