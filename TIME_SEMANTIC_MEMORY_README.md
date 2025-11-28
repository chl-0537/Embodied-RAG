# 时间语义记忆模块文档

## 概述

时间语义记忆模块使机器人能够学习物品随时间变化的位置规律，并在 RAG 检索和路径规划中利用时间信息优化物品查找效率。

## 功能实现

### 1. 时间语义解析模块

**函数**: `extract_time_context(timestamp: float) -> Dict[str, Any]`

从 Unix 时间戳提取时间语义上下文。

**输入**:
- `timestamp`: Unix 时间戳（秒）

**输出**:
```python
{
    "hour": int,          # 0-23
    "weekday": int,       # 0=Monday, 6=Sunday
    "period": str,        # "dawn"|"morning"|"afternoon"|"evening"|"night"
    "is_weekend": bool    # True if Saturday or Sunday
}
```

**时间段定义**:
- 0-6: `dawn` (黎明)
- 6-12: `morning` (上午)
- 12-18: `afternoon` (下午)
- 18-22: `evening` (晚上)
- 22-24: `night` (深夜)

### 2. 视觉记忆增强

**修改的函数**: `add_vision_memory()`

为每条视觉记忆自动添加 `time_context` 字段：

```python
vision_metadata = {
    ...
    "time_context": extract_time_context(time.time()),
    ...
}
```

**修改的函数**: `add_memory_entry_simple()`

确保 `time_context` 写入 Chroma metadata，包括：
- `time_context_hour`: 小时（字符串）
- `time_context_period`: 时间段
- `time_context_weekday`: 星期几（字符串）
- `time_context_is_weekend`: 是否周末（字符串）
- `time_context_json`: 完整时间上下文的 JSON 字符串

### 3. 时间优先级评分

**函数**: `time_prior_score(memory_item: Dict, current_time_context: Dict) -> float`

计算时间优先级分数，用于提升与当前时间上下文匹配的记忆的检索优先级。

**评分规则**:
- 时间段完全一致: `1.8`
- 小时差在 2 小时内: `1.2`
- 其他情况: `0.8`
- 无时间上下文: `1.0` (默认)

### 4. RAG 检索排序增强

**修改的函数**: `rerank_with_time_space()`

在原有评分基础上整合时间优先级：

```python
final_score = (
    sim * 0.5 +                    # 视觉相似度
    conf * 0.2 +                   # 置信度
    w_time * 0.1 +                 # 时间衰减
    w_space * 0.05 +               # 空间距离
    (time_prior - 1.0) * 0.15      # 时间优先级
)
```

### 5. 时间规律分析

**函数**: `analyze_time_patterns(object_name: str) -> Dict[str, Dict[str, float]]`

从 Chroma memory 中分析物品的时间规律，按时间段统计物品在不同位置的出现频率。

**输出示例**:
```python
{
    "morning": {"desk": 0.82, "kitchen": 0.12, ...},
    "afternoon": {"desk": 0.45, "living_room": 0.35, ...},
    "evening": {...},
    "night": {...},
    "dawn": {...}
}
```

**用途**:
- 在计划路径时可直接使用
- 后续可扩展为"时间 → 空间概率模型"

### 6. 时间优先检索

**函数**: `retrieve_candidates_with_time_prior(intent: Dict, current_time_context: Optional[Dict] = None, topk: int = 20) -> List[Dict]`

基于意图进行检索，并自动应用时间优先级重新排序。

**参数**:
- `intent`: 意图字典，包含 object, location 等字段
- `current_time_context`: 当前时间上下文（如果为 None，则自动提取）
- `topk`: 返回的候选数量

**流程**:
1. 调用 `retrieve_candidates_by_intent()` 获取初始候选
2. 使用 `rerank_with_time_space()` 应用时间优先级重新排序
3. 返回 topk 个结果

### 7. Habitat 集成

**修改的文件**: `habitat_hm3d_rag_search.py`

在搜索开始前自动提取当前时间上下文：

```python
# 提取当前时间上下文（用于时间语义记忆）
current_time_context = rag.extract_time_context(time.time())
```

所有记忆检索调用已更新为使用时间优先级：

```python
# 旧代码
memory_evidences = rag.retrieve_candidates_by_intent(intent, topk=5)

# 新代码
memory_evidences = rag.retrieve_candidates_with_time_prior(
    intent, 
    current_time_context=current_time_context, 
    topk=5
)
```

## 兼容性保证

✅ **不修改空间语义融合模块** (`SpatialSemanticFusionModule`)

✅ **不破坏 Chroma 的 metadata schema** - 时间上下文作为额外字段添加

✅ **不影响视觉检索与三角化** - 时间优先级仅影响记忆检索排序

✅ **所有新增内容可插拔** - 如果 `current_time_context` 为 None，函数会自动提取

## 使用示例

### 基本使用

```python
import rag_robot_framework as rag

# 初始化数据库
rag.init_chroma_db()

# 提取当前时间上下文
time_ctx = rag.extract_time_context(time.time())
print(time_ctx)
# {'hour': 14, 'weekday': 2, 'period': 'afternoon', 'is_weekend': False}

# 使用时间优先级检索
intent = {"object": "水杯", "location": "desk"}
results = rag.retrieve_candidates_with_time_prior(intent, current_time_context=time_ctx, topk=10)

# 分析物品时间规律
patterns = rag.analyze_time_patterns("水杯")
print(patterns)
# {'morning': {'desk': 0.82, 'kitchen': 0.12}, ...}
```

### 在 Habitat 搜索中使用

时间语义功能已自动集成到 `habitat_rag_search()` 函数中，无需额外配置。

## 技术细节

### 时间上下文存储

时间上下文以两种方式存储在 Chroma metadata 中：

1. **结构化字段**（用于查询过滤）:
   - `time_context_hour`: 小时
   - `time_context_period`: 时间段
   - `time_context_weekday`: 星期几
   - `time_context_is_weekend`: 是否周末

2. **JSON 字符串**（用于完整信息）:
   - `time_context_json`: 完整时间上下文的 JSON 字符串

### 评分权重

时间优先级在最终评分中的权重为 `0.15`，与其他因子平衡：
- 视觉相似度: 50%
- 置信度: 20%
- 时间衰减: 10%
- 空间距离: 5%
- 时间优先级: 15%

### 性能优化

- `current_time_context` 在搜索开始时计算一次，避免重复计算
- 时间优先级评分使用简单的字典查找，性能开销极小
- 时间规律分析使用批量查询，一次获取所有相关记忆

## 未来扩展

1. **时间-空间概率模型**: 基于 `analyze_time_patterns()` 的结果，构建更复杂的概率模型
2. **自适应权重**: 根据历史数据动态调整时间优先级权重
3. **周期性模式识别**: 识别物品位置的周期性规律（如工作日 vs 周末）
4. **多时间尺度**: 支持更细粒度的时间分析（如小时级别）

## 注意事项

1. 时间上下文基于 UTC 时间，确保系统时区设置正确
2. 对于没有时间上下文的旧记忆，使用默认分数 `1.0`
3. 时间优先级是软约束，不会完全排除不匹配的记忆
4. 时间段定义是固定的，如需修改请更新 `extract_time_context()` 函数

