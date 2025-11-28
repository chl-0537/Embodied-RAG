# 运行指南

## 快速开始

### 方法 1：使用测试脚本（推荐）

```bash
cd /home/szu/chl/Embodied-RAG

# 基本用法
./test_world_model_integration.sh <scene_path> <target_object> [max_steps] [rotate_sweep]

# 示例：短流程测试（验证功能）
./test_world_model_integration.sh /path/to/scene.glb 花瓶 40 4

# 示例：完整流程测试
./test_world_model_integration.sh /path/to/scene.glb watercup 1000 12
```

### 方法 2：直接运行主程序

```bash
cd /home/szu/chl/Embodied-RAG

# 基本命令
python3 habitat_hm3d_rag_search.py \
    --scene <场景文件路径> \
    --target <目标物品名称> \
    [其他参数...]

# 示例：短流程测试（验证 WorldModel 集成）
python3 habitat_hm3d_rag_search.py \
    --scene /path/to/scene.glb \
    --target 花瓶 \
    --max_steps 40 \
    --rotate_sweep 4 \
    --fusion_mode triangulate

# 示例：完整流程
python3 habitat_hm3d_rag_search.py \
    --scene /path/to/scene.glb \
    --target watercup \
    --max_steps 1000 \
    --rotate_sweep 12 \
    --explore_policy frontier \
    --fusion_mode triangulate
```

## 参数说明

### 必需参数

- `--scene`: Habitat 场景文件路径（.glb 或 .json 格式）
- `--target`: 目标物品名称（支持中文或英文，如 "花瓶"、"watercup"）

### 可选参数

- `--max_steps`: 最大探索步数（默认 1000，测试时建议用 40）
- `--rotate_sweep`: 初始环扫旋转次数（默认 12，测试时建议用 4）
- `--fusion_mode`: 融合模式
  - `triangulate`: RANSAC三角化（默认，推荐）
  - `depth`: 深度反投影
  - `voxel_vote`: 微体素投票
- `--explore_policy`: 探索策略
  - `frontier`: 前沿探索+知识引导（默认，推荐）
  - `simple`: 简单策略
- `--save_obs_dir`: 保存观测数据的目录（可选，不提供则自动生成）
- `--dataset_config`: Habitat 数据集配置文件路径（可选）

## 运行后验证

### 1. 查看运行日志

```bash
# 查看最新的日志文件
tail -f logs/rag_robot_*.log

# 或查看最新的日志
LATEST_LOG=$(ls -t logs/rag_robot_*.log | head -1)
cat "$LATEST_LOG" | grep -i "worldmodel\|memorygraph\|融合结果"
```

### 2. 验证 WorldModel 和 MemoryGraph 集成

```bash
# 运行验证脚本
python3 verify_world_model.py
```

### 3. 检查运行结果

```bash
# 查找最新的运行目录
LATEST_RUN=$(ls -td run_logs/obs_* | head -1)
echo "运行目录: $LATEST_RUN"

# 查看路径轨迹
cat "$LATEST_RUN/path_trajectory.json" | python3 -m json.tool | head -50

# 查看计划
cat "$LATEST_RUN/plan.json" | python3 -m json.tool

# 检查 WorldModel 数据库
python3 << 'EOF'
import sqlite3
import json

conn = sqlite3.connect("vector_db/world_model.sqlite")
cursor = conn.cursor()

# 查看实体
cursor.execute("SELECT COUNT(*) FROM entities")
print(f"实体数量: {cursor.fetchone()[0]}")

cursor.execute("SELECT entity_id, data FROM entities LIMIT 5")
for eid, data in cursor.fetchall():
    ent = json.loads(data)
    print(f"  - {eid[:8]}...: {ent.get('label')} [{ent.get('status')}]")

# 查看事件
cursor.execute("SELECT COUNT(*) FROM events")
print(f"\n事件数量: {cursor.fetchone()[0]}")

cursor.execute("SELECT event_id, data FROM events ORDER BY json_extract(data, '$.ts') DESC LIMIT 5")
for eid, data in cursor.fetchall():
    evt = json.loads(data)
    print(f"  - {evt.get('type')}: {evt.get('entities')}")

conn.close()
EOF
```

## 预期输出

运行成功后，你应该看到：

1. **控制台输出**：
   - 环扫阶段的检测信息
   - 探索阶段的动作和检测结果
   - 最后会打印 MemoryGraph & WorldModel 摘要

2. **摘要信息**（程序结束时）：
   ```
   ============================================================
   MemoryGraph & WorldModel 摘要
   ============================================================
   
   MemoryGraph:
     节点数: X
     边数: Y
     最近5个节点: ...
   
   WorldModel:
     活跃实体: X
     Missing 实体: Y
     已移除实体: Z
     总实体数: N
     最近5个实体: ...
   ```

3. **生成的文件**：
   - `run_logs/obs_<timestamp>_<scene>_<target>/` - 运行日志目录
     - `path_trajectory.json` - 路径轨迹
     - `plan.json` - 行动计划
     - `fusion_trace.json` - 融合追踪
     - `explore/` - 探索阶段图像
     - `sweep/` - 环扫阶段图像
     - `detections/` - 检测结果
   - `vector_db/world_model.sqlite` - WorldModel 数据库
   - `vector_db/memory_graph.sqlite` - MemoryGraph 数据库
   - `logs/rag_robot_*.log` - 运行日志

## 常见问题

### 1. 缺少依赖

如果遇到 `ModuleNotFoundError`，请安装依赖：

```bash
pip install chromadb langchain-openai numpy pillow
```

### 2. 场景文件路径

确保场景文件路径正确。Habitat 场景文件通常是 `.glb` 格式。

### 3. 内存不足

如果遇到内存问题，可以：
- 减少 `--max_steps`
- 减少 `--rotate_sweep`
- 使用 `--fusion_mode depth`（更轻量）

### 4. 查看详细错误

```bash
# 查看完整日志
cat logs/rag_robot_*.log | tail -100
```

## 测试建议

### 快速功能验证（推荐）

```bash
# 使用短流程测试，快速验证功能
./test_world_model_integration.sh /path/to/scene.glb 花瓶 40 4

# 然后验证结果
python3 verify_world_model.py
```

### 完整流程测试

```bash
# 完整流程，验证长期运行
python3 habitat_hm3d_rag_search.py \
    --scene /path/to/scene.glb \
    --target 花瓶 \
    --max_steps 1000 \
    --rotate_sweep 12
```

## 下一步

运行成功后，你可以：

1. 查看 `run_logs/` 目录下的运行结果
2. 使用 `verify_world_model.py` 验证 WorldModel 和 MemoryGraph 的状态
3. 检查日志文件了解详细执行过程
4. 查看生成的 `plan.json` 了解机器人的行动计划

