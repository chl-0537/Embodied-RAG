#!/bin/bash
# WorldModel 集成功能测试脚本

set -e

echo "============================================================"
echo "WorldModel 集成功能测试"
echo "============================================================"
echo ""

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <scene_path> <target_object> [max_steps] [rotate_sweep]"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/scene.glb 花瓶 40 4"
    echo "  $0 /path/to/scene.glb watercup 40 4"
    echo ""
    echo "参数说明:"
    echo "  scene_path:    Habitat 场景文件路径 (.glb 或 .json)"
    echo "  target_object: 目标物品名称（中文或英文）"
    echo "  max_steps:     最大探索步数（默认 40）"
    echo "  rotate_sweep:  初始环扫旋转次数（默认 4）"
    exit 1
fi

SCENE_PATH="$1"
TARGET_OBJECT="$2"
MAX_STEPS="${3:-40}"
ROTATE_SWEEP="${4:-4}"

echo "测试配置:"
echo "  场景路径: $SCENE_PATH"
echo "  目标物品: $TARGET_OBJECT"
echo "  最大步数: $MAX_STEPS"
echo "  环扫次数: $ROTATE_SWEEP"
echo ""

# 检查场景文件是否存在
if [ ! -f "$SCENE_PATH" ]; then
    echo "错误: 场景文件不存在: $SCENE_PATH"
    exit 1
fi

# 运行脚本
echo "开始运行 habitat_hm3d_rag_search.py..."
echo ""

python3 habitat_hm3d_rag_search.py \
    --scene "$SCENE_PATH" \
    --target "$TARGET_OBJECT" \
    --max_steps "$MAX_STEPS" \
    --rotate_sweep "$ROTATE_SWEEP" \
    --fusion_mode "triangulate"

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 脚本执行完成"
else
    echo "✗ 脚本执行失败 (退出码: $EXIT_CODE)"
fi
echo "============================================================"
echo ""

# 查找最新的运行日志目录
LATEST_RUN_DIR=$(ls -td run_logs/obs_* 2>/dev/null | head -1)

if [ -n "$LATEST_RUN_DIR" ]; then
    echo "运行日志目录: $LATEST_RUN_DIR"
    echo ""
    
    # 检查 WorldModel 数据库
    WORLD_MODEL_DB="vector_db/world_model.sqlite"
    if [ -f "$WORLD_MODEL_DB" ]; then
        echo "✓ 找到 WorldModel 数据库: $WORLD_MODEL_DB"
        echo ""
        echo "检查数据库内容..."
        python3 << 'PYEOF'
import sqlite3
import json
import sys

db_path = "vector_db/world_model.sqlite"
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 检查 entities 表
    cursor.execute("SELECT COUNT(*) FROM entities")
    entity_count = cursor.fetchone()[0]
    print(f"  实体数量: {entity_count}")
    
    if entity_count > 0:
        cursor.execute("SELECT entity_id, label, status, position FROM entities LIMIT 5")
        entities = cursor.fetchall()
        print("  前5个实体:")
        for eid, label, status, pos in entities:
            pos_str = json.loads(pos) if pos else "None"
            print(f"    - {eid}: {label} (status: {status}, position: {pos_str})")
    
    # 检查 events 表
    cursor.execute("SELECT COUNT(*) FROM events")
    event_count = cursor.fetchone()[0]
    print(f"  事件数量: {event_count}")
    
    if event_count > 0:
        cursor.execute("SELECT event_id, event_type, entity_id, data FROM events ORDER BY timestamp DESC LIMIT 10")
        events = cursor.fetchall()
        print("  最近10个事件:")
        for eid, etype, entity_id, data in events:
            data_str = json.loads(data) if data else "{}"
            print(f"    - {etype} (entity: {entity_id}): {data_str}")
    
    conn.close()
    print("")
    print("✓ 数据库检查完成")
except Exception as e:
    print(f"✗ 数据库检查失败: {e}")
    sys.exit(1)
PYEOF
    else
        echo "⚠ 未找到 WorldModel 数据库: $WORLD_MODEL_DB"
    fi
    
    # 检查 path_trajectory.json
    PATH_TRAJ="$LATEST_RUN_DIR/path_trajectory.json"
    if [ -f "$PATH_TRAJ" ]; then
        echo ""
        echo "检查 path_trajectory.json..."
        python3 << 'PYEOF'
import json
import sys

path_traj_file = sys.argv[1]
try:
    with open(path_traj_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    trajectory = data.get("trajectory", [])
    print(f"  轨迹条目数: {len(trajectory)}")
    
    # 查找 confirm_removed 事件
    confirm_removed_events = [e for e in trajectory if e.get("action") == "confirm_removed"]
    print(f"  confirm_removed 事件数: {len(confirm_removed_events)}")
    
    if confirm_removed_events:
        print("  confirm_removed 事件详情:")
        for event in confirm_removed_events[:5]:
            print(f"    - step {event.get('step')}: entity_id={event.get('entity_id')}, label={event.get('entity_label')}")
    
    # 查找 WorldModel 相关日志
    world_model_actions = [e for e in trajectory if "world_model" in str(e.get("action", "")).lower() or "entity" in str(e)]
    if world_model_actions:
        print(f"  WorldModel 相关事件数: {len(world_model_actions)}")
    
    print("")
    print("✓ path_trajectory.json 检查完成")
except Exception as e:
    print(f"✗ path_trajectory.json 检查失败: {e}")
    sys.exit(1)
PYEOF
        "$PATH_TRAJ"
    else
        echo "⚠ 未找到 path_trajectory.json: $PATH_TRAJ"
    fi
    
    # 检查日志文件
    LOG_FILE=$(ls -t logs/rag_robot_*.log 2>/dev/null | head -1)
    if [ -n "$LOG_FILE" ]; then
        echo ""
        echo "检查日志文件中的 WorldModel 相关消息..."
        if grep -q "WorldModel" "$LOG_FILE"; then
            echo "  找到 WorldModel 相关日志:"
            grep "WorldModel" "$LOG_FILE" | tail -10 | sed 's/^/    /'
        else
            echo "  ⚠ 未找到 WorldModel 相关日志"
        fi
    fi
fi

echo ""
echo "============================================================"
echo "测试完成"
echo "============================================================"

