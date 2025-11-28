#!/bin/bash
# 快速启动脚本

echo "============================================================"
echo "Embodied-RAG 快速启动"
echo "============================================================"
echo ""

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <scene_path> <target_object>"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/scene.glb 花瓶"
    echo "  $0 /path/to/scene.glb watercup"
    echo ""
    echo "提示: 使用短流程测试（40步，4次环扫）以快速验证功能"
    exit 1
fi

SCENE_PATH="$1"
TARGET_OBJECT="$2"

# 检查场景文件
if [ ! -f "$SCENE_PATH" ]; then
    echo "错误: 场景文件不存在: $SCENE_PATH"
    exit 1
fi

echo "配置:"
echo "  场景: $SCENE_PATH"
echo "  目标: $TARGET_OBJECT"
echo "  步数: 40 (快速测试)"
echo "  环扫: 4 次"
echo ""

# 运行
python3 habitat_hm3d_rag_search.py \
    --scene "$SCENE_PATH" \
    --target "$TARGET_OBJECT" \
    --max_steps 40 \
    --rotate_sweep 4 \
    --fusion_mode triangulate

echo ""
echo "============================================================"
echo "运行完成！"
echo "============================================================"
echo ""
echo "验证结果:"
echo "  python3 verify_world_model.py"
echo ""
