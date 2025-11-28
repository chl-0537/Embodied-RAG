#!/bin/bash
# 运行所有 MemoryGraph 相关测试并打印结果摘要

echo "============================================================"
echo "MemoryGraph 完整测试套件"
echo "============================================================"
echo ""

# 1. MemoryGraph 单元测试
echo "1. MemoryGraph 单元测试 (13个测试)"
echo "-----------------------------------"
cd "$(dirname "$0")"
python3 tests/test_memory_graph.py 2>&1 | tail -5
echo ""

# 2. 评估脚本
echo "2. 评估脚本 (3个场景)"
echo "-----------------------------------"
python3 scripts/eval_memory_graph.py 2>&1 | grep -E "(总场景|成功|成功率|平均)" | head -4
echo ""

# 3. 导出快照功能测试
echo "3. 导出快照功能"
echo "-----------------------------------"
python3 -c "
from memory_graph import MemoryGraph
import tempfile
import os
import shutil

db = tempfile.mktemp(suffix='.sqlite')
g = MemoryGraph(db)
ev = {'type': 'vision', 'label': 'test', 'confidence': 0.8}
g.add_evidence(ev)
out_dir = tempfile.mkdtemp()
html = g.export_graph_snapshot(out_dir)
print('✓ 快照导出成功:', html)
assert os.path.exists(html)
assert os.path.exists(os.path.join(out_dir, 'nodes.json'))
assert os.path.exists(os.path.join(out_dir, 'edges.json'))
os.remove(db)
shutil.rmtree(out_dir)
print('✓ 所有快照文件已生成')
"
echo ""

# 4. 集成测试（如果依赖可用）
echo "4. 集成测试"
echo "-----------------------------------"
python3 tests/test_integration_memory_graph.py 2>&1 | tail -10
echo ""

echo "============================================================"
echo "测试完成"
echo "============================================================"

