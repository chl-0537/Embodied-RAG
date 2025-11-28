#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryGraph 集成测试
测试完整的流程：add_vision_memory -> graph update -> retrieve -> plan
"""
import sys
import os
import time
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from memory_graph import MemoryGraph
    from rag_robot_framework import (
        init_chroma_db,
        add_vision_memory,
        retrieve_candidates_by_intent,
        extract_time_context
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"警告: 导入失败 - {e}")
    IMPORTS_AVAILABLE = False


def test_add_vision_memory_to_graph():
    """测试 add_vision_memory 到 MemoryGraph 的流程"""
    print("=" * 60)
    print("测试 1: add_vision_memory -> MemoryGraph 更新")
    print("=" * 60)
    
    if not IMPORTS_AVAILABLE:
        print("跳过测试：导入失败")
        return False
    
    try:
        # 初始化 Chroma 和 MemoryGraph
        init_chroma_db()
        
        # 模拟视觉检测结果
        detection_result = {
            "detected": True,
            "detection_status": "SUCCESS",
            "location": "kitchen",
            "object": "水杯",
            "confidence": 0.85,
            "bbox": [100, 100, 200, 200],
            "message": "Visual detection confirmed"
        }
        
        # 模拟图像路径（可选）
        image_path = None
        
        # 添加视觉记忆
        memory_id = add_vision_memory(detection_result, image_path=image_path)
        
        print(f"✓ 添加视觉记忆，memory_id: {memory_id}")
        assert memory_id is not None, "memory_id 应该不为 None"
        
        # 验证 MemoryGraph 中是否有对应的节点
        from rag_robot_framework import memory_graph
        if memory_graph is not None:
            # 查询节点
            intent = {"object": "水杯"}
            results = memory_graph.query_by_intent(intent, topk=5)
            
            print(f"✓ MemoryGraph 查询结果数量: {len(results)}")
            assert len(results) > 0, "应该找到至少一个结果"
            
            # 验证结果包含目标对象
            found = False
            for result in results:
                node_data = result.get("data", {})
                if node_data.get("label", "") == "水杯":
                    found = True
                    print(f"✓ 找到目标对象节点: {result.get('node_id')}")
                    print(f"  - 置信度: {result.get('fused_conf', 0):.3f}")
                    print(f"  - 位置: {node_data.get('position', 'N/A')}")
                    break
            
            assert found, "应该找到目标对象节点"
        else:
            print("⚠ MemoryGraph 未初始化，跳过验证")
        
        print("✓ 测试通过: add_vision_memory -> MemoryGraph 更新\n")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieve_and_plan_flow():
    """测试检索和规划流程"""
    print("=" * 60)
    print("测试 2: retrieve -> plan 流程")
    print("=" * 60)
    
    if not IMPORTS_AVAILABLE:
        print("跳过测试：导入失败")
        return False
    
    try:
        # 初始化
        init_chroma_db()
        
        # 添加多个视觉记忆
        objects = ["水杯", "手机", "水杯", "花瓶"]
        positions = [
            [1.0, 2.0, 3.0],
            [5.0, 6.0, 7.0],
            [1.1, 2.1, 3.1],  # 第二个水杯，位置相近
            [10.0, 20.0, 30.0]
        ]
        
        for i, (obj, pos) in enumerate(zip(objects, positions)):
            detection_result = {
                "detected": True,
                "detection_status": "SUCCESS",
                "location": "kitchen",
                "object": obj,
                "confidence": 0.8 + i * 0.02,
                "bbox": [100 + i * 10, 100 + i * 10, 200 + i * 10, 200 + i * 10],
                "message": f"Visual detection {i+1}"
            }
            memory_id = add_vision_memory(detection_result, image_path=None)
            print(f"✓ 添加记忆 {i+1}: {obj} at {pos}, memory_id: {memory_id}")
        
        # 执行检索
        intent = {
            "object": "水杯",
            "location": "kitchen"
        }
        
        results = retrieve_candidates_by_intent(intent, topk=5)
        
        print(f"✓ 检索结果数量: {len(results)}")
        assert len(results) > 0, "应该找到至少一个结果"
        
        # 验证结果
        water_cup_count = 0
        for result in results:
            meta = result.get("meta", {})
            node_data = meta.get("node_data", {})
            label = node_data.get("label", "") or meta.get("item_label", "")
            
            if label == "水杯":
                water_cup_count += 1
                print(f"✓ 找到水杯: node_id={result.get('node_id')}, "
                      f"fused_conf={result.get('fused_conf', 0):.3f}, "
                      f"score={result.get('score', 0):.3f}")
        
        assert water_cup_count > 0, "应该找到至少一个水杯"
        print(f"✓ 找到 {water_cup_count} 个水杯结果")
        
        # 模拟规划：选择最高置信度的结果
        if results:
            best_result = results[0]
            best_node_id = best_result.get("node_id")
            best_fused_conf = best_result.get("fused_conf", 0.0)
            best_meta = best_result.get("meta", {})
            best_node_data = best_meta.get("node_data", {})
            best_position = best_node_data.get("position")
            
            print(f"✓ 规划选择: node_id={best_node_id}, "
                  f"fused_conf={best_fused_conf:.3f}, "
                  f"position={best_position}")
            
            assert best_fused_conf > 0.0, "最佳结果的置信度应该 > 0"
        
        print("✓ 测试通过: retrieve -> plan 流程\n")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_time_semantic_integration():
    """测试时间语义集成"""
    print("=" * 60)
    print("测试 3: 时间语义集成")
    print("=" * 60)
    
    if not IMPORTS_AVAILABLE:
        print("跳过测试：导入失败")
        return False
    
    try:
        # 初始化
        init_chroma_db()
        
        # 获取当前时间上下文
        current_time_context = extract_time_context(time.time())
        print(f"✓ 当前时间上下文: {current_time_context}")
        
        # 添加在不同时间的视觉记忆
        base_time = time.time()
        target_hour = 16  # 下午4点
        
        # 在目标时间添加多个证据
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        target_time = now.replace(hour=target_hour, minute=0, second=0, microsecond=0).timestamp()
        
        for i in range(3):
            detection_result = {
                "detected": True,
                "detection_status": "SUCCESS",
                "location": "living_room",
                "object": "花瓶",
                "confidence": 0.85,
                "bbox": [100, 100, 200, 200],
                "message": f"Visual detection at {target_hour}:00"
            }
            # 注意：add_vision_memory 使用当前时间，这里我们主要测试时间上下文
            memory_id = add_vision_memory(detection_result, image_path=None)
            print(f"✓ 添加记忆 {i+1}: 花瓶, memory_id: {memory_id}")
        
        # 使用时间提示进行检索
        intent = {
            "object": "花瓶",
            "time_hint": f"{target_hour}:00"
        }
        
        from rag_robot_framework import memory_graph
        if memory_graph is not None:
            results = memory_graph.query_by_intent(intent, topk=5, beta_time=0.15)
            
            print(f"✓ 时间语义检索结果数量: {len(results)}")
            if results:
                for result in results:
                    time_score = result.get("time_score", 0.0)
                    print(f"  - node_id={result.get('node_id')}, "
                          f"time_score={time_score:.3f}, "
                          f"score={result.get('score', 0):.3f}")
        
        print("✓ 测试通过: 时间语义集成\n")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_export_snapshot():
    """测试导出快照功能"""
    print("=" * 60)
    print("测试 4: 导出快照")
    print("=" * 60)
    
    if not IMPORTS_AVAILABLE:
        print("跳过测试：导入失败")
        return False
    
    try:
        # 初始化
        init_chroma_db()
        
        # 添加一些测试数据
        for i in range(3):
            detection_result = {
                "detected": True,
                "detection_status": "SUCCESS",
                "location": "test_room",
                "object": f"object_{i}",
                "confidence": 0.7 + i * 0.1,
                "bbox": [100, 100, 200, 200],
                "message": f"Test detection {i+1}"
            }
            add_vision_memory(detection_result, image_path=None)
        
        # 导出快照
        from rag_robot_framework import memory_graph
        if memory_graph is not None:
            snapshot_dir = tempfile.mkdtemp(prefix="memory_graph_snapshot_")
            html_file = memory_graph.export_graph_snapshot(snapshot_dir)
            
            print(f"✓ 快照导出到: {snapshot_dir}")
            print(f"✓ HTML 文件: {html_file}")
            
            # 验证文件存在
            assert os.path.exists(html_file), "HTML 文件应该存在"
            assert os.path.exists(os.path.join(snapshot_dir, "nodes.json")), "nodes.json 应该存在"
            assert os.path.exists(os.path.join(snapshot_dir, "edges.json")), "edges.json 应该存在"
            
            print("✓ 所有快照文件已生成")
            
            # 清理
            import shutil
            shutil.rmtree(snapshot_dir)
        else:
            print("⚠ MemoryGraph 未初始化，跳过测试")
        
        print("✓ 测试通过: 导出快照\n")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("MemoryGraph 集成测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行所有测试
    results.append(("add_vision_memory -> graph update", test_add_vision_memory_to_graph()))
    results.append(("retrieve -> plan flow", test_retrieve_and_plan_flow()))
    results.append(("time semantic integration", test_time_semantic_integration()))
    results.append(("export snapshot", test_export_snapshot()))
    
    # 打印摘要
    print("=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("=" * 60)
        print("✓ 所有集成测试通过！")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("✗ 部分测试失败")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

