#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 WorldModel skeleton 的动态场景功能
"""
import sys
import os
import tempfile
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model import WorldModel


def test_create_entity():
    """测试：创建实体"""
    print("=" * 60)
    print("测试: 创建实体")
    print("=" * 60)
    
    db_path = tempfile.mktemp(suffix=".sqlite")
    wm = WorldModel(db_path=db_path)
    
    # 创建实体
    entity = wm.create_entity(
        label="水杯",
        position=[1.0, 2.0, 3.0],
        confidence=0.8
    )
    
    print(f"✓ 创建的实体 ID: {entity['entity_id']}")
    print(f"✓ 实体标签: {entity['label']}")
    print(f"✓ 实体位置: {entity['position']}")
    print(f"✓ 实体状态: {entity['status']}")
    
    # 验证实体存在
    retrieved = wm.get_entity(entity['entity_id'])
    assert retrieved is not None, "实体应该存在"
    assert retrieved['label'] == "水杯", "实体标签应该匹配"
    assert retrieved['status'] == "active", "实体状态应该是 active"
    
    print("✓ 测试通过: 创建实体\n")
    return True


def test_mark_no_detection_and_evaluate_missing():
    """测试：标记未检测并评估 missing 状态"""
    print("=" * 60)
    print("测试: mark_no_detection 和 evaluate_missing")
    print("=" * 60)
    
    db_path = tempfile.mktemp(suffix=".sqlite")
    wm = WorldModel(db_path=db_path)
    
    # 创建实体
    entity = wm.create_entity(
        label="手机",
        position=[5.0, 6.0, 7.0],
        confidence=0.9
    )
    entity_id = entity['entity_id']
    
    print(f"✓ 创建的实体 ID: {entity_id}")
    print(f"✓ 初始状态: {entity['status']}")
    
    # 标记多次未检测（vis_score 都 >= 0.6）
    n_no_detects = 3
    for i in range(n_no_detects):
        evidence = {
            "ts": time.time() + i,
            "agent_pose": [0.0, 0.0, 0.0],
            "vis_score": 0.7,  # >= 0.6
            "frame_id": f"frame_{i}",
            "depth_stats": {"mean": 2.5}
        }
        wm.mark_no_detection(entity_id, evidence)
        print(f"✓ 标记未检测 {i+1}/{n_no_detects}: vis_score={evidence['vis_score']}")
    
    # 验证 missing_evidence 已记录
    entity_after = wm.get_entity(entity_id)
    assert len(entity_after['missing_evidence']) == n_no_detects, \
        f"应该有 {n_no_detects} 条 missing_evidence，实际: {len(entity_after['missing_evidence'])}"
    print(f"✓ missing_evidence 数量: {len(entity_after['missing_evidence'])}")
    
    # 评估 missing 状态
    result = wm.evaluate_missing(entity_id, n_no_detects=n_no_detects, vis_th=0.6)
    print(f"✓ evaluate_missing 结果: {result}")
    
    # 验证状态变为 missing
    entity_final = wm.get_entity(entity_id)
    print(f"✓ 最终状态: {entity_final['status']}")
    assert entity_final['status'] == "missing", \
        f"实体状态应该是 'missing'，实际: {entity_final['status']}"
    assert entity_final['missing_since'] is not None, "missing_since 应该被设置"
    
    print("✓ 测试通过: mark_no_detection 和 evaluate_missing\n")
    return True


def test_register_movement():
    """测试：注册实体移动"""
    print("=" * 60)
    print("测试: register_movement")
    print("=" * 60)
    
    db_path = tempfile.mktemp(suffix=".sqlite")
    wm = WorldModel(db_path=db_path)
    
    # 创建实体
    entity = wm.create_entity(
        label="椅子",
        position=[1.0, 2.0, 3.0],
        confidence=0.8
    )
    entity_id = entity['entity_id']
    initial_pos = entity['position']
    
    print(f"✓ 初始位置: {initial_pos}")
    
    # 注册移动
    new_pos = [2.0, 3.0, 4.0]
    movement_event = wm.register_movement(
        entity_id,
        new_pos,
        method="triangulate",
        confidence=0.85
    )
    
    assert movement_event is not None, "移动事件应该被创建"
    print(f"✓ 移动事件: {movement_event['type']}")
    print(f"✓ 从位置: {movement_event['from']}")
    print(f"✓ 到位置: {movement_event['to']}")
    print(f"✓ 距离: {movement_event.get('distance', 'N/A')}")
    
    # 验证实体位置已更新
    entity_after = wm.get_entity(entity_id)
    assert entity_after['position'] == new_pos, "实体位置应该已更新"
    assert entity_after['status'] == "moved", "实体状态应该是 'moved'"
    assert len(entity_after['pos_history']) == 2, "位置历史应该有2条记录"
    
    print(f"✓ 更新后位置: {entity_after['position']}")
    print(f"✓ 位置历史长度: {len(entity_after['pos_history'])}")
    print("✓ 测试通过: register_movement\n")
    return True


def test_relink_candidate():
    """测试：重新链接候选实体"""
    print("=" * 60)
    print("测试: relink_candidate")
    print("=" * 60)
    
    db_path = tempfile.mktemp(suffix=".sqlite")
    wm = WorldModel(db_path=db_path)
    
    # 创建实体
    entity = wm.create_entity(
        label="桌子",
        position=[10.0, 20.0, 30.0],
        confidence=0.8
    )
    entity_id = entity['entity_id']
    
    # 测试：接近的观察应该匹配
    obs_close = {
        "label": "桌子",
        "fused_3d_pos": [10.1, 20.1, 30.1]  # 距离约 0.17m < 0.3m
    }
    matched_id = wm.relink_candidate(obs_close, spatial_th=0.3)
    assert matched_id == entity_id, f"应该匹配到 {entity_id}，实际: {matched_id}"
    print(f"✓ 接近的观察匹配成功: {matched_id}")
    
    # 测试：远离的观察不应该匹配
    obs_far = {
        "label": "桌子",
        "fused_3d_pos": [50.0, 60.0, 70.0]  # 距离很远
    }
    matched_id_far = wm.relink_candidate(obs_far, spatial_th=0.3)
    assert matched_id_far is None, "远离的观察不应该匹配"
    print(f"✓ 远离的观察不匹配: {matched_id_far}")
    
    print("✓ 测试通过: relink_candidate\n")
    return True


def test_export_snapshot():
    """测试：导出快照"""
    print("=" * 60)
    print("测试: export_snapshot")
    print("=" * 60)
    
    db_path = tempfile.mktemp(suffix=".sqlite")
    wm = WorldModel(db_path=db_path)
    
    # 创建多个实体
    entity1 = wm.create_entity("物体1", [1.0, 2.0, 3.0], 0.8)
    entity2 = wm.create_entity("物体2", [5.0, 6.0, 7.0], 0.9)
    
    # 创建一些事件
    wm.register_movement(entity1['entity_id'], [1.5, 2.5, 3.5])
    
    # 导出快照
    snapshot_path = tempfile.mktemp(suffix=".json")
    wm.export_snapshot(snapshot_path)
    
    print(f"✓ 快照导出到: {snapshot_path}")
    assert os.path.exists(snapshot_path), "快照文件应该存在"
    
    # 验证快照内容
    import json
    with open(snapshot_path, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)
    
    assert "entities" in snapshot, "快照应该包含 entities"
    assert "events" in snapshot, "快照应该包含 events"
    assert len(snapshot["entities"]) == 2, "应该有2个实体"
    assert len(snapshot["events"]) > 0, "应该有事件"
    
    print(f"✓ 实体数量: {len(snapshot['entities'])}")
    print(f"✓ 事件数量: {len(snapshot['events'])}")
    print("✓ 测试通过: export_snapshot\n")
    
    # 清理
    os.remove(snapshot_path)
    return True


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("WorldModel Skeleton 测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行所有测试
    results.append(("创建实体", test_create_entity()))
    results.append(("mark_no_detection 和 evaluate_missing", test_mark_no_detection_and_evaluate_missing()))
    results.append(("register_movement", test_register_movement()))
    results.append(("relink_candidate", test_relink_candidate()))
    results.append(("export_snapshot", test_export_snapshot()))
    
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
        print("✓ 所有测试通过！")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("✗ 部分测试失败")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

