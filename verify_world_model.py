#!/usr/bin/env python3
"""
WorldModel 集成功能验证脚本

用于检查 WorldModel 数据库和运行日志，验证集成功能是否正常工作。
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional

def check_world_model_db(db_path: str = "vector_db/world_model.sqlite") -> Dict[str, Any]:
    """检查 WorldModel 数据库"""
    results = {
        "db_exists": False,
        "entity_count": 0,
        "entities": [],
        "event_count": 0,
        "events": [],
        "movement_events": [],
        "missing_entities": [],
        "removed_entities": []
    }
    
    if not os.path.exists(db_path):
        return results
    
    results["db_exists"] = True
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 检查 entities
        cursor.execute("SELECT entity_id, label, status, position, confidence FROM entities")
        entities = cursor.fetchall()
        results["entity_count"] = len(entities)
        
        for eid, label, status, pos_json, conf in entities:
            pos = json.loads(pos_json) if pos_json else None
            entity_info = {
                "entity_id": eid,
                "label": label,
                "status": status,
                "position": pos,
                "confidence": conf
            }
            results["entities"].append(entity_info)
            
            if status == "missing":
                results["missing_entities"].append(entity_info)
            elif status == "removed":
                results["removed_entities"].append(entity_info)
        
        # 检查 events
        cursor.execute("""
            SELECT event_id, event_type, entity_id, timestamp, data 
            FROM events 
            ORDER BY timestamp DESC
        """)
        events = cursor.fetchall()
        results["event_count"] = len(events)
        
        for eid, etype, entity_id, ts, data_json in events:
            data = json.loads(data_json) if data_json else {}
            event_info = {
                "event_id": eid,
                "event_type": etype,
                "entity_id": entity_id,
                "timestamp": ts,
                "data": data
            }
            results["events"].append(event_info)
            
            if etype == "movement":
                results["movement_events"].append(event_info)
        
        conn.close()
    except Exception as e:
        results["error"] = str(e)
    
    return results

def check_path_trajectory(traj_path: str) -> Dict[str, Any]:
    """检查 path_trajectory.json"""
    results = {
        "file_exists": False,
        "total_steps": 0,
        "trajectory_size": 0,
        "confirm_removed_events": [],
        "world_model_related": []
    }
    
    if not os.path.exists(traj_path):
        return results
    
    results["file_exists"] = True
    
    try:
        with open(traj_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results["total_steps"] = data.get("total_steps", 0)
        trajectory = data.get("trajectory", [])
        results["trajectory_size"] = len(trajectory)
        
        # 查找 confirm_removed 事件
        for entry in trajectory:
            if entry.get("action") == "confirm_removed":
                results["confirm_removed_events"].append(entry)
            
            # 查找 WorldModel 相关事件
            action = str(entry.get("action", "")).lower()
            if "world_model" in action or "entity" in action or "removed" in action:
                results["world_model_related"].append(entry)
    
    except Exception as e:
        results["error"] = str(e)
    
    return results

def check_log_file(log_path: str) -> Dict[str, Any]:
    """检查日志文件中的 WorldModel 相关消息"""
    results = {
        "file_exists": False,
        "world_model_messages": [],
        "mark_no_detection_count": 0,
        "register_movement_count": 0,
        "confirm_removed_count": 0,
        "evaluate_missing_count": 0
    }
    
    if not os.path.exists(log_path):
        return results
    
    results["file_exists"] = True
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            if "WorldModel" in line:
                results["world_model_messages"].append(line.strip())
                
                if "mark_no_detection" in line.lower() or "标记未检测" in line:
                    results["mark_no_detection_count"] += 1
                if "register_movement" in line.lower() or "注册移动" in line or "检测到实体移动" in line:
                    results["register_movement_count"] += 1
                if "confirm_removed" in line.lower() or "确认移除" in line:
                    results["confirm_removed_count"] += 1
                if "evaluate_missing" in line.lower() or "评估 missing" in line:
                    results["evaluate_missing_count"] += 1
        
        # 只保留最近20条消息
        results["world_model_messages"] = results["world_model_messages"][-20:]
    
    except Exception as e:
        results["error"] = str(e)
    
    return results

def main():
    print("=" * 60)
    print("WorldModel 集成功能验证")
    print("=" * 60)
    print()
    
    # 1. 检查 WorldModel 数据库
    print("1. 检查 WorldModel 数据库...")
    print("-" * 60)
    db_results = check_world_model_db()
    
    if db_results["db_exists"]:
        print(f"✓ 数据库存在")
        print(f"  实体数量: {db_results['entity_count']}")
        print(f"  事件数量: {db_results['event_count']}")
        print(f"  移动事件数: {len(db_results['movement_events'])}")
        print(f"  Missing 实体数: {len(db_results['missing_entities'])}")
        print(f"  Removed 实体数: {len(db_results['removed_entities'])}")
        
        if db_results["entities"]:
            print("\n  实体列表:")
            for ent in db_results["entities"][:5]:
                print(f"    - {ent['entity_id']}: {ent['label']} (status: {ent['status']})")
        
        if db_results["movement_events"]:
            print("\n  移动事件:")
            for evt in db_results["movement_events"][:3]:
                print(f"    - {evt['event_type']} (entity: {evt['entity_id']})")
    else:
        print("⚠ 数据库不存在: vector_db/world_model.sqlite")
    
    print()
    
    # 2. 查找最新的运行日志目录
    run_logs_dir = Path("run_logs")
    if run_logs_dir.exists():
        latest_run = max(run_logs_dir.glob("obs_*"), key=os.path.getmtime, default=None)
        
        if latest_run:
            print(f"2. 检查最新运行日志: {latest_run.name}")
            print("-" * 60)
            
            # 检查 path_trajectory.json
            traj_path = latest_run / "path_trajectory.json"
            if traj_path.exists():
                traj_results = check_path_trajectory(str(traj_path))
                print(f"✓ path_trajectory.json 存在")
                print(f"  总步数: {traj_results['total_steps']}")
                print(f"  轨迹条目数: {traj_results['trajectory_size']}")
                print(f"  confirm_removed 事件数: {len(traj_results['confirm_removed_events'])}")
                
                if traj_results["confirm_removed_events"]:
                    print("\n  confirm_removed 事件:")
                    for evt in traj_results["confirm_removed_events"][:3]:
                        print(f"    - step {evt.get('step')}: entity_id={evt.get('entity_id')}, label={evt.get('entity_label')}")
            else:
                print("⚠ path_trajectory.json 不存在")
    
    print()
    
    # 3. 检查日志文件
    print("3. 检查日志文件...")
    print("-" * 60)
    logs_dir = Path("logs")
    if logs_dir.exists():
        latest_log = max(logs_dir.glob("rag_robot_*.log"), key=os.path.getmtime, default=None)
        
        if latest_log:
            log_results = check_log_file(str(latest_log))
            print(f"✓ 日志文件存在: {latest_log.name}")
            print(f"  WorldModel 消息数: {len(log_results['world_model_messages'])}")
            print(f"  mark_no_detection 调用数: {log_results['mark_no_detection_count']}")
            print(f"  register_movement 调用数: {log_results['register_movement_count']}")
            print(f"  confirm_removed 调用数: {log_results['confirm_removed_count']}")
            print(f"  evaluate_missing 调用数: {log_results['evaluate_missing_count']}")
            
            if log_results["world_model_messages"]:
                print("\n  最近的 WorldModel 消息:")
                for msg in log_results["world_model_messages"][-5:]:
                    print(f"    {msg[:100]}...")
        else:
            print("⚠ 未找到日志文件")
    else:
        print("⚠ logs 目录不存在")
    
    print()
    print("=" * 60)
    print("验证完成")
    print("=" * 60)

if __name__ == "__main__":
    main()

