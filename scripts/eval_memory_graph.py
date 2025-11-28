#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryGraph 评估脚本
模拟多个场景，注入证据，执行检索，输出评估指标
"""
import sys
import os
import json
import time
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_graph import MemoryGraph


class MemoryGraphEvaluator:
    """MemoryGraph 评估器"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        初始化评估器
        
        Args:
            db_path: 数据库路径（如果为 None，使用临时文件）
        """
        if db_path is None:
            self.db_path = tempfile.mktemp(suffix=".sqlite")
            self.temp_db = True
        else:
            self.db_path = db_path
            self.temp_db = False
        
        self.graph = MemoryGraph(self.db_path)
        self.scenarios = []
    
    def add_scenario(self, scenario_name: str, ground_truth: Dict, evidences: List[Dict]):
        """
        添加评估场景
        
        Args:
            scenario_name: 场景名称
            ground_truth: 真实位置 {"object": "水杯", "position": [x, y, z]}
            evidences: 证据列表，每个证据包含 type, label, confidence, position, ts 等
        """
        self.scenarios.append({
            "name": scenario_name,
            "ground_truth": ground_truth,
            "evidences": evidences
        })
    
    def run_scenario(self, scenario: Dict, t_now: float, topk: int = 5) -> Dict:
        """
        运行单个场景
        
        Args:
            scenario: 场景字典
            t_now: 当前时间戳
            topk: 检索 top-k 结果
        
        Returns:
            评估结果字典
        """
        # 清空图（使用新数据库）
        if self.temp_db and os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.graph = MemoryGraph(self.db_path)
        
        # 注入证据
        for evidence in scenario["evidences"]:
            self.graph.add_evidence(evidence)
        
        # 执行检索
        ground_truth = scenario["ground_truth"]
        intent = {
            "object": ground_truth.get("object", ""),
            "time_hint": None  # 可以根据需要添加时间提示
        }
        
        results = self.graph.query_by_intent(intent, topk=topk)
        
        # 计算指标
        metrics = self._compute_metrics(results, ground_truth, topk)
        
        return {
            "scenario_name": scenario["name"],
            "n_evidences": len(scenario["evidences"]),
            "n_results": len(results),
            "metrics": metrics,
            "results": results
        }
    
    def _compute_metrics(self, results: List[Dict], ground_truth: Dict, topk: int) -> Dict:
        """
        计算评估指标
        
        Args:
            results: 检索结果列表
            ground_truth: 真实位置
        
        Returns:
            指标字典
        """
        gt_position = np.array(ground_truth.get("position", [0, 0, 0]))
        gt_object = ground_truth.get("object", "")
        
        # Success@k: 在前 k 个结果中是否找到目标对象
        success_at_k = False
        for i, result in enumerate(results[:topk]):
            node_data = result.get("data", {})
            if node_data.get("label", "") == gt_object:
                success_at_k = True
                break
        
        # 平均定位误差
        localization_errors = []
        for result in results:
            node_data = result.get("data", {})
            if node_data.get("label", "") == gt_object:
                position = node_data.get("position")
                if position:
                    try:
                        pred_position = np.array(position)
                        error = np.linalg.norm(pred_position - gt_position)
                        localization_errors.append(error)
                    except:
                        pass
        
        avg_localization_error = np.mean(localization_errors) if localization_errors else float('inf')
        min_localization_error = np.min(localization_errors) if localization_errors else float('inf')
        
        # 排名（找到目标对象的排名，从1开始）
        rank = None
        for i, result in enumerate(results):
            node_data = result.get("data", {})
            if node_data.get("label", "") == gt_object:
                rank = i + 1
                break
        
        return {
            "success_at_k": success_at_k,
            "rank": rank,
            "avg_localization_error": float(avg_localization_error),
            "min_localization_error": float(min_localization_error),
            "n_matches": len(localization_errors)
        }
    
    def evaluate_all(self, t_now: Optional[float] = None, topk: int = 5) -> Dict:
        """
        评估所有场景
        
        Args:
            t_now: 当前时间戳（如果为 None，使用当前时间）
            topk: 检索 top-k 结果
        
        Returns:
            评估结果字典
        """
        if t_now is None:
            t_now = time.time()
        
        results = []
        for scenario in self.scenarios:
            result = self.run_scenario(scenario, t_now, topk=topk)
            results.append(result)
        
        # 计算总体指标
        total_scenarios = len(results)
        success_count = sum(1 for r in results if r["metrics"]["success_at_k"])
        avg_rank = np.mean([r["metrics"]["rank"] for r in results if r["metrics"]["rank"] is not None])
        avg_error = np.mean([r["metrics"]["avg_localization_error"] for r in results 
                            if r["metrics"]["avg_localization_error"] != float('inf')])
        
        return {
            "total_scenarios": total_scenarios,
            "success_count": success_count,
            "success_rate": success_count / total_scenarios if total_scenarios > 0 else 0.0,
            "avg_rank": float(avg_rank) if not np.isnan(avg_rank) else None,
            "avg_localization_error": float(avg_error) if not np.isnan(avg_error) else None,
            "scenario_results": results
        }
    
    def cleanup(self):
        """清理临时文件"""
        if self.temp_db and os.path.exists(self.db_path):
            os.remove(self.db_path)


def create_test_scenarios() -> List[Dict]:
    """创建测试场景"""
    scenarios = []
    
    # 场景1：单个对象，多个证据
    base_time = time.time()
    gt_pos1 = [1.0, 2.0, 3.0]
    evidences1 = []
    for i in range(5):
        evidence = {
            "type": "vision",
            "label": "水杯",
            "confidence": 0.8 + i * 0.02,
            "position": [gt_pos1[0] + np.random.randn() * 0.1,
                        gt_pos1[1] + np.random.randn() * 0.1,
                        gt_pos1[2] + np.random.randn() * 0.1],
            "embedding": np.random.rand(128).tolist(),
            "ts": base_time - (5 - i) * 3600
        }
        evidences1.append(evidence)
    
    scenarios.append({
        "name": "场景1：单个对象，多个证据",
        "ground_truth": {"object": "水杯", "position": gt_pos1},
        "evidences": evidences1
    })
    
    # 场景2：多个对象，混合证据
    gt_pos2 = [5.0, 6.0, 7.0]
    evidences2 = []
    for i in range(3):
        evidence = {
            "type": "vision",
            "label": "手机",
            "confidence": 0.7 + i * 0.05,
            "position": [gt_pos2[0] + np.random.randn() * 0.15,
                        gt_pos2[1] + np.random.randn() * 0.15,
                        gt_pos2[2] + np.random.randn() * 0.15],
            "embedding": np.random.rand(128).tolist(),
            "ts": base_time - (3 - i) * 3600
        }
        evidences2.append(evidence)
    
    # 添加干扰对象
    for i in range(2):
        evidence = {
            "type": "vision",
            "label": "水杯",
            "confidence": 0.6,
            "position": [10.0, 10.0, 10.0],
            "embedding": np.random.rand(128).tolist(),
            "ts": base_time - i * 3600
        }
        evidences2.append(evidence)
    
    scenarios.append({
        "name": "场景2：多个对象，混合证据",
        "ground_truth": {"object": "手机", "position": gt_pos2},
        "evidences": evidences2
    })
    
    # 场景3：时间语义场景（在特定时间有更多证据）
    gt_pos3 = [10.0, 20.0, 30.0]
    evidences3 = []
    # 在 16:00 添加多个证据
    target_hour = 16
    now = datetime.now(timezone.utc)
    target_time = now.replace(hour=target_hour, minute=0, second=0, microsecond=0).timestamp()
    
    for i in range(5):
        evidence = {
            "type": "vision",
            "label": "花瓶",
            "confidence": 0.85,
            "position": [gt_pos3[0] + np.random.randn() * 0.1,
                        gt_pos3[1] + np.random.randn() * 0.1,
                        gt_pos3[2] + np.random.randn() * 0.1],
            "embedding": np.random.rand(128).tolist(),
            "ts": target_time + i * 3600  # 都在 16:00 附近
        }
        evidences3.append(evidence)
    
    scenarios.append({
        "name": "场景3：时间语义场景",
        "ground_truth": {"object": "花瓶", "position": gt_pos3},
        "evidences": evidences3
    })
    
    return scenarios


def main():
    """主函数"""
    print("=" * 60)
    print("MemoryGraph 评估脚本")
    print("=" * 60)
    
    evaluator = MemoryGraphEvaluator()
    
    # 创建测试场景
    scenarios = create_test_scenarios()
    for scenario in scenarios:
        evaluator.add_scenario(
            scenario["name"],
            scenario["ground_truth"],
            scenario["evidences"]
        )
    
    # 运行评估
    print(f"\n运行 {len(scenarios)} 个场景...")
    results = evaluator.evaluate_all(topk=5)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果摘要")
    print("=" * 60)
    print(f"总场景数: {results['total_scenarios']}")
    print(f"成功场景数: {results['success_count']}")
    print(f"成功率: {results['success_rate']:.2%}")
    if results['avg_rank']:
        print(f"平均排名: {results['avg_rank']:.2f}")
    if results['avg_localization_error']:
        print(f"平均定位误差: {results['avg_localization_error']:.3f} m")
    
    print("\n" + "=" * 60)
    print("详细场景结果")
    print("=" * 60)
    for scenario_result in results['scenario_results']:
        print(f"\n场景: {scenario_result['scenario_name']}")
        print(f"  证据数量: {scenario_result['n_evidences']}")
        print(f"  检索结果数: {scenario_result['n_results']}")
        metrics = scenario_result['metrics']
        print(f"  Success@5: {metrics['success_at_k']}")
        if metrics['rank']:
            print(f"  排名: {metrics['rank']}")
        if metrics['avg_localization_error'] != float('inf'):
            print(f"  平均定位误差: {metrics['avg_localization_error']:.3f} m")
            print(f"  最小定位误差: {metrics['min_localization_error']:.3f} m")
        print(f"  匹配数量: {metrics['n_matches']}")
    
    # 清理
    evaluator.cleanup()
    
    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

