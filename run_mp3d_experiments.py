#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MP3D数据集批量实验脚本
计算指标：Success Rate (SR), SPL, Localization Error (LE), Recall@K, Memory Precision (MP)
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

# 导入主搜索函数
import habitat_hm3d_rag_search as habitat_search
import rag_robot_framework as rag


# 实验配置
SCENES = [
    "17DRP5sb8fy",
    "1LXtFkjw3qL",
    "5q7pvUzZiYa",
    "2n8kARJN3HM",
    "pLe4wQe7qrG"
]

TARGET_OBJECTS = [
    "杯子",
    "椅子",
    "书本",
    "桌子",
    "床"
]

# 目标对象的中英文映射（用于检索评估）
OBJECT_MAPPING = {
    "杯子": ["cup", "mug", "glass", "watercup", "杯子", "水杯"],
    "椅子": ["chair", "seat", "stool", "armchair", "椅子"],
    "书本": ["book", "books", "novel", "textbook", "书本", "书"],
    "桌子": ["table", "desk", "dining table", "桌子"],
    "床": ["bed", "mattress", "床", "大床"]
}

# MP3D数据集路径
MP3D_BASE_PATH = "/home/szu/JH/data/mp3d"


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, output_dir: str = "experiment_results", dataset_config: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_config = dataset_config
        self.results = []
        
        # 创建实验日志目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_log_dir = self.output_dir / f"experiment_{timestamp}"
        self.experiment_log_dir.mkdir(parents=True, exist_ok=True)
        
        rag.log_info(f"实验结果将保存到: {self.experiment_log_dir}")
    
    def get_scene_path(self, scene_id: str) -> str:
        """获取场景文件路径"""
        scene_path = os.path.join(MP3D_BASE_PATH, scene_id, f"{scene_id}.glb")
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"场景文件不存在: {scene_path}")
        return scene_path
    
    def compute_path_length(self, trajectory: List[Dict]) -> float:
        """计算路径长度（米）"""
        if len(trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(trajectory)):
            pos1 = trajectory[i-1].get("position")
            pos2 = trajectory[i].get("position")
            if pos1 and pos2:
                try:
                    p1 = np.array(pos1)
                    p2 = np.array(pos2)
                    dist = np.linalg.norm(p2 - p1)
                    total_length += dist
                except:
                    pass
        
        return total_length
    
    def compute_shortest_path_length(self, start_pos: List[float], end_pos: List[float]) -> float:
        """
        计算最短路径长度（简化：使用欧氏距离）
        注意：实际应该使用A*等路径规划算法，这里使用欧氏距离作为下界
        """
        try:
            start = np.array(start_pos)
            end = np.array(end_pos)
            return float(np.linalg.norm(end - start))
        except:
            return 0.0
    
    def compute_spl(self, success: bool, path_length: float, shortest_path_length: float) -> float:
        """
        计算 SPL (Success weighted by Path Length)
        SPL = (1/N) * Σ (Si * min(Li, Li*) / max(Li, Li*))
        其中 Si = 1 如果成功，否则为 0
        """
        if not success:
            return 0.0
        
        if shortest_path_length <= 0:
            return 0.0
        
        if path_length <= 0:
            return 1.0
        
        # SPL = Si * min(Li, Li*) / max(Li, Li*)
        min_len = min(path_length, shortest_path_length)
        max_len = max(path_length, shortest_path_length)
        
        if max_len == 0:
            return 1.0
        
        return min_len / max_len
    
    def compute_localization_error(self, predicted_pos: Optional[List[float]], 
                                   ground_truth_pos: Optional[List[float]]) -> float:
        """
        计算定位误差（欧氏距离）
        LE = ||p_pred - p_gt||_2
        """
        if predicted_pos is None or ground_truth_pos is None:
            return float('inf')
        
        try:
            pred = np.array(predicted_pos)
            gt = np.array(ground_truth_pos)
            error = np.linalg.norm(pred - gt)
            return float(error)
        except:
            return float('inf')
    
    def evaluate_retrieval_recall_at_k(self, intent: Dict, retrieved_candidates: List[Dict], 
                                       target_object: str, k: int = 5) -> float:
        """
        计算 Recall@K
        评估检索模块是否能找到相关记忆
        
        Args:
            intent: 意图字典
            retrieved_candidates: 检索到的候选列表
            target_object: 目标对象名称
            k: K值（默认5）
        
        Returns:
            Recall@K 分数 (0.0-1.0)
        """
        if len(retrieved_candidates) == 0:
            return 0.0
        
        # 获取目标对象的同义词
        target_synonyms = OBJECT_MAPPING.get(target_object, [target_object.lower()])
        target_synonyms = [s.lower() for s in target_synonyms]
        
        # 检查前k个结果中是否有相关记忆
        relevant_count = 0
        for i, candidate in enumerate(retrieved_candidates[:k]):
            # 从候选数据中提取标签
            node_data = candidate.get("data", {})
            label = node_data.get("label", "").lower()
            
            # 检查是否匹配目标对象
            is_relevant = any(syn in label for syn in target_synonyms)
            if is_relevant:
                relevant_count += 1
        
        # Recall@K = 相关记忆数 / min(k, 总相关记忆数)
        # 简化：如果前k个中有至少1个相关，则返回1.0，否则返回0.0
        return 1.0 if relevant_count > 0 else 0.0
    
    def evaluate_memory_precision(self, intent: Dict, retrieved_candidates: List[Dict], 
                                  target_object: str, k: int = 5) -> float:
        """
        计算 Memory Precision (MP)
        评估检索到的记忆中有多少是相关的
        
        Args:
            intent: 意图字典
            retrieved_candidates: 检索到的候选列表
            target_object: 目标对象名称
            k: K值（默认5）
        
        Returns:
            Memory Precision 分数 (0.0-1.0)
        """
        if len(retrieved_candidates) == 0:
            return 0.0
        
        # 获取目标对象的同义词
        target_synonyms = OBJECT_MAPPING.get(target_object, [target_object.lower()])
        target_synonyms = [s.lower() for s in target_synonyms]
        
        # 检查前k个结果中有多少是相关的
        relevant_count = 0
        total_count = min(k, len(retrieved_candidates))
        
        for candidate in retrieved_candidates[:k]:
            # 从候选数据中提取标签
            node_data = candidate.get("data", {})
            label = node_data.get("label", "").lower()
            
            # 检查是否匹配目标对象
            is_relevant = any(syn in label for syn in target_synonyms)
            if is_relevant:
                relevant_count += 1
        
        # MP = 相关记忆数 / 检索到的记忆总数
        if total_count == 0:
            return 0.0
        
        return relevant_count / total_count
    
    def get_ground_truth_position(self, scene_id: str, target_object: str) -> Optional[List[float]]:
        """
        获取ground truth位置
        注意：MP3D数据集可能不直接提供GT位置，这里尝试从场景数据中获取
        如果无法获取，返回None（将使用估计位置作为参考）
        """
        # 尝试从场景的语义标注中获取（如果有）
        # 这里简化处理，实际应该从场景的语义标注文件中读取
        # 如果无法获取，返回None，实验中将使用估计位置作为参考
        
        # TODO: 如果场景数据中有GT位置，可以从这里读取
        # 例如：从 .house 文件或语义标注文件中读取
        
        return None
    
    def run_single_experiment(self, scene_id: str, target_object: str, 
                             max_steps: int = 1000, rotate_sweep: int = 12,
                             explore_policy: str = "frontier", fusion_mode: str = "triangulate") -> Dict:
        """
        运行单个实验
        
        Returns:
            实验结果字典
        """
        rag.log_info(f"开始实验: scene={scene_id}, target={target_object}")
        
        # 获取场景路径
        scene_path = self.get_scene_path(scene_id)
        
        # 创建实验保存目录
        exp_save_dir = self.experiment_log_dir / f"{scene_id}_{target_object}"
        exp_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 记录开始时间
        start_time = time.time()
        
        # 获取ground truth位置（如果可用）
        gt_position = self.get_ground_truth_position(scene_id, target_object)
        
        # 运行搜索
        try:
            result = habitat_search.habitat_rag_search(
                scene_path=scene_path,
                target_object=target_object,
                max_steps=max_steps,
                rotate_sweep=rotate_sweep,
                save_obs_dir=str(exp_save_dir),
                explore_policy=explore_policy,
                fusion_mode=fusion_mode,
                dataset_config=self.dataset_config
            )
        except Exception as e:
            rag.log_error(f"实验失败: scene={scene_id}, target={target_object}", e)
            result = {
                "status": "ERROR",
                "found": False,
                "error": str(e)
            }
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        # 读取路径轨迹（如果存在）
        path_length = 0.0
        shortest_path_length = 0.0
        trajectory = []
        
        path_trajectory_file = exp_save_dir / "path_trajectory.json"
        if path_trajectory_file.exists():
            try:
                with open(path_trajectory_file, "r", encoding="utf-8") as f:
                    path_data = json.load(f)
                    trajectory = path_data.get("trajectory", [])
                    path_length = self.compute_path_length(trajectory)
                    
                    # 计算最短路径长度
                    if len(trajectory) > 0:
                        start_pos = trajectory[0].get("position")
                        end_pos = trajectory[-1].get("position")
                        if start_pos and end_pos:
                            shortest_path_length = self.compute_shortest_path_length(start_pos, end_pos)
            except Exception as e:
                rag.log_warning(f"读取路径轨迹失败: {e}")
        
        # 计算指标
        success = result.get("found", False)
        estimated_pos = result.get("estimated_3d_position")
        
        # 1. Success Rate (SR)
        sr = 1.0 if success else 0.0
        
        # 2. SPL
        spl = self.compute_spl(success, path_length, shortest_path_length)
        
        # 3. Localization Error (LE)
        # 如果没有GT位置，使用估计位置作为参考（LE=0）
        le = self.compute_localization_error(estimated_pos, gt_position)
        if gt_position is None:
            # 如果没有GT，LE无法计算，标记为None
            le = None
        
        # 4. Recall@K 和 Memory Precision
        # 需要从检索结果中计算
        recall_at_k = 0.0
        memory_precision = 0.0
        
        # 尝试从MemoryGraph中获取检索结果
        if rag.memory_graph:
            try:
                # 构建意图
                intent = {
                    "object": target_object,
                    "text": f"找到{target_object}"
                }
                
                # 检索候选
                retrieved_candidates = rag.memory_graph.query_by_intent(
                    intent, topk=10
                )
                
                # 计算 Recall@K (K=5)
                recall_at_k = self.evaluate_retrieval_recall_at_k(
                    intent, retrieved_candidates, target_object, k=5
                )
                
                # 计算 Memory Precision
                memory_precision = self.evaluate_memory_precision(
                    intent, retrieved_candidates, target_object, k=5
                )
            except Exception as e:
                rag.log_warning(f"计算检索指标失败: {e}")
        
        # 构建实验结果
        experiment_result = {
            "scene_id": scene_id,
            "target_object": target_object,
            "success": success,
            "status": result.get("status", "UNKNOWN"),
            "steps": result.get("steps", 0),
            "duration_seconds": duration,
            "path_length": path_length,
            "shortest_path_length": shortest_path_length,
            "estimated_3d_position": estimated_pos,
            "ground_truth_position": gt_position,
            "metrics": {
                "SR": sr,  # Success Rate
                "SPL": spl,  # Success weighted by Path Length
                "LE": le,  # Localization Error (None if no GT)
                "Recall@5": recall_at_k,
                "Memory_Precision": memory_precision
            },
            "save_dir": str(exp_save_dir)
        }
        
        # 保存单个实验结果
        result_file = exp_save_dir / "experiment_result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(experiment_result, f, ensure_ascii=False, indent=2)
        
        rag.log_info(f"实验完成: scene={scene_id}, target={target_object}, success={success}, SR={sr:.3f}, SPL={spl:.3f}")
        
        return experiment_result
    
    def run_all_experiments(self, max_steps: int = 1000, rotate_sweep: int = 12,
                           explore_policy: str = "frontier", fusion_mode: str = "triangulate",
                           skip_existing: bool = False):
        """
        运行所有实验
        
        Args:
            skip_existing: 如果为True，跳过已存在的实验结果
        """
        total_experiments = len(SCENES) * len(TARGET_OBJECTS)
        current = 0
        
        rag.log_info(f"开始批量实验: {total_experiments} 个实验任务")
        
        for scene_id in SCENES:
            for target_object in TARGET_OBJECTS:
                current += 1
                
                # 检查是否已存在
                if skip_existing:
                    exp_save_dir = self.experiment_log_dir / f"{scene_id}_{target_object}"
                    result_file = exp_save_dir / "experiment_result.json"
                    if result_file.exists():
                        rag.log_info(f"跳过已存在的实验: {scene_id}_{target_object} ({current}/{total_experiments})")
                        try:
                            with open(result_file, "r", encoding="utf-8") as f:
                                result = json.load(f)
                                self.results.append(result)
                        except:
                            pass
                        continue
                
                rag.log_info(f"实验进度: {current}/{total_experiments}")
                
                # 运行实验
                try:
                    result = self.run_single_experiment(
                        scene_id=scene_id,
                        target_object=target_object,
                        max_steps=max_steps,
                        rotate_sweep=rotate_sweep,
                        explore_policy=explore_policy,
                        fusion_mode=fusion_mode
                    )
                    self.results.append(result)
                except Exception as e:
                    rag.log_error(f"实验异常: scene={scene_id}, target={target_object}", e)
                    # 添加失败记录
                    self.results.append({
                        "scene_id": scene_id,
                        "target_object": target_object,
                        "success": False,
                        "status": "ERROR",
                        "error": str(e),
                        "metrics": {
                            "SR": 0.0,
                            "SPL": 0.0,
                            "LE": None,
                            "Recall@5": 0.0,
                            "Memory_Precision": 0.0
                        }
                    })
        
        # 保存汇总结果
        self.save_summary()
    
    def save_summary(self):
        """保存实验结果汇总"""
        if len(self.results) == 0:
            rag.log_warning("没有实验结果可汇总")
            return
        
        # 计算总体指标
        total_experiments = len(self.results)
        successful_experiments = sum(1 for r in self.results if r.get("success", False))
        
        # 计算平均指标
        avg_sr = np.mean([r["metrics"]["SR"] for r in self.results])
        avg_spl = np.mean([r["metrics"]["SPL"] for r in self.results])
        
        # LE（只计算有GT的情况）
        le_values = [r["metrics"]["LE"] for r in self.results if r["metrics"]["LE"] is not None]
        avg_le = np.mean(le_values) if le_values else None
        
        avg_recall_at_5 = np.mean([r["metrics"]["Recall@5"] for r in self.results])
        avg_memory_precision = np.mean([r["metrics"]["Memory_Precision"] for r in self.results])
        
        # 按场景和对象分组统计
        scene_stats = {}
        object_stats = {}
        
        for result in self.results:
            scene_id = result["scene_id"]
            target_object = result["target_object"]
            
            # 按场景统计
            if scene_id not in scene_stats:
                scene_stats[scene_id] = {
                    "total": 0,
                    "success": 0,
                    "sr": [],
                    "spl": [],
                    "le": [],
                    "recall": [],
                    "precision": []
                }
            
            scene_stats[scene_id]["total"] += 1
            if result.get("success", False):
                scene_stats[scene_id]["success"] += 1
            scene_stats[scene_id]["sr"].append(result["metrics"]["SR"])
            scene_stats[scene_id]["spl"].append(result["metrics"]["SPL"])
            if result["metrics"]["LE"] is not None:
                scene_stats[scene_id]["le"].append(result["metrics"]["LE"])
            scene_stats[scene_id]["recall"].append(result["metrics"]["Recall@5"])
            scene_stats[scene_id]["precision"].append(result["metrics"]["Memory_Precision"])
            
            # 按对象统计
            if target_object not in object_stats:
                object_stats[target_object] = {
                    "total": 0,
                    "success": 0,
                    "sr": [],
                    "spl": [],
                    "le": [],
                    "recall": [],
                    "precision": []
                }
            
            object_stats[target_object]["total"] += 1
            if result.get("success", False):
                object_stats[target_object]["success"] += 1
            object_stats[target_object]["sr"].append(result["metrics"]["SR"])
            object_stats[target_object]["spl"].append(result["metrics"]["SPL"])
            if result["metrics"]["LE"] is not None:
                object_stats[target_object]["le"].append(result["metrics"]["LE"])
            object_stats[target_object]["recall"].append(result["metrics"]["Recall@5"])
            object_stats[target_object]["precision"].append(result["metrics"]["Memory_Precision"])
        
        # 计算分组平均值
        for scene_id in scene_stats:
            stats = scene_stats[scene_id]
            stats["avg_sr"] = np.mean(stats["sr"])
            stats["avg_spl"] = np.mean(stats["spl"])
            stats["avg_le"] = np.mean(stats["le"]) if stats["le"] else None
            stats["avg_recall"] = np.mean(stats["recall"])
            stats["avg_precision"] = np.mean(stats["precision"])
            stats["success_rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
        
        for target_object in object_stats:
            stats = object_stats[target_object]
            stats["avg_sr"] = np.mean(stats["sr"])
            stats["avg_spl"] = np.mean(stats["spl"])
            stats["avg_le"] = np.mean(stats["le"]) if stats["le"] else None
            stats["avg_recall"] = np.mean(stats["recall"])
            stats["avg_precision"] = np.mean(stats["precision"])
            stats["success_rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
        
        # 构建汇总报告
        summary = {
            "experiment_info": {
                "total_experiments": total_experiments,
                "successful_experiments": successful_experiments,
                "overall_success_rate": successful_experiments / total_experiments if total_experiments > 0 else 0.0,
                "timestamp": datetime.now().isoformat()
            },
            "overall_metrics": {
                "SR": float(avg_sr),
                "SPL": float(avg_spl),
                "LE": float(avg_le) if avg_le is not None else None,
                "Recall@5": float(avg_recall_at_5),
                "Memory_Precision": float(avg_memory_precision)
            },
            "by_scene": scene_stats,
            "by_object": object_stats,
            "detailed_results": self.results
        }
        
        # 保存汇总报告
        summary_file = self.experiment_log_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 打印汇总报告
        print("\n" + "=" * 80)
        print("实验汇总报告")
        print("=" * 80)
        print(f"总实验数: {total_experiments}")
        print(f"成功实验数: {successful_experiments}")
        print(f"总体成功率: {successful_experiments / total_experiments * 100:.2f}%")
        print("\n总体指标:")
        print(f"  Success Rate (SR): {avg_sr:.4f}")
        print(f"  SPL: {avg_spl:.4f}")
        le_str = f"{avg_le:.4f}" if avg_le is not None else "N/A"
        print(f"  Localization Error (LE): {le_str}")
        print(f"  Recall@5: {avg_recall_at_5:.4f}")
        print(f"  Memory Precision: {avg_memory_precision:.4f}")
        
        print("\n按场景统计:")
        for scene_id, stats in sorted(scene_stats.items()):
            print(f"  {scene_id}:")
            print(f"    成功率: {stats['success_rate']*100:.2f}% ({stats['success']}/{stats['total']})")
            le_str = f"{stats['avg_le']:.4f}" if stats['avg_le'] else 'N/A'
            print(f"    SR: {stats['avg_sr']:.4f}, SPL: {stats['avg_spl']:.4f}, "
                  f"LE: {le_str}, "
                  f"Recall@5: {stats['avg_recall']:.4f}, MP: {stats['avg_precision']:.4f}")
        
        print("\n按对象统计:")
        for target_object, stats in sorted(object_stats.items()):
            print(f"  {target_object}:")
            print(f"    成功率: {stats['success_rate']*100:.2f}% ({stats['success']}/{stats['total']})")
            le_str = f"{stats['avg_le']:.4f}" if stats['avg_le'] else 'N/A'
            print(f"    SR: {stats['avg_sr']:.4f}, SPL: {stats['avg_spl']:.4f}, "
                  f"LE: {le_str}, "
                  f"Recall@5: {stats['avg_recall']:.4f}, MP: {stats['avg_precision']:.4f}")
        
        print("\n" + "=" * 80)
        print(f"详细结果已保存到: {summary_file}")
        print("=" * 80 + "\n")
        
        rag.log_info(f"实验汇总已保存: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="MP3D数据集批量实验")
    parser.add_argument("--output_dir", type=str, default="experiment_results",
                       help="实验结果输出目录")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="最大探索步数")
    parser.add_argument("--rotate_sweep", type=int, default=12,
                       help="初始环扫旋转次数")
    parser.add_argument("--explore_policy", type=str, default="frontier",
                       choices=["simple", "frontier"],
                       help="探索策略")
    parser.add_argument("--fusion_mode", type=str, default="triangulate",
                       choices=["triangulate", "depth", "voxel_vote"],
                       help="融合模式")
    parser.add_argument("--dataset_config", type=str, default=None,
                       help="Habitat场景数据集配置文件路径")
    parser.add_argument("--skip_existing", action="store_true",
                       help="跳过已存在的实验结果")
    parser.add_argument("--scene", type=str, default=None,
                       help="指定单个场景（用于调试）")
    parser.add_argument("--target", type=str, default=None,
                       help="指定单个目标对象（用于调试）")
    
    args = parser.parse_args()
    
    # 初始化实验运行器
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        dataset_config=args.dataset_config
    )
    
    # 如果指定了单个场景和目标，只运行该实验
    if args.scene and args.target:
        rag.log_info(f"运行单个实验: scene={args.scene}, target={args.target}")
        result = runner.run_single_experiment(
            scene_id=args.scene,
            target_object=args.target,
            max_steps=args.max_steps,
            rotate_sweep=args.rotate_sweep,
            explore_policy=args.explore_policy,
            fusion_mode=args.fusion_mode
        )
        runner.results.append(result)
        runner.save_summary()
    else:
        # 运行所有实验
        runner.run_all_experiments(
            max_steps=args.max_steps,
            rotate_sweep=args.rotate_sweep,
            explore_policy=args.explore_policy,
            fusion_mode=args.fusion_mode,
            skip_existing=args.skip_existing
        )


if __name__ == "__main__":
    main()

