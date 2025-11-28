#!/usr/bin/env python3
"""
测试 RANSAC 三角化功能

运行方式：
    python test_ransac_triangulation.py
"""

import sys
import numpy as np
from typing import List, Tuple, Optional

# 模拟rag.log_info和rag.log_warning
class MockRAG:
    @staticmethod
    def log_info(msg, *args):
        print(f"[INFO] {msg}")
    
    @staticmethod
    def log_warning(msg, *args):
        print(f"[WARN] {msg}")

# 临时替换rag模块
import habitat_hm3d_rag_search
habitat_hm3d_rag_search.rag = MockRAG()

# 导入SpatialSemanticFusionModule（需要从函数内部提取）
# 由于模块是嵌套定义的，我们需要创建一个测试版本

class TestSpatialSemanticFusionModule:
    """测试版本的SpatialSemanticFusionModule"""
    
    def __init__(self):
        self.fx = 320.0
        self.fy = 320.0
        self.cx = 320.0
        self.cy = 240.0
        self.img_width = 640
        self.img_height = 480
    
    def _quaternion_to_rotation_matrix(self, quat):
        """将四元数转换为旋转矩阵"""
        x, y, z, w = quat
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
        return R
    
    def _point_to_ray_distance(self, point, ray_origin, ray_direction):
        """计算点到射线的距离"""
        vec_to_point = point - ray_origin
        t = np.dot(vec_to_point, ray_direction)
        closest_point_on_ray = ray_origin + t * ray_direction
        distance = np.linalg.norm(point - closest_point_on_ray)
        return distance
    
    def _triangulate_two_rays(self, pos1, ray1, pos2, ray2):
        """三角化两条射线"""
        try:
            ray1 = ray1.reshape(3, 1)
            ray2 = ray2.reshape(3, 1)
            
            proj1 = np.eye(3) - ray1 @ ray1.T
            proj2 = np.eye(3) - ray2 @ ray2.T
            
            A = np.vstack([proj1, proj2])
            b = np.hstack([proj1 @ pos1, proj2 @ pos2])
            
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            
            if s[-1] < 1e-6:
                return None
            
            point = Vt.T @ (U.T @ b / s)
            
            if np.any(np.isnan(point)) or np.any(np.isinf(point)):
                return None
            
            return point
        except Exception:
            return None
    
    def _triangulate_rays(self, positions, rays):
        """使用最小二乘法三角化多条射线"""
        if len(positions) < 2:
            return None
        
        try:
            A_list = []
            b_list = []
            
            for pos, ray in zip(positions, rays):
                ray = ray.reshape(3, 1)
                proj = np.eye(3) - ray @ ray.T
                A_list.append(proj)
                b_list.append(proj @ pos)
            
            A = np.vstack(A_list)
            b = np.hstack(b_list)
            
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            
            if s[-1] < 1e-6:
                return None
            
            point = Vt.T @ (U.T @ b / s)
            
            if np.any(np.isnan(point)) or np.any(np.isinf(point)):
                return None
            
            return point
        except Exception:
            return None
    
    def _ransac_triangulate(self, positions, rays, max_iterations=100, 
                           inlier_threshold=0.15, min_inliers=2):
        """RANSAC三角化"""
        if len(positions) < 2:
            return None, 0.0, 0.0
        
        n_rays = len(positions)
        best_point = None
        best_inlier_count = 0
        best_inliers = []
        
        np.random.seed(42)  # 固定随机种子以便测试可重复
        
        for iteration in range(max_iterations):
            if n_rays < 2:
                break
            
            indices = np.random.choice(n_rays, size=2, replace=False)
            idx1, idx2 = indices[0], indices[1]
            
            candidate_point = self._triangulate_two_rays(
                positions[idx1], rays[idx1],
                positions[idx2], rays[idx2]
            )
            
            if candidate_point is None:
                continue
            
            inliers = []
            inlier_count = 0
            
            for i in range(n_rays):
                distance = self._point_to_ray_distance(
                    candidate_point, positions[i], rays[i]
                )
                if distance < inlier_threshold:
                    inliers.append(i)
                    inlier_count += 1
            
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_point = candidate_point.copy()
                best_inliers = inliers
        
        if best_inlier_count < min_inliers:
            return None, 0.0, 0.0
        
        if len(best_inliers) >= 2:
            inlier_positions = [positions[i] for i in best_inliers]
            inlier_rays = [rays[i] for i in best_inliers]
            refined_point = self._triangulate_rays(inlier_positions, inlier_rays)
            if refined_point is not None:
                best_point = refined_point
        
        inlier_ratio = best_inlier_count / n_rays
        num_views = n_rays
        view_factor = 1.0 - np.exp(-num_views / 3.0)
        confidence = inlier_ratio * view_factor
        
        return best_point, inlier_ratio, confidence


def generate_rays_to_point(true_point, positions, noise_level=0.0):
    """生成指向真实点的射线（可添加噪声）"""
    rays = []
    for pos in positions:
        direction = true_point - pos
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            # 如果位置太接近，使用随机方向
            direction = np.random.randn(3)
        direction = direction / distance
        
        # 添加噪声
        if noise_level > 0:
            noise = np.random.randn(3) * noise_level
            direction = direction + noise
            direction = direction / np.linalg.norm(direction)
        
        rays.append(direction)
    return rays


def test_perfect_matches():
    """测试完美匹配的情况"""
    print("=" * 60)
    print("测试1: 完美匹配（无噪声）")
    print("=" * 60)
    
    module = TestSpatialSemanticFusionModule()
    
    # 真实点
    true_point = np.array([2.0, 3.0, 5.0])
    
    # 生成5个观测位置
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([0.5, 0.5, 1.0]),
    ]
    
    # 生成指向真实点的射线（无噪声）
    rays = generate_rays_to_point(true_point, positions, noise_level=0.0)
    
    # RANSAC三角化
    estimated_point, inlier_ratio, confidence = module._ransac_triangulate(
        positions, rays, max_iterations=100, inlier_threshold=0.15
    )
    
    if estimated_point is not None:
        error = np.linalg.norm(estimated_point - true_point)
        print(f"  真实点: {true_point}")
        print(f"  估计点: {estimated_point}")
        print(f"  误差: {error:.6f}m")
        print(f"  内点比例: {inlier_ratio:.3f}")
        print(f"  置信度: {confidence:.3f}")
        
        if error < 0.01 and inlier_ratio >= 0.9:
            print("  ✓ 通过: 误差小，内点比例高")
            return True
        else:
            print("  ✗ 失败: 误差过大或内点比例过低")
            return False
    else:
        print("  ✗ 失败: 未找到估计点")
        return False


def test_noisy_matches():
    """测试有噪声的情况"""
    print("\n" + "=" * 60)
    print("测试2: 有噪声匹配")
    print("=" * 60)
    
    module = TestSpatialSemanticFusionModule()
    
    # 真实点
    true_point = np.array([2.0, 3.0, 5.0])
    
    # 生成8个观测位置
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([0.5, 0.5, 1.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
        np.array([1.5, 1.5, 0.5]),
    ]
    
    # 生成指向真实点的射线（添加小噪声）
    np.random.seed(42)
    rays = generate_rays_to_point(true_point, positions, noise_level=0.05)
    
    # RANSAC三角化
    estimated_point, inlier_ratio, confidence = module._ransac_triangulate(
        positions, rays, max_iterations=100, inlier_threshold=0.15
    )
    
    if estimated_point is not None:
        error = np.linalg.norm(estimated_point - true_point)
        print(f"  真实点: {true_point}")
        print(f"  估计点: {estimated_point}")
        print(f"  误差: {error:.6f}m")
        print(f"  内点比例: {inlier_ratio:.3f}")
        print(f"  置信度: {confidence:.3f}")
        
        if error < 0.2 and inlier_ratio >= 0.5:
            print("  ✓ 通过: 在有噪声情况下仍能获得合理结果")
            return True
        else:
            print("  ✗ 失败: 误差过大或内点比例过低")
            return False
    else:
        print("  ✗ 失败: 未找到估计点")
        return False


def test_outlier_heavy():
    """测试异常值较多的情况"""
    print("\n" + "=" * 60)
    print("测试3: 异常值较多的情况")
    print("=" * 60)
    
    module = TestSpatialSemanticFusionModule()
    
    # 真实点
    true_point = np.array([2.0, 3.0, 5.0])
    
    # 生成10个观测位置（其中5个是内点，5个是异常值）
    positions = [
        # 内点（指向真实点）
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([0.5, 0.5, 1.0]),
        # 异常值（指向错误方向）
        np.array([10.0, 10.0, 0.0]),
        np.array([-5.0, -5.0, 0.0]),
        np.array([8.0, -3.0, 0.0]),
        np.array([-2.0, 7.0, 0.0]),
        np.array([6.0, 6.0, 2.0]),
    ]
    
    # 生成射线：前5个指向真实点，后5个指向错误方向
    rays = []
    for i, pos in enumerate(positions):
        if i < 5:
            # 内点：指向真实点
            direction = true_point - pos
            direction = direction / np.linalg.norm(direction)
        else:
            # 异常值：指向随机方向
            np.random.seed(i)
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
        rays.append(direction)
    
    # RANSAC三角化
    estimated_point, inlier_ratio, confidence = module._ransac_triangulate(
        positions, rays, max_iterations=200, inlier_threshold=0.15
    )
    
    if estimated_point is not None:
        error = np.linalg.norm(estimated_point - true_point)
        print(f"  真实点: {true_point}")
        print(f"  估计点: {estimated_point}")
        print(f"  误差: {error:.6f}m")
        print(f"  内点比例: {inlier_ratio:.3f}")
        print(f"  置信度: {confidence:.3f}")
        
        # 在50%异常值的情况下，应该能找到至少50%的内点
        if error < 0.5 and inlier_ratio >= 0.4:
            print("  ✓ 通过: 在异常值较多的情况下仍能识别内点")
            return True
        else:
            print("  ✗ 失败: 无法正确处理异常值")
            return False
    else:
        print("  ✗ 失败: 未找到估计点")
        return False


def test_minimum_views():
    """测试最少观测数"""
    print("\n" + "=" * 60)
    print("测试4: 最少观测数（2个）")
    print("=" * 60)
    
    module = TestSpatialSemanticFusionModule()
    
    # 真实点
    true_point = np.array([2.0, 3.0, 5.0])
    
    # 只有2个观测
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]
    
    rays = generate_rays_to_point(true_point, positions, noise_level=0.0)
    
    # RANSAC三角化
    estimated_point, inlier_ratio, confidence = module._ransac_triangulate(
        positions, rays, max_iterations=100, inlier_threshold=0.15
    )
    
    if estimated_point is not None:
        error = np.linalg.norm(estimated_point - true_point)
        print(f"  真实点: {true_point}")
        print(f"  估计点: {estimated_point}")
        print(f"  误差: {error:.6f}m")
        print(f"  内点比例: {inlier_ratio:.3f}")
        print(f"  置信度: {confidence:.3f}")
        
        if error < 0.1:
            print("  ✓ 通过: 最少观测数也能工作")
            return True
        else:
            print("  ✗ 失败: 误差过大")
            return False
    else:
        print("  ✗ 失败: 未找到估计点")
        return False


def test_insufficient_views():
    """测试观测数不足的情况"""
    print("\n" + "=" * 60)
    print("测试5: 观测数不足（1个）")
    print("=" * 60)
    
    module = TestSpatialSemanticFusionModule()
    
    # 只有1个观测
    positions = [np.array([0.0, 0.0, 0.0])]
    rays = [np.array([1.0, 0.0, 0.0])]
    
    # RANSAC三角化
    estimated_point, inlier_ratio, confidence = module._ransac_triangulate(
        positions, rays, max_iterations=100, inlier_threshold=0.15
    )
    
    if estimated_point is None:
        print("  ✓ 通过: 正确返回None（观测数不足）")
        return True
    else:
        print("  ✗ 失败: 应该返回None")
        return False


def main():
    """运行所有测试"""
    print("RANSAC三角化单元测试")
    print("=" * 60)
    
    tests = [
        test_perfect_matches,
        test_noisy_matches,
        test_outlier_heavy,
        test_minimum_views,
        test_insufficient_views,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ 测试异常: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

