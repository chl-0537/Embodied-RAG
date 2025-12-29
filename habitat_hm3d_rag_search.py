import os
import sys
import io
import argparse
import time
import base64
import json
import math
import random
from typing import Tuple, Dict, Any, List, Optional
from collections import deque
from datetime import datetime, timezone

import numpy as np

# 依赖 rag 框架中的组件（以模块形式导入，确保使用同一份全局对象）
import rag_robot_framework as rag


def _encode_image_data_url(img: np.ndarray, max_side: int = 640, fmt: str = "JPEG", quality: int = 85) -> str:
    """将 numpy RGB 图像编码为 base64 data URL，支持缩放与格式选择（默认JPEG以降低体积）。"""
    try:
        from PIL import Image  # Pillow 用于编码
    except Exception as e:
        rag.log_error("需要 Pillow: pip install pillow", e)
        raise

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[-1] == 4:
        mode = "RGBA"
    else:
        mode = "RGB"
    image = Image.fromarray(img, mode=mode)

    # 等比例缩放，最长边不超过 max_side
    w, h = image.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        image = image.resize(new_size)

    buf = io.BytesIO()
    if fmt.upper() == "JPEG" and mode == "RGBA":
        image = image.convert("RGB")
    save_kwargs = {"format": fmt.upper()}
    if fmt.upper() == "JPEG":
        save_kwargs.update({"quality": quality})
    image.save(buf, **save_kwargs)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _post_process_memory_world():
    """
    后处理：在所有帧处理完成后运行
    - MemoryGraph: 应用置信度衰减
    - WorldModel: 评估 missing 状态和确认 removed 状态
    """
    try:
        # 1. MemoryGraph: 应用置信度衰减
        if rag.memory_graph:
            try:
                updated_count = rag.memory_graph.decay_nodes()
                rag.log_info(f"MemoryGraph 衰减完成: 更新了 {updated_count} 个节点")
            except Exception as e:
                rag.log_warning(f"MemoryGraph 衰减失败: {e}")
        
        # 2. WorldModel: 评估每个实体的 missing 状态
        if rag.world_model:
            try:
                missing_count = 0
                for eid, ent in list(rag.world_model.entities.items()):
                    if ent.get("status") == "active":
                        # 评估 missing 状态
                        if rag.world_model_evaluate_missing(eid, n_no_detects=3, vis_th=0.6):
                            missing_count += 1
                
                if missing_count > 0:
                    rag.log_info(f"WorldModel: {missing_count} 个实体进入 missing 状态")
            except Exception as e:
                rag.log_warning(f"WorldModel 评估 missing 状态失败: {e}")
        
        # 3. WorldModel: 确认 removed 状态
        if rag.world_model:
            try:
                removed_count = 0
                for eid, ent in list(rag.world_model.entities.items()):
                    if ent.get("status") == "missing":
                        # 确认移除
                        if rag.world_model_confirm_removed(eid, removed_T=30*60):
                            removed_count += 1
                
                if removed_count > 0:
                    rag.log_info(f"WorldModel: {removed_count} 个实体确认移除")
            except Exception as e:
                rag.log_warning(f"WorldModel 确认移除失败: {e}")
    
    except Exception as e:
        rag.log_warning(f"后处理失败: {e}")


def _is_position_in_view(agent, world_pos: List[float], camera_intrinsics: Dict[str, float], depth: Optional[np.ndarray] = None) -> Tuple[bool, float]:
    """
    判断一个世界坐标是否在相机视锥并且深度显示可见
    
    Args:
        agent: HabitatAgentWrapper 实例
        world_pos: 世界坐标 [x, y, z]
        camera_intrinsics: 相机内参字典，包含 fx, fy, cx, cy, width, height
        depth: 深度图（可选）
    
    Returns:
        (in_view: bool, vis_score: float) - 是否在视野内，可见性分数 (0.0-1.0)
    """
    try:
        # 获取 agent 状态
        agent_state = agent.get_agent_state()
        agent_pos = np.array(agent_state.position)
        agent_rotation = agent_state.rotation
        
        # 计算距离
        world_pos_array = np.array(world_pos)
        dist = np.linalg.norm(world_pos_array - agent_pos)
        
        # 超出感知范围（10米）
        if dist > 10.0:
            return False, 0.0
        
        # 简化视锥检查：计算相对位置
        relative_pos = world_pos_array - agent_pos
        
        # 使用旋转矩阵将相对位置转换到相机坐标系
        # 简化处理：假设相机朝向与 agent 朝向一致
        # 这里使用简化的前向检查（z轴为正方向）
        # 实际应该使用完整的旋转矩阵，但为了简化，我们只做距离和粗略方向检查
        
        # 计算可见性分数（基于距离，越近分数越高）
        vis_score = max(0.0, 1.0 - (dist / 10.0))
        
        # 如果有深度图，可以进一步检查深度一致性
        if depth is not None:
            try:
                # 简化：如果距离在合理范围内，认为可见
                if 0.1 < dist < 10.0:
                    return True, vis_score
            except:
                pass
        
        # 简化判断：距离在合理范围内就认为在视野内
        if 0.1 < dist < 10.0:
            return True, vis_score
        
        return False, 0.0
    except Exception as e:
        rag.log_warning(f"_is_position_in_view 检查失败: {e}")
        return False, 0.0


def _vision_with_retries(rgb: np.ndarray, target_syns: List[str], retries: int = 3):
    """视觉调用带重试/降采样/改格式回退。返回 (success, result_or_errmsg)."""
    settings = [
        {"max_side": 640, "fmt": "JPEG", "quality": 85},
        {"max_side": 480, "fmt": "JPEG", "quality": 80},
        {"max_side": 480, "fmt": "PNG", "quality": 0},
    ]
    for i in range(min(retries, len(settings))):
        enc = settings[i]
        try:
            data_url = _encode_image_data_url(rgb, enc["max_side"], enc["fmt"], enc["quality"]) if enc["fmt"] != "PNG" else _encode_image_data_url(rgb, enc["max_side"], enc["fmt"], 0)
            res = rag.call_chatglm_multi_object_detection(data_url, target_objects=target_syns)
            if res.is_success():
                return True, res
            else:
                rag.log_warning(f"视觉检测失败(尝试{i+1}/{retries})：{res.error_msg}")
        except Exception as e:
            rag.log_warning(f"视觉调用异常(尝试{i+1}/{retries})：{e}")
    return False, f"视觉检测多次失败({retries})"


class HabitatAgentWrapper:
    """
    一个极简 Habitat-Lab/Habitat-Sim 代理包装：
    - 初始化场景
    - 提供 reset/step
    - 暴露 RGB 观测
    - 使用三种离散动作：MOVE_FORWARD / TURN_LEFT / TURN_RIGHT
    """

    def __init__(self, scene_path: str, sensor_width: int = 640, sensor_height: int = 480, move_step: float = 0.25, turn_angle: float = 10.0, dataset_config: str = None):
        # 导入 habitat_sim（如果失败会给出友好提示）
        try:
            import habitat_sim
            from habitat_sim.utils.common import d3_40_colors_rgb
            from habitat_sim import SimulatorConfiguration, AgentConfiguration, agent
            from habitat_sim.sensor import SensorType, SensorSubType
        except ImportError as e:
            # 检查是否是环境问题（os 和 sys 已在文件顶部导入）
            conda_env = os.environ.get('CONDA_DEFAULT_ENV', '未设置')
            python_path = sys.executable
            
            error_msg = (
                f"habitat_sim 模块无法导入。\n"
                f"当前 Python: {python_path}\n"
                f"当前 Conda 环境: {conda_env}\n"
                f"\n可能的原因：\n"
                f"1. habitat_sim 未安装\n"
                f"2. 未激活正确的 conda 环境（如果之前可以运行，很可能是这个问题）\n"
                f"3. Python 环境不匹配\n"
                f"\n解决方法：\n"
                f"1. 确认已激活正确的 conda 环境: conda activate <环境名>\n"
                f"2. 安装 habitat-sim: conda install -c conda-forge habitat-sim\n"
                f"3. 或参考: https://github.com/facebookresearch/habitat-sim\n"
                f"\n错误详情: {e}"
            )
            rag.log_error(error_msg)
            raise ImportError(error_msg) from e
        except Exception as e:
            rag.log_error("需要 habitat-sim/habitat-lab 环境，请参见安装文档。", e)
            raise

        self._hsim = habitat_sim
        self._move_step = move_step
        self._turn_angle = turn_angle

        # 基础检查：场景文件存在
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"场景文件不存在: {scene_path}")

        sim_cfg = SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        if dataset_config is not None:
            # 指定数据集配置，避免默认数据集缺失导致的断言失败
            sim_cfg.scene_dataset_config_file = dataset_config
        
        # 注意：不在这里禁用渲染器，因为需要 RGB 和深度传感器数据
        # 如果遇到 EGL/CUDA 错误，会在下面的异常处理中自动切换到 headless 模式

        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "rgb"
        rgb_sensor_spec.sensor_type = SensorType.COLOR
        rgb_sensor_spec.resolution = [sensor_height, sensor_width]
        rgb_sensor_spec.position = [0.0, 1.5, 0.0]
        rgb_sensor_spec.orientation = [0.0, 0.0, 0.0]
        rgb_sensor_spec.sensor_subtype = SensorSubType.PINHOLE
        # 设置水平视场角（HFOV），默认60度
        if not hasattr(rgb_sensor_spec, 'hfov') or rgb_sensor_spec.hfov is None:
            rgb_sensor_spec.hfov = math.radians(60.0)  # 默认60度，转换为弧度

        # 添加深度传感器
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth"
        depth_sensor_spec.sensor_type = SensorType.DEPTH
        depth_sensor_spec.resolution = [sensor_height, sensor_width]
        depth_sensor_spec.position = [0.0, 1.5, 0.0]
        depth_sensor_spec.orientation = [0.0, 0.0, 0.0]
        depth_sensor_spec.sensor_subtype = SensorSubType.PINHOLE
        # 深度传感器使用相同的HFOV
        if not hasattr(depth_sensor_spec, 'hfov') or depth_sensor_spec.hfov is None:
            depth_sensor_spec.hfov = rgb_sensor_spec.hfov

        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
        agent_cfg.action_space = {
            "move_forward": agent.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=self._move_step)),
            "move_backward": agent.ActionSpec("move_backward", habitat_sim.ActuationSpec(amount=self._move_step)),
            "turn_left": agent.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=self._turn_angle)),
            "turn_right": agent.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=self._turn_angle)),
        }

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        try:
            # 尝试创建模拟器，如果 EGL 失败则尝试其他配置
            try:
                self._sim = habitat_sim.Simulator(cfg)
            except RuntimeError as e:
                # 如果是 EGL/窗口上下文错误，尝试禁用渲染器
                error_str = str(e)
                if "EGL" in error_str or "WindowlessContext" in error_str or "CUDA device" in error_str:
                    rag.log_warning(f"EGL/窗口上下文创建失败，尝试使用 headless 模式: {e}")
                    # 重新配置为 headless 模式
                    sim_cfg.create_renderer = False
                    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
                    self._sim = habitat_sim.Simulator(cfg)
                    rag.log_info("已切换到 headless 模式（无渲染器）")
                else:
                    raise
        except AssertionError as ae:
            # 常见：dataset config 缺失或 scene 无效
            msg = (
                f"Habitat Simulator 初始化失败：{ae}\n"
                f"scene: {scene_path}\n"
                f"dataset_config: {dataset_config or 'None'}\n"
                f"建议：确认 scene 路径正确；若为 MP3D/HM3D 请提供 --dataset_config 指向 scene_dataset_config.json；"
                f"或改用 .scene_instance.json。"
            )
            rag.log_error(msg)
            raise
        except Exception as e:
            rag.log_error("Habitat Simulator 初始化异常", e)
            raise
        self._agent = self._sim.initialize_agent(0)
        
        # 提取相机内参
        self.camera_intrinsics = self._extract_camera_intrinsics(rgb_sensor_spec)
        rag.log_info("提取相机内参", self.camera_intrinsics)
    
    def _extract_camera_intrinsics(self, sensor_spec) -> Dict[str, float]:
        """
        从sensor_spec提取相机内参
        
        Args:
            sensor_spec: Habitat相机传感器规格
        
        Returns:
            包含fx, fy, cx, cy, width, height, hfov的字典
        """
        # 获取分辨率
        resolution = sensor_spec.resolution  # [height, width]
        height, width = resolution[0], resolution[1]
        
        # 获取水平视场角（HFOV）
        # 尝试多种方式获取HFOV
        hfov_rad = None
        
        # 方法1: 从sensor_spec直接获取
        if hasattr(sensor_spec, 'hfov') and sensor_spec.hfov is not None:
            hfov_rad = float(sensor_spec.hfov)
        
        # 方法2: 从simulator的agent配置中获取
        if hfov_rad is None:
            try:
                agent_config = self._sim.agents[0].agent_config
                for spec in agent_config.sensor_specifications:
                    if spec.uuid == sensor_spec.uuid and hasattr(spec, 'hfov') and spec.hfov is not None:
                        hfov_rad = float(spec.hfov)
                        break
            except Exception:
                pass
        
        # 方法3: 使用默认值
        if hfov_rad is None:
            hfov_rad = math.radians(60.0)  # 默认60度
            rag.log_warning(f"sensor_spec中未找到hfov，使用默认值60度 (传感器: {sensor_spec.uuid})")
        
        # 计算内参
        # fx = (width / 2) / tan(hfov / 2)
        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        fy = fx  # 对于针孔模型，通常fy = fx
        cx = width / 2.0
        cy = height / 2.0
        
        intrinsics = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
            "hfov_rad": hfov_rad,
            "hfov_deg": math.degrees(hfov_rad)
        }
        
        return intrinsics

    def reset(self):
        # 使用 habitat-sim 的 reset() 重置场景
        self._sim.reset()
        obs = self._sim.get_sensor_observations()
        return obs

    def get_rgb(self) -> np.ndarray:
        obs = self._sim.get_sensor_observations()
        rgb = obs["rgb"]  # HxWx3 uint8
        return rgb

    def get_depth(self) -> np.ndarray:
        """获取深度图"""
        obs = self._sim.get_sensor_observations()
        depth = obs.get("depth", None)  # HxW float32，单位：米
        return depth

    def get_observations(self) -> Dict[str, np.ndarray]:
        """获取所有观测（RGB + Depth）"""
        obs = self._sim.get_sensor_observations()
        return {
            "rgb": obs["rgb"],
            "depth": obs.get("depth", None)
        }

    def step(self, action: str):
        assert action in ("move_forward", "move_backward", "turn_left", "turn_right")
        self._sim.step(action)
        return self.get_rgb()
    
    def step_with_collision_check(self, action: str) -> Tuple[np.ndarray, bool]:
        """执行动作并检测碰撞，返回 (RGB图像, 是否碰撞)"""
        assert action in ("move_forward", "move_backward", "turn_left", "turn_right")
        prev_pos = np.array(self.get_agent_state().position)
        self._sim.step(action)
        new_pos = np.array(self.get_agent_state().position)
        # 计算位置变化
        pos_change = np.linalg.norm(new_pos - prev_pos)
        is_collision = pos_change < 1e-4  # 如果位置几乎没变化，说明可能碰撞了
        return self.get_rgb(), is_collision

    def get_agent_state(self):
        state = self._sim.get_agent(0).get_state()
        return state
    
    def get_position_and_rotation(self):
        """获取当前位置和旋转信息"""
        state = self.get_agent_state()
        pos = state.position
        rot = state.rotation  # 四元数
        return {
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "rotation": [float(rot.x), float(rot.y), float(rot.z), float(rot.w)],
            "rotation_quat": [float(rot.x), float(rot.y), float(rot.z), float(rot.w)]
        }


class SimpleExplorePolicy:
    """一个简单的探索策略：
    - 每个位置量化到网格，避免在同一处反复旋转
    - 尝试前进，碰撞或原地未变化则转向
    """

    def __init__(self, pos_quant: float = 0.25):
        self.visited_cells = set()
        self.pos_quant = pos_quant
        self.last_pos = None
        self.turn_count = 0

    def _quant_cell(self, pos: np.ndarray) -> Tuple[int, int]:
        return (
            int(math.floor(pos[0] / self.pos_quant)),
            int(math.floor(pos[2] / self.pos_quant)),
        )

    def plan(self, current_pos: np.ndarray) -> str:
        cell = self._quant_cell(current_pos)
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            self.turn_count = 0
            return "move_forward"

        # 已访问：小概率继续前进，否则转向探索新视角
        if self.turn_count < 10:
            self.turn_count += 1
            return "turn_left"
        else:
            self.turn_count = 0
            return "move_forward"


class FrontierBasedExplorer:
    """
    基于前沿探索（Frontier-based Exploration）的智能探索策略
    结合知识引导和A*路径规划
    """
    
    def __init__(self, pos_quant: float = 0.25, frontier_threshold: float = 3.0):
        self.pos_quant = pos_quant
        self.frontier_threshold = frontier_threshold  # 前沿检测的深度阈值（米）
        self.visited_cells = set()
        self.explored_map = {}  # 网格坐标 -> 探索状态
        self.frontiers = []  # 前沿点列表
        self.current_path = []  # 当前路径
        self.last_frontier_update = 0
        self.frontier_update_interval = 5  # 每5步更新一次前沿
        
    def _quant_cell(self, pos: np.ndarray) -> Tuple[int, int]:
        """将位置量化到网格"""
        return (
            int(math.floor(pos[0] / self.pos_quant)),
            int(math.floor(pos[2] / self.pos_quant)),
        )
    
    def _update_explored_map(self, current_pos: np.ndarray, depth: np.ndarray):
        """基于深度图更新探索地图"""
        if depth is None or depth.size == 0:
            return
        
        cell = self._quant_cell(current_pos)
        self.visited_cells.add(cell)
        
        # 分析深度图：标记已探索区域
        # 有效深度范围：0.5m - 5.0m
        valid_depth = (depth >= 0.5) & (depth <= 5.0)
        explored_ratio = np.sum(valid_depth) / depth.size
        
        # 存储探索信息
        self.explored_map[cell] = {
            "explored_ratio": explored_ratio,
            "timestamp": time.time()
        }
    
    def _detect_frontiers(self, current_pos: np.ndarray, depth: np.ndarray) -> List[Tuple[float, float]]:
        """
        检测前沿点：已探索和未探索区域的边界
        使用真正的frontier detection算法：
        - 已探索：深度值在有效范围内（0.5m - 5.0m）
        - 未探索：深度值过大（>5.0m）或无效
        - 前沿：已探索和未探索区域的边界
        返回前沿点的世界坐标列表
        """
        if depth is None or depth.size == 0:
            return []
        
        frontiers = []
        H, W = depth.shape
        
        # 定义已探索和未探索区域
        explored_mask = (depth >= 0.5) & (depth <= 5.0)  # 有效深度范围
        unexplored_mask = (depth > 5.0) | (depth < 0.5)  # 超出范围或无效
        
        # 检测前沿：已探索区域与未探索区域的边界
        # 使用形态学操作检测边界
        from scipy import ndimage
        try:
            # 膨胀已探索区域
            explored_dilated = ndimage.binary_dilation(explored_mask, structure=np.ones((3, 3)))
            # 前沿 = 膨胀后的已探索区域 - 原始已探索区域（即边界）
            frontier_mask = explored_dilated & ~explored_mask
        except ImportError:
            # 如果没有scipy，使用简单的边界检测
            frontier_mask = np.zeros_like(explored_mask, dtype=bool)
            for y in range(1, H-1):
                for x in range(1, W-1):
                    # 如果当前点是已探索，但邻居中有未探索，则是前沿
                    if explored_mask[y, x]:
                        neighbors = [
                            explored_mask[y-1, x], explored_mask[y+1, x],
                            explored_mask[y, x-1], explored_mask[y, x+1]
                        ]
                        if not all(neighbors):  # 有邻居是未探索
                            frontier_mask[y, x] = True
        
        # 采样前沿点（避免过多）
        step = max(1, min(H, W) // 15)  # 每15个像素采样一个
        for y in range(0, H, step):
            for x in range(0, W, step):
                if frontier_mask[y, x]:
                    depth_val = depth[y, x]
                    if 0.5 < depth_val < 5.0:
                        # 计算相对于当前位置的前沿点
                        # 使用简化的相机模型：假设FOV约为60度
                        fov_rad = math.radians(60)
                        pixel_angle_x = (x - W/2) / (W/2) * (fov_rad / 2)
                        # 计算世界坐标（简化：假设相机水平，只考虑x-z平面）
                        frontier_x = current_pos[0] + depth_val * math.sin(pixel_angle_x)
                        frontier_z = current_pos[2] + depth_val * math.cos(pixel_angle_x)
                        frontiers.append((frontier_x, frontier_z))
        
        # 去重和限制数量
        unique_frontiers = []
        seen = set()
        for f in frontiers[:30]:  # 最多保留30个前沿点
            f_cell = (int(f[0] / self.pos_quant), int(f[1] / self.pos_quant))
            if f_cell not in seen:
                seen.add(f_cell)
                unique_frontiers.append(f)
        
        return unique_frontiers
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """A*算法的启发式函数（曼哈顿距离）"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _astar_path(self, start: np.ndarray, goal: Tuple[float, float]) -> List[str]:
        """
        使用A*算法规划从当前位置到目标前沿的路径
        返回动作序列
        """
        start_cell = self._quant_cell(start)
        goal_cell = (int(goal[0] / self.pos_quant), int(goal[1] / self.pos_quant))
        
        # 如果目标就在当前网格，直接返回空路径
        if start_cell == goal_cell:
            return []
        
        # 计算目标方向（相对于起始位置）
        dx_world = goal[0] - start[0]
        dz_world = goal[1] - start[2]
        dist_world = math.sqrt(dx_world**2 + dz_world**2)
        
        # 如果距离很近，直接朝目标方向前进
        if dist_world < self.pos_quant * 2:
            # 计算需要转向的角度（简化：只考虑x-z平面）
            angle = math.atan2(dx_world, dz_world)
            # 简化：假设当前朝向是z轴正方向，需要转向angle角度
            if abs(angle) > math.radians(10):  # 如果角度大于10度，需要转向
                if angle > 0:
                    return ["turn_right", "move_forward"]
                else:
                    return ["turn_left", "move_forward"]
            else:
                return ["move_forward"]
        
        # 使用A*进行网格搜索
        open_set = [(0, start_cell)]
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: self._heuristic(start_cell, goal_cell)}
        
        max_iterations = 100  # 限制搜索范围
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            open_set.sort(key=lambda x: x[0])
            current = open_set.pop(0)[1]
            
            if current == goal_cell:
                # 重建路径
                path = []
                path_cells = [current]
                while current in came_from:
                    current = came_from[current]
                    path_cells.append(current)
                path_cells.reverse()
                
                # 将网格路径转换为动作序列
                for i in range(len(path_cells) - 1):
                    curr_cell = path_cells[i]
                    next_cell = path_cells[i + 1]
                    dx = next_cell[0] - curr_cell[0]
                    dz = next_cell[1] - curr_cell[1]
                    
                    # 根据方向决定动作
                    if dx > 0:  # 向东（x增加）
                        path.append("turn_right")
                        path.append("move_forward")
                    elif dx < 0:  # 向西（x减少）
                        path.append("turn_left")
                        path.append("move_forward")
                    elif dz > 0:  # 向北（z增加）
                        path.append("move_forward")
                    elif dz < 0:  # 向南（z减少）
                        path.append("turn_left")
                        path.append("turn_left")
                        path.append("move_forward")
                
                return path
            
            # 探索邻居（4方向：上下左右）
            neighbors = [
                (current[0] + 1, current[1]),  # 东
                (current[0] - 1, current[1]),  # 西
                (current[0], current[1] + 1),  # 北
                (current[0], current[1] - 1),  # 南
            ]
            
            for neighbor in neighbors:
                # 跳过已访问的网格（除非是目标）
                if neighbor in self.visited_cells and neighbor != goal_cell:
                    # 给已访问的网格更高的代价，但不完全禁止
                    cost = 2.0
                else:
                    cost = 1.0
                
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_cell)
                    # 检查是否已在open_set中
                    in_open = any(n == neighbor for _, n in open_set)
                    if not in_open:
                        open_set.append((f_score[neighbor], neighbor))
        
        # 如果无法找到路径，返回朝向目标的简单动作
        if dist_world > 0:
            angle = math.atan2(dx_world, dz_world)
            if abs(angle) > math.radians(10):
                if angle > 0:
                    return ["turn_right", "move_forward"]
                else:
                    return ["turn_left", "move_forward"]
            else:
                return ["move_forward"]
        
        return []
    
    def _get_knowledge_guided_direction(self, target_object: str, memory_evidences: List[Dict]) -> Optional[Tuple[float, float]]:
        """
        基于RAG记忆和知识引导，推断目标可能的位置方向
        返回建议的前沿点坐标（如果可用）
        """
        if not memory_evidences:
            return None
        
        # 从记忆证据中提取位置信息
        location_hints = []
        for evidence in memory_evidences:
            if evidence.get("type") == "memory":
                meta = evidence.get("meta", {})
                location = meta.get("spatial", {}).get("location_semantic", "")
                if location and location != "unknown":
                    location_hints.append(location)
        
        # 基于目标物品的常见位置知识
        location_knowledge = {
            "水杯": ["桌子", "desk", "table", "厨房", "kitchen"],
            "杯子": ["桌子", "desk", "table", "厨房", "kitchen"],
            "手机": ["桌子", "desk", "沙发", "sofa", "床", "bed"],
            "书": ["桌子", "desk", "书架", "bookshelf"],
            "电脑": ["桌子", "desk", "书房", "study"],
            "床": ["卧室", "bedroom"],
            "钢琴": ["客厅", "living room", "房间", "room"],
            "椅子": ["桌子", "desk", "餐厅", "dining room"],
            "台灯": ["桌子", "desk", "床头柜", "nightstand"],
            "时钟": ["墙", "wall", "客厅", "living room"],
        }
        
        # 检查是否有匹配的位置提示
        target_lower = target_object.lower()
        for item, locations in location_knowledge.items():
            if item in target_lower:
                for loc in locations:
                    if any(loc.lower() in hint.lower() for hint in location_hints):
                        rag.log_info(f"知识引导：目标 '{target_object}' 可能在 {loc} 附近")
                        # 返回None表示需要继续探索，但可以优先选择相关方向
                        return None
        
        return None
    
    def plan(self, current_pos: np.ndarray, depth: np.ndarray = None, target_object: str = None, 
             memory_evidences: List[Dict] = None, step_count: int = 0) -> str:
        """
        智能规划下一步动作
        """
        # 更新探索地图
        if depth is not None:
            self._update_explored_map(current_pos, depth)
        
        # 如果有当前路径，继续执行
        if self.current_path:
            action = self.current_path.pop(0)
            return action
        
        # 定期更新前沿
        if step_count - self.last_frontier_update >= self.frontier_update_interval and depth is not None:
            self.frontiers = self._detect_frontiers(current_pos, depth)
            self.last_frontier_update = step_count
            rag.log_info(f"检测到 {len(self.frontiers)} 个前沿点")
        
        # 如果有前沿点，选择最近的一个并规划路径
        if self.frontiers:
            # 选择最近的前沿点
            current_cell = self._quant_cell(current_pos)
            best_frontier = None
            min_dist = float('inf')
            
            for frontier in self.frontiers:
                frontier_cell = (int(frontier[0] / self.pos_quant), int(frontier[1] / self.pos_quant))
                dist = abs(frontier_cell[0] - current_cell[0]) + abs(frontier_cell[1] - current_cell[1])
                if dist < min_dist and frontier_cell not in self.visited_cells:
                    min_dist = dist
                    best_frontier = frontier
            
            if best_frontier:
                # 使用A*规划路径
                path = self._astar_path(current_pos, best_frontier)
                if path:
                    self.current_path = path[1:]  # 跳过第一个动作，立即执行
                    return path[0] if path else "move_forward"
                else:
                    # 如果无法规划路径，直接朝向前沿方向
                    self.frontiers.remove(best_frontier)
        
        # 如果没有前沿或无法到达，使用知识引导
        if target_object and memory_evidences:
            knowledge_goal = self._get_knowledge_guided_direction(target_object, memory_evidences)
            if knowledge_goal:
                path = self._astar_path(current_pos, knowledge_goal)
                if path:
                    self.current_path = path[1:]
                    return path[0] if path else "move_forward"
        
        # 回退到简单策略
        cell = self._quant_cell(current_pos)
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            return "move_forward"
        else:
            # 随机选择转向或前进
            if random.random() < 0.7:
                return "turn_left"
            else:
                return "move_forward"


def habitat_rag_search(scene_path: str, target_object: str, max_steps: int = 1000, rotate_sweep: int = 12, 
                       save_obs_dir: str = None, explore_policy: str = "frontier",
                       fusion_mode: str = "triangulate") -> Dict[str, Any]:
    """
    主循环：
    1) 初始化向量库 + Habitat 场景
    2) 解析意图（对象、位置可选）
    3) 探索循环：捕获 RGB -> 调用视觉检测 -> 写回视觉记忆 -> 分析是否找到
    4) 若未找到：根据探索策略继续（前沿探索+知识引导 或 简单策略）
    5) 找到则返回成功
    
    Args:
        explore_policy: "simple" 或 "frontier"（默认）
    """

    # 初始化向量数据库
    db = rag.init_chroma_db()
    if db is None:
        rag.log_warning("Chroma 向量库初始化失败（ChromaDB 未安装），某些功能可能不可用")
        # 不抛出异常，允许程序继续运行（某些功能可能受限）
        rag.chroma_client, rag.chroma_collection = None, None
    else:
        rag.chroma_client, rag.chroma_collection = db

    # 构造用户指令并解析
    user_cmd = f"请帮我在场景中找到：{target_object}。"
    intent_res = rag.parse_user_intent_llm(user_cmd)
    if not intent_res.is_success():
        raise RuntimeError(f"意图解析失败: {intent_res.error_msg}")
    intent = intent_res.data
    if not intent.get("object"):
        intent["object"] = target_object

    # 提取当前时间上下文（用于时间语义记忆）
    current_time_context = rag.extract_time_context(time.time())
    rag.log_info("当前时间上下文已提取", current_time_context)

    # 初步记忆检索（可用于日志/后续融合，使用时间优先级）
    _ = rag.retrieve_candidates_with_time_prior(intent, current_time_context=current_time_context, topk=10)

    # 目标同义词（中英文）用于视觉检测强聚焦与匹配
    def _target_synonyms(name: str) -> List[str]:
        name = str(name or "").strip()
        lower = name.lower()
        syn = {name}
        # 扩展同义词映射
        mapping = {
            "水杯": ["watercup", "cup", "mug", "glass"],
            "杯子": ["cup", "mug", "glass"],
            "手机": ["phone", "smartphone", "mobile phone", "cell phone"],
            "书": ["book", "books", "novel"],
            "电脑": ["laptop", "computer", "pc", "desktop"],
            "笔记本": ["laptop", "notebook computer"],
            "床": ["bed", "mattress"],
            "大床": ["bed", "king bed", "double bed", "queen bed"],
            "钢琴": ["piano", "grand piano", "upright piano", "keyboard", "pianoforte"],
            "椅子": ["chair", "seat", "stool", "armchair"],
            "桌子": ["table", "desk", "dining table"],
            "台灯": ["lamp", "desk lamp", "table lamp", "light"],
            "时钟": ["clock", "wall clock", "timepiece"],
            "沙发": ["sofa", "couch", "settee"],
            "电视": ["tv", "television", "television set"],
            "冰箱": ["refrigerator", "fridge", "freezer"],
        }
        for zh, lst in mapping.items():
            if zh in name:
                syn.update(lst)
                syn.update([zh])  # 也包含中文原词
        return list(syn)
    target_syns = _target_synonyms(intent.get("object", target_object))

    # 初始化 Habitat 代理
    try:
        agent = HabitatAgentWrapper(scene_path, dataset_config=globals().get("_dataset_config_path"))
        agent.reset()
    except ImportError as e:
        # habitat_sim 未安装，给出明确的错误提示
        error_msg = (
            f"\n{'='*60}\n"
            f"错误: habitat_sim 模块未安装\n"
            f"{'='*60}\n"
            f"请安装 habitat-sim 和 habitat-lab：\n"
            f"  1. conda install -c conda-forge habitat-sim\n"
            f"  2. 或参考: https://github.com/facebookresearch/habitat-sim\n"
            f"\n详细错误: {e}\n"
            f"{'='*60}\n"
        )
        rag.log_error(error_msg)
        raise RuntimeError("habitat_sim 未安装，无法运行。请先安装 habitat-sim。") from e
    # 根据参数选择探索策略
    if explore_policy == "frontier":
        policy = FrontierBasedExplorer(pos_quant=0.25, frontier_threshold=3.0)
        rag.log_info("使用前沿探索+知识引导策略")
    else:
        policy = SimpleExplorePolicy(pos_quant=0.25)
        rag.log_info("使用简单探索策略")

    # 准备观测保存目录（未指定则自动生成）
    if not save_obs_dir:
        ts = time.strftime("%Y%m%d_%H%M%S")
        scene_base = os.path.splitext(os.path.basename(scene_path))[0]
        target_safe = str(target_object).strip().replace(" ", "_")
        save_obs_dir = os.path.join(os.path.dirname(__file__), "run_logs", f"obs_{ts}_{scene_base}_{target_safe}")
    os.makedirs(save_obs_dir, exist_ok=True)
    sweep_dir = os.path.join(save_obs_dir, "sweep")
    explore_dir = os.path.join(save_obs_dir, "explore")
    depth_dir = os.path.join(save_obs_dir, "depth")
    detection_dir = os.path.join(save_obs_dir, "detections")
    frontier_dir = os.path.join(save_obs_dir, "frontiers")
    os.makedirs(sweep_dir, exist_ok=True)
    os.makedirs(explore_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(detection_dir, exist_ok=True)
    os.makedirs(frontier_dir, exist_ok=True)
    
    # 初始化路径轨迹记录
    path_trajectory = []
    start_time = time.time()
    
    # 记录起始位置
    start_pose = agent.get_position_and_rotation()
    path_trajectory.append({
        "timestamp": time.time(),
        "step": 0,
        "phase": "start",
        "action": "reset",
        **start_pose
    })
    rag.log_info("记录起始位置", start_pose)

    def _save_rgb(img: np.ndarray, out_path: str):
        try:
            from PIL import Image
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(out_path)
        except Exception as e:
            rag.log_warning(f"保存观测图像失败: {out_path}")
    
    def _save_depth(depth: np.ndarray, out_path: str, normalize: bool = True):
        """
        保存深度图
        
        Args:
            depth: 深度图数组 (HxW, float32，单位：米)
            out_path: 输出路径
            normalize: 是否归一化到0-255范围（便于可视化）
        """
        try:
            from PIL import Image
            if depth is None or depth.size == 0:
                rag.log_warning(f"深度图为空，跳过保存: {out_path}")
                return
            
            if normalize:
                # 归一化到0-255范围（假设深度范围0-10米）
                depth_min = 0.0
                depth_max = 10.0
                depth_clipped = np.clip(depth, depth_min, depth_max)
                depth_normalized = ((depth_clipped - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                Image.fromarray(depth_normalized, mode='L').save(out_path)
            else:
                # 保存原始深度值（使用16位PNG，转换为毫米）
                depth_mm = (depth * 1000).astype(np.uint16)  # 转换为毫米，然后转为uint16
                # 使用PIL保存16位灰度图
                depth_img = Image.fromarray(depth_mm, mode='I;16')
                depth_img.save(out_path.replace('.png', '_raw.png'))
                # 同时保存归一化版本便于查看
                depth_normalized = ((np.clip(depth, 0, 10) / 10.0) * 255).astype(np.uint8)
                Image.fromarray(depth_normalized, mode='L').save(out_path)
            
            rag.log_info(f"保存深度图: {out_path} (范围: {depth.min():.2f}m - {depth.max():.2f}m)")
        except Exception as e:
            rag.log_warning(f"保存深度图失败: {out_path} - {e}")
    
    def _save_detection_result(vision_result: Dict, out_path: str):
        """保存检测结果到JSON文件"""
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(vision_result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            rag.log_warning(f"保存检测结果失败: {out_path} - {e}")
    
    def normalize_and_validate_bbox(bbox: List[float], img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int, float]]:
        """
        统一的边界框归一化和验证工具函数
        
        功能：
        1. 自动检测bbox格式（归一化[0,1]或像素坐标）
        2. 转换为像素坐标 (x1, y1, x2, y2)
        3. 裁剪到图像边界
        4. 确保 x1 < x2, y1 < y2
        5. 计算边界框面积占比
        
        Args:
            bbox: 边界框 [x1, y1, x2, y2]，可能是归一化坐标[0,1]或像素坐标
            img_w: 图像宽度（像素）
            img_h: 图像高度（像素）
        
        Returns:
            (x1, y1, x2, y2, area_ratio) 或 None（如果bbox无效）
            - x1, y1, x2, y2: 像素坐标，已确保 x1 < x2, y1 < y2
            - area_ratio: 边界框面积占比 (bbox_area / (img_w * img_h))
        """
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (ValueError, TypeError):
            return None
        
        # 检测是否为归一化坐标
        # 判断标准：如果所有值都在[0, 1]范围内，或者最大值<=1.5，认为是归一化坐标
        is_normalized = (
            (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1) or
            (max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5)
        )
        
        if is_normalized:
            # 归一化坐标 -> 像素坐标
            x1, x2 = x1 * img_w, x2 * img_w
            y1, y2 = y1 * img_h, y2 * img_h
        
        # 确保顺序正确：x1 < x2, y1 < y2
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        
        # 裁剪到图像边界
        x1 = max(0, min(img_w - 1, int(round(x1))))
        y1 = max(0, min(img_h - 1, int(round(y1))))
        x2 = max(x1 + 1, min(img_w, int(round(x2))))
        y2 = max(y1 + 1, min(img_h, int(round(y2))))
        
        # 计算面积占比
        bbox_area = (x2 - x1) * (y2 - y1)
        img_area = img_w * img_h
        area_ratio = bbox_area / img_area if img_area > 0 else 0.0
        
        # 验证：确保bbox有效（面积>0）
        if bbox_area <= 0:
            return None
        
        return (x1, y1, x2, y2, area_ratio)
    
    def _save_frontier_info(frontiers: List[Tuple[float, float]], current_pos: np.ndarray, out_path: str):
        """保存前沿点信息到JSON文件"""
        try:
            frontier_data = {
                "current_position": current_pos.tolist() if isinstance(current_pos, np.ndarray) else current_pos,
                "frontier_count": len(frontiers),
                "frontiers": [
                    {"x": float(f[0]), "z": float(f[1])} for f in frontiers
                ],
                "timestamp": time.time()
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(frontier_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            rag.log_warning(f"保存前沿信息失败: {out_path} - {e}")
    
    def _save_path_log(trajectory: List[Dict], save_dir: str, start_time: float):
        """保存路径轨迹日志到JSON文件（使用原子性写入确保完整性）"""
        try:
            # 清理trajectory数据，确保所有数据可序列化
            cleaned_trajectory = []
            for entry in trajectory:
                cleaned_entry = {}
                for key, value in entry.items():
                    # 转换numpy数组为list
                    if isinstance(value, np.ndarray):
                        cleaned_entry[key] = value.tolist()
                    # 转换numpy标量为Python原生类型
                    elif isinstance(value, (np.integer, np.floating)):
                        cleaned_entry[key] = value.item()
                    # 跳过不可序列化的对象（如函数、类实例等）
                    elif isinstance(value, (str, int, float, bool, type(None))):
                        cleaned_entry[key] = value
                    elif isinstance(value, (list, tuple)):
                        # 递归清理列表/元组中的numpy对象
                        cleaned_list = []
                        for item in value:
                            if isinstance(item, np.ndarray):
                                cleaned_list.append(item.tolist())
                            elif isinstance(item, (np.integer, np.floating)):
                                cleaned_list.append(item.item())
                            else:
                                cleaned_list.append(item)
                        cleaned_entry[key] = cleaned_list
                    elif isinstance(value, dict):
                        # 递归清理字典
                        cleaned_dict = {}
                        for k, v in value.items():
                            if isinstance(v, np.ndarray):
                                cleaned_dict[k] = v.tolist()
                            elif isinstance(v, (np.integer, np.floating)):
                                cleaned_dict[k] = v.item()
                            else:
                                cleaned_dict[k] = v
                        cleaned_entry[key] = cleaned_dict
                    else:
                        # 对于其他类型，尝试转换为字符串
                        try:
                            cleaned_entry[key] = str(value)
                        except:
                            # 如果转换失败，跳过该字段
                            rag.log_warning(f"跳过不可序列化的字段: {key}")
                            continue
                cleaned_trajectory.append(cleaned_entry)
            
            log_data = {
                "start_time": start_time,
                "end_time": time.time(),
                "duration_seconds": time.time() - start_time,
                "total_steps": len(trajectory) - 1,  # 减去起始步骤
                "start_position": cleaned_trajectory[0]["position"] if cleaned_trajectory else None,
                "end_position": cleaned_trajectory[-1]["position"] if cleaned_trajectory else None,
                "trajectory": cleaned_trajectory
            }
            
            log_file = os.path.join(save_dir, "path_trajectory.json")
            
            # 使用原子性写入：先写入临时文件，成功后再替换原文件
            temp_file = log_file + ".tmp"
            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)
                    f.flush()  # 确保数据写入磁盘缓冲区
                    os.fsync(f.fileno())  # 强制同步到磁盘
                
                # 原子性替换：只有在写入成功后才替换原文件
                # 如果原文件存在，先备份（可选）
                if os.path.exists(log_file):
                    backup_file = log_file + ".bak"
                    try:
                        os.replace(log_file, backup_file)
                    except:
                        pass  # 如果备份失败，继续执行
                
                os.replace(temp_file, log_file)
                
                rag.log_info(f"路径轨迹日志已保存: {log_file}", {
                    "total_steps": log_data["total_steps"],
                    "duration": log_data["duration_seconds"],
                    "trajectory_size": len(cleaned_trajectory)
                })
            except Exception as write_error:
                # 如果写入失败，尝试清理临时文件
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                raise write_error
                
        except Exception as e:
            rag.log_warning(f"保存路径轨迹日志失败: {e}", {
                "trajectory_size": len(trajectory) if trajectory else 0,
                "save_dir": save_dir
            })
            # 如果保存失败，尝试保存一个最小化的错误日志
            try:
                error_log = os.path.join(save_dir, "path_trajectory_error.json")
                with open(error_log, "w", encoding="utf-8") as f:
                    json.dump({
                        "error": str(e),
                        "trajectory_size": len(trajectory) if trajectory else 0,
                        "timestamp": time.time()
                    }, f, ensure_ascii=False, indent=2)
            except:
                pass  # 如果连错误日志都保存失败，放弃

    def _find_matching_object(vision_result: Dict, target_name: str, img_shape: Tuple[int, int, int] = None, min_conf: float = 0.6, min_area_frac: float = 0.005, max_area_frac: float = 0.8) -> Dict[str, Any]:
        """严格匹配：仅当 label 含同义词且置信度/面积阈值达标时返回对象。"""
        objs = vision_result.get("objects", []) if isinstance(vision_result, dict) else []
        if not target_name or not objs:
            return None
        target_lower = str(target_name).lower()
        H = img_shape[0] if (img_shape is not None and len(img_shape) >= 2) else None
        W = img_shape[1] if (img_shape is not None and len(img_shape) >= 2) else None
        # 扩展同义词表
        keyword_mapping = {
            "水杯": ["watercup", "cup", "mug", "glass"],
            "杯子": ["cup", "mug", "glass"],
            "手机": ["phone", "smartphone", "mobile phone", "cell phone"],
            "书": ["book", "books", "novel", "textbook"],
            "电脑": ["laptop", "computer", "pc", "desktop"],
            "笔记本": ["laptop", "notebook computer"],
            "床": ["bed", "mattress"],
            "大床": ["bed", "king bed", "double bed", "queen bed"],
            "钢琴": ["piano", "grand piano", "upright piano", "keyboard", "pianoforte"],
            "椅子": ["chair", "seat", "stool", "armchair"],
            "桌子": ["table", "desk", "dining table"],
            "台灯": ["lamp", "desk lamp", "table lamp", "light"],
            "时钟": ["clock", "wall clock", "timepiece"],
            "沙发": ["sofa", "couch", "settee"],
            "电视": ["tv", "television", "television set"],
            "冰箱": ["refrigerator", "fridge", "freezer"],
            "门": ["door", "doorway"],
            "窗户": ["window", "glass"],
        }
        synonyms = set()
        for zh, en_list in keyword_mapping.items():
            if zh in target_lower:
                synonyms.update(en_list)
        synonyms.update([target_lower])
        best = None
        for obj in objs:
            lab = str(obj.get("label", "")).lower()
            conf = float(obj.get("confidence", 0.0) or 0.0)
            bbox = obj.get("bbox")
            
            # 必须在 label 中命中同义词
            if not any(kw in lab for kw in synonyms):
                continue
            if conf < min_conf:
                continue
            
            # 使用统一的bbox归一化函数
            if H and W:
                bbox_result = normalize_and_validate_bbox(bbox, W, H)
                if bbox_result is None:
                    continue
                x1, y1, x2, y2, area_frac = bbox_result
                # 检查面积占比是否在合理范围内
                if area_frac < min_area_frac or area_frac > max_area_frac:
                    continue
            elif not H or not W:
                # 如果没有图像尺寸信息，只检查bbox格式
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
            
            best = obj
            break
        return best

    class SpatialSemanticFusionModule:
        """
        空间语义融合模块：通过多视角几何三角化实现小物体的精确3D定位
        """
        
        def __init__(self, max_buffer: int = 10, min_views: int = 3, 
                     camera_intrinsics: Dict[str, float] = None,
                     alpha: float = 0.7, lambda_spatial: float = 1.0,
                     log_dir: str = None, fusion_mode: str = "triangulate",
                     voxel_size: float = 0.02, grid_width: float = 0.4, grid_height: float = 0.4):
            """
            Args:
                max_buffer: 缓冲区最大帧数
                min_views: 进行三角化所需的最少视角数
                camera_intrinsics: 相机内参字典，包含fx, fy, cx, cy, width, height
                                  如果为None，使用默认值（适配640x480图像）
                alpha: 视觉置信度权重（默认0.7）
                lambda_spatial: 空间距离衰减系数（默认1.0米）
                log_dir: 日志目录，用于保存fusion_trace.json
                fusion_mode: 融合模式，"triangulate" | "depth" | "voxel_vote"（默认"triangulate"）
                voxel_size: 体素大小（米），默认0.02m
                grid_width: 体素网格宽度（米），默认0.4m
                grid_height: 体素网格高度（米），默认0.4m
            """
            self.max_buffer = max_buffer
            self.min_views = min_views
            self.buffer = []  # 存储多帧观测
            self.log_dir = log_dir
            self.fusion_trace_file = None
            self.fusion_mode = fusion_mode  # 融合模式
            self.voxel_size = voxel_size  # 体素大小（米）
            self.grid_width = grid_width  # 体素网格宽度（米）
            self.grid_height = grid_height  # 体素网格高度（米）
            
            # 初始化融合追踪日志文件
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
                self.fusion_trace_file = os.path.join(self.log_dir, "fusion_trace.json")
                # 如果文件不存在，创建空数组
                if not os.path.exists(self.fusion_trace_file):
                    with open(self.fusion_trace_file, "w", encoding="utf-8") as f:
                        json.dump([], f, ensure_ascii=False, indent=2)
                rag.log_info(f"空间融合模块：融合追踪日志文件: {self.fusion_trace_file}")
            
            # 从camera_intrinsics提取内参，如果没有提供则使用默认值
            if camera_intrinsics is not None:
                self.fx = float(camera_intrinsics.get("fx", 320.0))
                self.fy = float(camera_intrinsics.get("fy", 320.0))
                self.cx = float(camera_intrinsics.get("cx", 320.0))
                self.cy = float(camera_intrinsics.get("cy", 240.0))
                self.img_width = int(camera_intrinsics.get("width", 640))
                self.img_height = int(camera_intrinsics.get("height", 480))
                rag.log_info(f"空间融合模块：使用相机内参 fx={self.fx:.2f}, fy={self.fy:.2f}, "
                           f"cx={self.cx:.2f}, cy={self.cy:.2f}, 分辨率={self.img_width}x{self.img_height}")
            else:
                # 默认值（向后兼容）
                self.fx = 320.0
                self.fy = 320.0
                self.cx = 320.0
                self.cy = 240.0
                self.img_width = 640
                self.img_height = 480
                rag.log_warning("空间融合模块：未提供相机内参，使用默认值（640x480）")
            
            self.alpha = alpha
            self.lambda_spatial = lambda_spatial
            self.estimated_position = None  # 估计的3D位置
            self.estimated_confidence = 0.0  # 融合后的置信度
            self.last_reliable_estimate = None  # 最后一个可靠估计
            self.last_reliable_confidence = 0.0  # 最后一个可靠估计的置信度
            self.anomaly_count = 0  # 异常观测计数
        
        def _check_observation_anomaly(self, bbox_result: Optional[Tuple], depth: Optional[np.ndarray],
                                       x1: int, y1: int, x2: int, y2: int, area_ratio: float,
                                       img_w: int, img_h: int) -> Tuple[bool, str, Dict]:
            """
            检查观测是否异常
            
            Args:
                bbox_result: bbox归一化结果
                depth: 深度图
                x1, y1, x2, y2: bbox像素坐标
                area_ratio: bbox面积占比
                img_w, img_h: 图像尺寸
            
            Returns:
                (is_anomaly, reason, raw_data)
            """
            raw_data = {
                "bbox": [x1, y1, x2, y2],
                "area_ratio": area_ratio,
                "has_depth": depth is not None
            }
            
            # 1. 检查bbox是否有效
            if bbox_result is None:
                return True, "invalid_bbox", raw_data
            
            # 2. 检查bbox大小（太小或太大）
            min_area_ratio = 0.0005  # 0.05% 最小面积
            max_area_ratio = 0.80   # 80% 最大面积
            
            if area_ratio < min_area_ratio:
                raw_data["area_ratio"] = area_ratio
                return True, f"bbox_too_small (area_ratio={area_ratio:.6f} < {min_area_ratio})", raw_data
            
            if area_ratio > max_area_ratio:
                raw_data["area_ratio"] = area_ratio
                return True, f"bbox_too_large (area_ratio={area_ratio:.6f} > {max_area_ratio})", raw_data
            
            # 3. 检查bbox中心是否在有效区域（距离图像边缘至少5%）
            margin_ratio = 0.05
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            margin_x = img_w * margin_ratio
            margin_y = img_h * margin_ratio
            
            if cx < margin_x or cx > (img_w - margin_x) or cy < margin_y or cy > (img_h - margin_y):
                raw_data["bbox_center"] = [cx, cy]
                raw_data["margins"] = [margin_x, margin_y]
                return True, f"bbox_center_out_of_valid_region (center=({cx:.1f}, {cy:.1f}), margins=({margin_x:.1f}, {margin_y:.1f}))", raw_data
            
            # 4. 检查深度（如果提供了深度图）
            if depth is not None:
                # 提取bbox区域的深度值
                depth_h, depth_w = depth.shape[0], depth.shape[1]
                
                # 确保坐标在有效范围内
                x1_clamped = max(0, min(depth_w - 1, x1))
                y1_clamped = max(0, min(depth_h - 1, y1))
                x2_clamped = max(x1_clamped + 1, min(depth_w, x2))
                y2_clamped = max(y1_clamped + 1, min(depth_h, y2))
                
                depth_roi = depth[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
                valid_depths = depth_roi[(depth_roi >= 0.1) & (depth_roi <= 10.0) & 
                                         (~np.isnan(depth_roi)) & (~np.isinf(depth_roi))]
                
                if len(valid_depths) == 0:
                    raw_data["depth_stats"] = {
                        "min": float(np.nanmin(depth_roi)) if depth_roi.size > 0 else None,
                        "max": float(np.nanmax(depth_roi)) if depth_roi.size > 0 else None,
                        "valid_count": 0
                    }
                    return True, "missing_valid_depth (bbox区域内无有效深度值)", raw_data
                
                mean_depth = np.mean(valid_depths)
                min_depth = 0.1  # 10cm
                max_depth = 10.0  # 10m
                
                if mean_depth < min_depth:
                    raw_data["mean_depth"] = float(mean_depth)
                    return True, f"depth_too_small (mean_depth={mean_depth:.3f}m < {min_depth}m)", raw_data
                
                if mean_depth > max_depth:
                    raw_data["mean_depth"] = float(mean_depth)
                    return True, f"depth_too_large (mean_depth={mean_depth:.3f}m > {max_depth}m)", raw_data
                
                raw_data["mean_depth"] = float(mean_depth)
                raw_data["valid_depth_count"] = len(valid_depths)
            
            # 所有检查通过
            return False, "", raw_data
        
        def add_observation(self, position: List[float], rotation: List[float], 
                          bbox: List[float], confidence: float, label: str, 
                          img_shape: Tuple[int, int] = None, depth: np.ndarray = None):
            """
            添加一次观测到缓冲区
            
            Args:
                position: 机器人位置 [x, y, z]
                rotation: 机器人旋转四元数 [x, y, z, w]
                bbox: 边界框 [x1, y1, x2, y2]（像素坐标或归一化坐标）
                confidence: 检测置信度
                label: 物体标签
                img_shape: 图像尺寸 (H, W)
                depth: 深度图 (H, W)，单位：米，可选
            """
            # 检查图像尺寸是否与内参一致
            if img_shape is not None:
                img_h, img_w = img_shape[0], img_shape[1]
                # 如果图像尺寸与内参中的尺寸不一致，发出警告但继续使用内参尺寸
                if img_w != self.img_width or img_h != self.img_height:
                    rag.log_warning(f"空间融合模块：图像尺寸 ({img_w}x{img_h}) 与内参尺寸 ({self.img_width}x{self.img_height}) 不一致，"
                                  f"使用内参尺寸进行归一化")
                # 使用内参中的尺寸（而不是实际图像尺寸），确保内参一致性
                bbox_img_w, bbox_img_h = self.img_width, self.img_height
            else:
                # 使用内参中的尺寸
                bbox_img_w, bbox_img_h = self.img_width, self.img_height
            
            # 使用统一的bbox归一化函数（使用内参中的尺寸）
            bbox_result = normalize_and_validate_bbox(bbox, bbox_img_w, bbox_img_h)
            if bbox_result is None:
                rag.log_warning(f"空间融合模块：异常观测 - invalid_bbox, 原始数据: {bbox}")
                self.anomaly_count += 1
                return None
            
            x1, y1, x2, y2, area_ratio = bbox_result
            
            # 检查观测异常
            is_anomaly, anomaly_reason, raw_data = self._check_observation_anomaly(
                bbox_result, depth, x1, y1, x2, y2, area_ratio, bbox_img_w, bbox_img_h
            )
            
            if is_anomaly:
                rag.log_warning(f"空间融合模块：异常观测被拒绝 - {anomaly_reason}, 原始数据: {raw_data}")
                self.anomaly_count += 1
                return None
            
            # 计算bbox中心（归一化坐标）
            cx = (x1 + x2) / 2.0 / self.img_width
            cy = (y1 + y2) / 2.0 / self.img_height
            
            # 初始化观测字典
            observation = {
                "position": position,
                "rotation": rotation,
                "bbox_center": [cx, cy],
                "confidence": confidence,
                "label": label,
                "timestamp": time.time(),
                "type": "triangulation",  # 默认类型
                "world_position": None,  # 世界坐标位置（如果可用）
            }
            
            # 深度反投影：如果深度图可用，进行深度反投影
            if depth is not None:
                depth_world_pos = self._backproject_depth_to_world(
                    depth, x1, y1, x2, y2, position, rotation
                )
                if depth_world_pos is not None:
                    observation["type"] = "depth"
                    observation["world_position"] = depth_world_pos.tolist()
                    # 深度观测的置信度更高（在视觉置信度基础上增加）
                    observation["confidence"] = min(1.0, confidence + 0.2)
                    rag.log_info(f"空间融合模块：深度反投影成功，3D位置: {depth_world_pos.tolist()}, "
                               f"置信度: {observation['confidence']:.3f}")
                else:
                    rag.log_warning(f"空间融合模块：深度反投影失败（无效深度值或超出范围）")
            
            self.buffer.append(observation)
            
            # 保持缓冲区大小
            if len(self.buffer) > self.max_buffer:
                self.buffer.pop(0)
            
            # 观测成功添加，重置异常计数
            self.anomaly_count = 0
            
            obs_type = observation.get("type", "triangulation")
            rag.log_info(f"空间融合模块：添加观测 (类型: {obs_type}, 缓冲区大小: {len(self.buffer)}/{self.max_buffer})")
            
            return observation  # 返回观测对象以便后续使用
        
        def _backproject_depth_to_world(self, depth: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                                       agent_position: List[float], agent_rotation: List[float]) -> Optional[np.ndarray]:
            """
            使用深度图反投影bbox到世界坐标系
            
            Args:
                depth: 深度图 (H, W)，单位：米
                x1, y1, x2, y2: bbox像素坐标
                agent_position: 机器人位置 [x, y, z]
                agent_rotation: 机器人旋转四元数 [x, y, z, w]
            
            Returns:
                世界坐标系下的3D位置 [x, y, z] 或 None（如果失败）
            """
            try:
                depth_h, depth_w = depth.shape[0], depth.shape[1]
                
                # 检查深度图尺寸
                if depth_h != self.img_height or depth_w != self.img_width:
                    rag.log_warning(f"深度图尺寸 ({depth_w}x{depth_h}) 与内参尺寸 ({self.img_width}x{self.img_height}) 不一致")
                
                # 提取采样点：bbox中心 + 4个角点
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                sample_points = [
                    (cx, cy),  # 中心
                    (x1, y1),  # 左上
                    (x2, y1),  # 右上
                    (x1, y2),  # 左下
                    (x2, y2),  # 右下
                ]
                
                # 提取深度值
                depths = []
                for u, v in sample_points:
                    # 确保坐标在有效范围内
                    u_clamped = max(0, min(depth_w - 1, u))
                    v_clamped = max(0, min(depth_h - 1, v))
                    d = depth[v_clamped, u_clamped]
                    
                    # 检查深度值是否有效（通常在0.1米到10米之间）
                    if 0.1 <= d <= 10.0 and not np.isnan(d) and not np.isinf(d):
                        depths.append(d)
                    else:
                        rag.log_warning(f"空间融合模块：无效深度值 d={d:.3f}m 在 ({u_clamped}, {v_clamped})")
                
                # 如果有效深度值太少，返回None
                if len(depths) < 2:
                    rag.log_warning(f"空间融合模块：有效深度值不足 ({len(depths)}/5)，无法进行反投影")
                    return None
                
                # 计算平均深度（使用中位数更鲁棒）
                mean_depth = np.median(depths)
                
                # 使用bbox中心进行反投影
                u_center = cx
                v_center = cy
                
                # 反投影到相机坐标系
                # Xc = (u - cx) * d / fx
                # Yc = (v - cy) * d / fy
                # Zc = d
                Xc = (u_center - self.cx) * mean_depth / self.fx
                Yc = (v_center - self.cy) * mean_depth / self.fy
                Zc = mean_depth
                
                point_cam = np.array([Xc, Yc, Zc])
                
                # 转换到世界坐标系
                # 首先需要获取相机到世界的变换矩阵
                # Habitat中，agent的position和rotation表示agent在世界坐标系中的位姿
                # 相机相对于agent的位姿通常是固定的（在agent坐标系中）
                # 对于Habitat，相机通常位于agent前方，高度1.5米
                
                # 将四元数转换为旋转矩阵
                R_agent_to_world = self._quaternion_to_rotation_matrix(agent_rotation)
                t_agent_to_world = np.array(agent_position)
                
                # 相机在agent坐标系中的位置（通常为[0, 1.5, 0]）
                # 但这里我们假设相机与agent位置相同（简化处理）
                # 如果需要更精确，可以从sensor_spec获取相机相对位置
                point_world = R_agent_to_world @ point_cam + t_agent_to_world
                
                # 验证结果
                if np.any(np.isnan(point_world)) or np.any(np.isinf(point_world)):
                    rag.log_warning(f"空间融合模块：反投影结果包含NaN或Inf: {point_world}")
                    return None
                
                # 检查距离是否合理（不应该太远）
                distance = np.linalg.norm(point_world - t_agent_to_world)
                if distance > 20.0:  # 超过20米认为不合理
                    rag.log_warning(f"空间融合模块：反投影距离过远: {distance:.2f}m")
                    return None
                
                return point_world
                
            except Exception as e:
                rag.log_warning(f"空间融合模块：深度反投影异常: {e}")
                return None
        
        def _quaternion_to_rotation_matrix(self, quat: List[float]) -> np.ndarray:
            """将四元数转换为旋转矩阵"""
            x, y, z, w = quat
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
            ])
            return R
        
        def _pixel_to_ray(self, u_norm: float, v_norm: float, R: np.ndarray) -> np.ndarray:
            """
            将归一化像素坐标转换为世界坐标系下的射线方向
            
            Args:
                u_norm, v_norm: 归一化像素坐标 [0, 1]
                R: 相机到世界的旋转矩阵
            
            Returns:
                世界坐标系下的射线方向向量（单位向量）
            """
            # 转换为像素坐标（使用实际图像尺寸）
            u = u_norm * self.img_width
            v = v_norm * self.img_height
            
            # 相机坐标系下的射线方向
            x_cam = (u - self.cx) / self.fx
            y_cam = (v - self.cy) / self.fy
            z_cam = 1.0
            ray_cam = np.array([x_cam, y_cam, z_cam])
            ray_cam = ray_cam / np.linalg.norm(ray_cam)  # 归一化
            
            # 转换到世界坐标系
            ray_world = R @ ray_cam
            return ray_world / np.linalg.norm(ray_world)  # 确保单位向量
        
        def _point_to_ray_distance(self, point: np.ndarray, ray_origin: np.ndarray, ray_direction: np.ndarray) -> float:
            """
            计算点到射线的距离（残差）
            
            Args:
                point: 3D点 [x, y, z]
                ray_origin: 射线起点 [x, y, z]
                ray_direction: 射线方向向量（单位向量）[x, y, z]
            
            Returns:
                点到射线的距离（米）
            """
            # 计算点相对于射线起点的向量
            vec_to_point = point - ray_origin
            
            # 计算点在射线方向上的投影长度
            t = np.dot(vec_to_point, ray_direction)
            
            # 计算射线上最近的点
            closest_point_on_ray = ray_origin + t * ray_direction
            
            # 计算点到最近点的距离
            distance = np.linalg.norm(point - closest_point_on_ray)
            
            return distance
        
        def _triangulate_two_rays(self, pos1: np.ndarray, ray1: np.ndarray, 
                                  pos2: np.ndarray, ray2: np.ndarray) -> Optional[np.ndarray]:
            """
            使用线性最小二乘法三角化两条射线
            
            Args:
                pos1, pos2: 两条射线的起点
                ray1, ray2: 两条射线的方向向量（单位向量）
            
            Returns:
                估计的3D位置 [x, y, z] 或 None
            """
            try:
                # 构建线性系统：最小化 ||P - (pos1 + t1*ray1)||^2 + ||P - (pos2 + t2*ray2)||^2
                # 等价于：minimize ||(I - ray1*ray1^T)(P - pos1)||^2 + ||(I - ray2*ray2^T)(P - pos2)||^2
                
                ray1 = ray1.reshape(3, 1)
                ray2 = ray2.reshape(3, 1)
                
                proj1 = np.eye(3) - ray1 @ ray1.T
                proj2 = np.eye(3) - ray2 @ ray2.T
                
                A = np.vstack([proj1, proj2])
                b = np.hstack([proj1 @ pos1, proj2 @ pos2])
                
                # 使用SVD求解
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                
                if s[-1] < 1e-6:
                    return None
                
                point = Vt.T @ (U.T @ b / s)
                
                # 验证
                if np.any(np.isnan(point)) or np.any(np.isinf(point)):
                    return None
                
                return point
            except Exception:
                return None
        
        def _ransac_triangulate(self, positions: List[np.ndarray], rays: List[np.ndarray],
                               max_iterations: int = 100, inlier_threshold: float = 0.15,
                               min_inliers: int = 2) -> Tuple[Optional[np.ndarray], float, float]:
            """
            使用RANSAC进行鲁棒三角化
            
            Args:
                positions: 每条射线的起点（世界坐标）
                rays: 每条射线的方向向量（单位向量）
                max_iterations: 最大迭代次数
                inlier_threshold: 内点阈值（米）
                min_inliers: 最少内点数
            
            Returns:
                (best_point, inlier_ratio, confidence) 或 (None, 0.0, 0.0)
            """
            if len(positions) < 2:
                return None, 0.0, 0.0
            
            n_rays = len(positions)
            best_point = None
            best_inlier_count = 0
            best_inliers = []
            
            # RANSAC循环
            for iteration in range(max_iterations):
                # 随机选择2条射线
                if n_rays < 2:
                    break
                
                indices = np.random.choice(n_rays, size=2, replace=False)
                idx1, idx2 = indices[0], indices[1]
                
                # 三角化这两条射线
                candidate_point = self._triangulate_two_rays(
                    positions[idx1], rays[idx1],
                    positions[idx2], rays[idx2]
                )
                
                if candidate_point is None:
                    continue
                
                # 计算所有射线到候选点的残差
                inliers = []
                inlier_count = 0
                
                for i in range(n_rays):
                    distance = self._point_to_ray_distance(
                        candidate_point, positions[i], rays[i]
                    )
                    if distance < inlier_threshold:
                        inliers.append(i)
                        inlier_count += 1
                
                # 更新最佳结果
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_point = candidate_point.copy()
                    best_inliers = inliers
            
            # 如果没有找到足够的内点，返回None
            if best_inlier_count < min_inliers:
                rag.log_warning(f"RANSAC三角化：内点数不足 ({best_inlier_count} < {min_inliers})")
                return None, 0.0, 0.0
            
            # 使用所有内点重新三角化（提高精度）
            if len(best_inliers) >= 2:
                # 使用所有内点进行最终三角化
                inlier_positions = [positions[i] for i in best_inliers]
                inlier_rays = [rays[i] for i in best_inliers]
                
                # 使用最小二乘法三角化所有内点
                refined_point = self._triangulate_rays(inlier_positions, inlier_rays)
                if refined_point is not None:
                    best_point = refined_point
            
            # 计算内点比例和置信度
            inlier_ratio = best_inlier_count / n_rays
            
            # 置信度计算：基于内点比例和观测数量
            # confidence = inlier_ratio * (1 - exp(-num_views / 3))
            num_views = n_rays
            view_factor = 1.0 - np.exp(-num_views / 3.0)
            confidence = inlier_ratio * view_factor
            
            rag.log_info(f"RANSAC三角化：找到 {best_inlier_count}/{n_rays} 内点，"
                        f"内点比例: {inlier_ratio:.3f}, 置信度: {confidence:.3f}")
            
            return best_point, inlier_ratio, confidence
        
        def _triangulate_rays(self, positions: List[np.ndarray], rays: List[np.ndarray]) -> Optional[np.ndarray]:
            """
            使用最小二乘法求解多条射线的最接近交点
            
            方法：对于每条射线 P_i + t_i * ray_i，寻找点P使得：
            minimize sum_i ||P - (P_i + t_i * ray_i)||^2
            
            这等价于求解线性系统，使用SVD求解
            
            Args:
                positions: 每条射线的起点（世界坐标）
                rays: 每条射线的方向向量（单位向量）
            
            Returns:
                估计的3D位置 [x, y, z] 或 None
            """
            if len(positions) < self.min_views:
                return None
            
            try:
                # 使用更稳定的方法：对于每条射线，构建约束
                # 对于射线 i: P = pos_i + t_i * ray_i
                # 最小化 sum_i ||(P - pos_i) - ray_i * ray_i^T * (P - pos_i)||^2
                # 这等价于：minimize sum_i ||(I - ray_i * ray_i^T) * (P - pos_i)||^2
                
                # 构建线性系统 A * P = b
                A_list = []
                b_list = []
                
                for pos, ray in zip(positions, rays):
                    # 投影矩阵：I - ray * ray^T
                    ray = ray.reshape(3, 1)
                    proj = np.eye(3) - ray @ ray.T
                    A_list.append(proj)
                    b_list.append(proj @ pos)
                
                # 组合所有约束
                A = np.vstack(A_list)
                b = np.hstack(b_list)
                
                # 使用SVD求解最小二乘问题
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                
                # 检查条件数
                if s[-1] < 1e-6:
                    rag.log_warning("三角化系统条件数过小，可能共线")
                    return None
                
                # 求解
                estimated_pos = Vt.T @ (U.T @ b / s)
                
                # 验证：检查估计位置是否合理
                if np.any(np.isnan(estimated_pos)) or np.any(np.isinf(estimated_pos)):
                    return None
                
                # 检查估计位置是否在合理范围内（距离所有观测点不太远）
                max_distance = 10.0  # 最大距离10米
                for pos in positions:
                    dist = np.linalg.norm(estimated_pos - pos)
                    if dist > max_distance:
                        rag.log_warning(f"估计位置距离观测点过远: {dist:.2f}m")
                        return None
                
                return estimated_pos
            except Exception as e:
                rag.log_warning(f"三角化求解失败: {e}")
                return None
        
        def _log_fusion_trace(self, method: str, estimated_position: Optional[np.ndarray],
                            confidence: float, inlier_ratio: Optional[float] = None,
                            anomaly_flags: List[str] = None, observations_data: List[Dict] = None,
                            entropy: Optional[float] = None):
            """
            记录融合过程到fusion_trace.json
            
            Args:
                method: 融合方法 ("depth", "triangulation", "voxel_vote")
                estimated_position: 估计的3D位置
                confidence: 置信度
                inlier_ratio: 内点比例（仅用于triangulation）
                anomaly_flags: 异常标志列表
                observations_data: 观测数据列表
                entropy: 熵值（仅用于voxel_vote）
            """
            if not self.fusion_trace_file:
                return
            
            try:
                # 读取现有日志
                trace_entries = []
                if os.path.exists(self.fusion_trace_file):
                    try:
                        with open(self.fusion_trace_file, "r", encoding="utf-8") as f:
                            trace_entries = json.load(f)
                        if not isinstance(trace_entries, list):
                            trace_entries = []
                    except (json.JSONDecodeError, IOError) as e:
                        rag.log_warning(f"读取融合追踪日志失败，创建新文件: {e}")
                        trace_entries = []
                
                # 准备观测数据
                observations = []
                if observations_data is None:
                    # 从buffer提取观测数据
                    for obs in self.buffer:
                        obs_type = obs.get("type", "triangulation")
                        pos = obs.get("position", [0, 0, 0])
                        rot = obs.get("rotation", [0, 0, 0, 1])
                        bbox_center = obs.get("bbox_center", [0.5, 0.5])
                        conf = obs.get("confidence", 0.0)
                        world_pos = obs.get("world_position")
                        
                        # 计算射线方向（如果是triangulation类型）
                        ray_dir = None
                        if obs_type == "triangulation":
                            R = self._quaternion_to_rotation_matrix(rot)
                            ray_dir = self._pixel_to_ray(bbox_center[0], bbox_center[1], R)
                            ray_dir = ray_dir.tolist()
                        
                        # 提取深度信息（如果有）
                        depth = None
                        if world_pos is not None:
                            # 计算深度（从agent位置到世界位置的距离）
                            depth = np.linalg.norm(np.array(world_pos) - np.array(pos))
                        
                        # 提取bbox（如果有）
                        bbox = None
                        if "bbox" in obs:
                            bbox = obs["bbox"]
                        elif bbox_center:
                            # 从bbox_center推断bbox（近似）
                            cx, cy = bbox_center[0] * self.img_width, bbox_center[1] * self.img_height
                            bbox_size = 50  # 假设bbox大小为50像素
                            bbox = [
                                cx - bbox_size,
                                cy - bbox_size,
                                cx + bbox_size,
                                cy + bbox_size
                            ]
                        
                        observations.append({
                            "ray_origin": pos,
                            "ray_dir": ray_dir,
                            "depth": float(depth) if depth is not None else None,
                            "bbox": bbox,
                            "type": obs_type,
                            "confidence": float(conf),
                            "world_position": world_pos
                        })
                else:
                    observations = observations_data
                
                # 构建日志条目
                trace_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "n_observations": len(observations),
                    "observations": observations,
                    "method": method,
                    "estimated_position": estimated_position.tolist() if estimated_position is not None else None,
                    "confidence": float(confidence),
                    "inlier_ratio": float(inlier_ratio) if inlier_ratio is not None else None,
                    "entropy": float(entropy) if entropy is not None else None,
                    "anomaly_flags": anomaly_flags if anomaly_flags else []
                }
                
                # 追加到日志
                trace_entries.append(trace_entry)
                
                # 写入文件（使用临时文件确保原子性）
                temp_file = self.fusion_trace_file + ".tmp"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(trace_entries, f, ensure_ascii=False, indent=2)
                
                # 原子性替换
                os.replace(temp_file, self.fusion_trace_file)
                
                rag.log_info(f"融合追踪：已记录 {method} 方法，观测数: {len(observations)}, 置信度: {confidence:.3f}")
                
            except Exception as e:
                rag.log_warning(f"记录融合追踪日志失败: {e}")
        
        def _voxel_vote_localize(self, candidate_points: List[np.ndarray], 
                                candidate_confidences: List[float],
                                reference_point: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], float, float]:
            """
            使用微体素投票进行3D定位
            
            Args:
                candidate_points: 候选3D点列表
                candidate_confidences: 对应的置信度列表
                reference_point: 参考点（用于确定体素网格中心），如果为None则使用候选点的中心
            
            Returns:
                (estimated_position, confidence, entropy)
                - estimated_position: 估计的3D位置（最高投票体素的中心）
                - confidence: 融合后的置信度
                - entropy: 直方图熵（不确定性度量）
            """
            if len(candidate_points) == 0:
                return None, 0.0, 1.0
            
            candidate_points = np.array(candidate_points)
            
            # 确定体素网格中心
            if reference_point is not None:
                grid_center = np.array(reference_point)
            else:
                # 使用候选点的加权中心
                if len(candidate_confidences) > 0:
                    weights = np.array(candidate_confidences)
                    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
                    grid_center = np.average(candidate_points, axis=0, weights=weights)
                else:
                    grid_center = np.mean(candidate_points, axis=0)
            
            # 计算体素网格范围
            grid_half_width = self.grid_width / 2.0
            grid_half_height = self.grid_height / 2.0
            grid_half_depth = self.grid_height / 2.0  # 使用相同的高度作为深度
            
            # 计算体素网格尺寸
            n_voxels_x = int(np.ceil(self.grid_width / self.voxel_size))
            n_voxels_y = int(np.ceil(self.grid_height / self.voxel_size))
            n_voxels_z = int(np.ceil(self.grid_height / self.voxel_size))
            
            # 初始化体素直方图
            voxel_counts = np.zeros((n_voxels_x, n_voxels_y, n_voxels_z), dtype=np.float32)
            
            # 将每个候选点量化到体素并投票
            for i, point in enumerate(candidate_points):
                # 计算相对于网格中心的偏移
                offset = point - grid_center
                
                # 计算体素索引
                voxel_x = int((offset[0] + grid_half_width) / self.voxel_size)
                voxel_y = int((offset[1] + grid_half_height) / self.voxel_size)
                voxel_z = int((offset[2] + grid_half_depth) / self.voxel_size)
                
                # 检查是否在网格范围内
                if (0 <= voxel_x < n_voxels_x and 
                    0 <= voxel_y < n_voxels_y and 
                    0 <= voxel_z < n_voxels_z):
                    # 使用置信度作为投票权重
                    weight = candidate_confidences[i] if i < len(candidate_confidences) else 1.0
                    voxel_counts[voxel_x, voxel_y, voxel_z] += weight
            
            # 找到最高投票的体素
            max_count = np.max(voxel_counts)
            if max_count == 0:
                return None, 0.0, 1.0
            
            # 找到所有最大投票的体素（处理平局）
            max_indices = np.where(voxel_counts == max_count)
            if len(max_indices[0]) > 0:
                # 如果有多个最大投票体素，选择第一个
                best_voxel_x = max_indices[0][0]
                best_voxel_y = max_indices[1][0]
                best_voxel_z = max_indices[2][0]
            else:
                return None, 0.0, 1.0
            
            # 计算体素中心位置
            voxel_center_x = grid_center[0] - grid_half_width + (best_voxel_x + 0.5) * self.voxel_size
            voxel_center_y = grid_center[1] - grid_half_height + (best_voxel_y + 0.5) * self.voxel_size
            voxel_center_z = grid_center[2] - grid_half_depth + (best_voxel_z + 0.5) * self.voxel_size
            estimated_position = np.array([voxel_center_x, voxel_center_y, voxel_center_z])
            
            # 计算置信度（归一化的最大投票数）
            total_votes = np.sum(voxel_counts)
            confidence = max_count / total_votes if total_votes > 0 else 0.0
            
            # 计算熵（不确定性）
            # 归一化直方图
            normalized_counts = voxel_counts / total_votes if total_votes > 0 else voxel_counts
            # 避免log(0)
            normalized_counts = normalized_counts + 1e-10
            # 计算熵: H = -sum(p * log(p))
            entropy = -np.sum(normalized_counts * np.log(normalized_counts))
            # 归一化熵到[0, 1]范围（相对于最大熵log(n_voxels)）
            max_entropy = np.log(n_voxels_x * n_voxels_y * n_voxels_z)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
            
            rag.log_info(f"微体素定位：网格中心={grid_center.tolist()}, "
                        f"最高投票体素=({best_voxel_x}, {best_voxel_y}, {best_voxel_z}), "
                        f"投票数={max_count:.1f}/{total_votes:.1f}, "
                        f"置信度={confidence:.3f}, 熵={normalized_entropy:.3f}")
            
            return estimated_position, confidence, normalized_entropy
        
        def _estimate_with_voxel_vote(self) -> Optional[np.ndarray]:
            """
            使用体素投票模式估计3D位置
            
            收集所有候选点：
            1. 深度反投影点
            2. 三角化结果点
            
            然后使用微体素投票融合
            """
            candidate_points = []
            candidate_confidences = []
            anomaly_flags = []
            
            # 1. 收集深度反投影点
            depth_positions = []
            depth_confidences = []
            
            for obs in self.buffer:
                obs_type = obs.get("type", "triangulation")
                world_pos = obs.get("world_position")
                conf = obs.get("confidence", 0.0)
                
                if obs_type == "depth" and world_pos is not None:
                    depth_positions.append(np.array(world_pos))
                    depth_confidences.append(conf)
                    candidate_points.append(np.array(world_pos))
                    candidate_confidences.append(conf)
            
            # 2. 收集三角化候选点（通过RANSAC）
            if len(self.buffer) >= 2:
                # 提取所有观测的位置和射线
                positions = []
                rays = []
                confidences = []
                
                for i, obs in enumerate(self.buffer):
                    pos = np.array(obs["position"])
                    rot_quat = obs["rotation"]
                    bbox_center = obs["bbox_center"]
                    conf = obs.get("confidence", 0.0)
                    
                    # 转换旋转矩阵
                    R = self._quaternion_to_rotation_matrix(rot_quat)
                    
                    # 计算射线方向
                    ray = self._pixel_to_ray(bbox_center[0], bbox_center[1], R)
                    
                    # 检查射线是否与之前的估计差异过大
                    if self.last_reliable_estimate is not None:
                        last_est = np.array(self.last_reliable_estimate)
                        distance_to_ray = self._point_to_ray_distance(last_est, pos, ray)
                        
                        if distance_to_ray > 2.0:
                            anomaly_flags.append(f"ray_{i}_diverged")
                            continue
                    
                    positions.append(pos)
                    rays.append(ray)
                    confidences.append(conf)
                
                # 使用RANSAC三角化获取候选点
                if len(positions) >= 2:
                    triangulated_pos, inlier_ratio, ransac_confidence = self._ransac_triangulate(
                        positions, rays,
                        max_iterations=100,
                        inlier_threshold=0.15,
                        min_inliers=2
                    )
                    
                    if triangulated_pos is not None:
                        # 将三角化结果作为候选点
                        candidate_points.append(triangulated_pos)
                        # 使用RANSAC置信度和平均视觉置信度的乘积
                        visual_conf = np.mean(confidences) if len(confidences) > 0 else 0.5
                        candidate_confidences.append(ransac_confidence * visual_conf)
                        
                        if inlier_ratio < 0.3:
                            anomaly_flags.append("low_inlier_ratio")
            
            # 3. 使用体素投票融合所有候选点
            if len(candidate_points) == 0:
                rag.log_warning("微体素定位：没有候选点可用")
                return None
            
            # 确定参考点（使用最后一个可靠估计或深度点的中心）
            reference_point = None
            if self.last_reliable_estimate is not None:
                reference_point = np.array(self.last_reliable_estimate)
            elif len(depth_positions) > 0:
                if len(depth_confidences) > 0:
                    weights = np.array(depth_confidences)
                    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
                    reference_point = np.average(depth_positions, axis=0, weights=weights)
                else:
                    reference_point = np.mean(depth_positions, axis=0)
            
            # 执行体素投票
            estimated_pos, voxel_confidence, entropy = self._voxel_vote_localize(
                candidate_points, candidate_confidences, reference_point
            )
            
            if estimated_pos is None:
                return None
            
            # 检查与上一个可靠估计的距离
            is_suspect = False
            if self.last_reliable_estimate is not None:
                distance_to_last = np.linalg.norm(estimated_pos - np.array(self.last_reliable_estimate))
                if distance_to_last > 2.0:
                    is_suspect = True
                    anomaly_flags.append("diverged_from_last_estimate")
                    rag.log_warning(f"微体素定位：估计与上一个可靠估计差异过大 "
                                  f"(distance={distance_to_last:.3f}m > 2.0m)")
                    # 降低置信度
                    voxel_confidence *= 0.5
            
            # 使用熵调整置信度（高熵 = 低置信度）
            entropy_penalty = 1.0 - entropy * 0.5  # 最多降低50%置信度
            final_confidence = voxel_confidence * entropy_penalty
            
            self.estimated_position = estimated_pos
            self.estimated_confidence = final_confidence
            
            # 如果置信度足够高且不可疑，更新可靠估计
            if final_confidence >= 0.5 and not is_suspect:
                self.last_reliable_estimate = estimated_pos.tolist()
                self.last_reliable_confidence = final_confidence
                rag.log_info(f"微体素定位：更新可靠估计 ({estimated_pos[0]:.3f}, "
                           f"{estimated_pos[1]:.3f}, {estimated_pos[2]:.3f}), "
                           f"置信度: {final_confidence:.3f}, 熵: {entropy:.3f}")
            
            rag.log_info(f"微体素定位：估计3D位置 ({estimated_pos[0]:.3f}, {estimated_pos[1]:.3f}, {estimated_pos[2]:.3f}), "
                        f"置信度: {final_confidence:.3f}, 熵: {entropy:.3f}, "
                        f"候选点数: {len(candidate_points)} (深度: {len(depth_positions)}, 三角化: {len(candidate_points) - len(depth_positions)})")
            
            # 记录融合追踪日志
            self._log_fusion_trace(
                method="voxel_vote",
                estimated_position=estimated_pos,
                confidence=final_confidence,
                inlier_ratio=None,  # 体素投票不使用inlier_ratio
                anomaly_flags=anomaly_flags,
                observations_data=None,  # 使用默认的观测数据提取
                entropy=entropy  # 传递熵值
            )
            
            return estimated_pos
        
        def estimate_object_position(self) -> Optional[np.ndarray]:
            """
            估计目标物体的3D位置
            
            根据fusion_mode选择融合策略：
            - "depth": 优先使用深度反投影
            - "triangulate": 使用RANSAC三角化
            - "voxel_vote": 使用微体素投票融合所有候选点
            
            Returns:
                估计的3D位置 [x, y, z] 或 None
            """
            if len(self.buffer) == 0:
                return None
            
            # 如果使用体素投票模式，收集所有候选点
            if self.fusion_mode == "voxel_vote":
                return self._estimate_with_voxel_vote()
            
            # 优先使用深度观测
            depth_positions = []
            depth_confidences = []
            
            for obs in self.buffer:
                obs_type = obs.get("type", "triangulation")
                world_pos = obs.get("world_position")
                conf = obs.get("confidence", 0.0)
                
                if obs_type == "depth" and world_pos is not None:
                    depth_positions.append(np.array(world_pos))
                    depth_confidences.append(conf)
            
            # 如果有深度观测，使用深度观测（加权平均）
            if len(depth_positions) > 0:
                if len(depth_positions) == 1:
                    # 只有一个深度观测，直接使用
                    estimated_pos = depth_positions[0]
                    self.estimated_confidence = depth_confidences[0]
                else:
                    # 多个深度观测，使用加权平均（按置信度加权）
                    weights = np.array(depth_confidences)
                    weights = weights / weights.sum()  # 归一化
                    estimated_pos = np.average(depth_positions, axis=0, weights=weights)
                    self.estimated_confidence = np.mean(depth_confidences)
                
                # 检查与上一个可靠估计的距离
                is_suspect = False
                if self.last_reliable_estimate is not None:
                    distance_to_last = np.linalg.norm(estimated_pos - np.array(self.last_reliable_estimate))
                    if distance_to_last > 2.0:  # 2米阈值
                        is_suspect = True
                        rag.log_warning(f"空间融合模块：深度反投影估计与上一个可靠估计差异过大 "
                                      f"(distance={distance_to_last:.3f}m > 2.0m), "
                                      f"新估计: {estimated_pos.tolist()}, 上一个: {self.last_reliable_estimate}")
                        # 降低置信度
                        self.estimated_confidence *= 0.5
                
                self.estimated_position = estimated_pos
                
                # 如果置信度足够高且不可疑，更新可靠估计
                if self.estimated_confidence >= 0.5 and not is_suspect:
                    self.last_reliable_estimate = estimated_pos.tolist()
                    self.last_reliable_confidence = self.estimated_confidence
                    rag.log_info(f"空间融合模块：更新可靠估计（深度反投影） ({estimated_pos[0]:.3f}, "
                               f"{estimated_pos[1]:.3f}, {estimated_pos[2]:.3f}), "
                               f"置信度: {self.estimated_confidence:.3f}")
                
                rag.log_info(f"空间融合模块：使用深度反投影估计3D位置 ({estimated_pos[0]:.3f}, {estimated_pos[1]:.3f}, {estimated_pos[2]:.3f}), "
                           f"平均置信度: {self.estimated_confidence:.3f} (深度观测数: {len(depth_positions)})")
                
                # 记录融合追踪日志
                anomaly_flags = []
                if is_suspect:
                    anomaly_flags.append("diverged_from_last_estimate")
                self._log_fusion_trace(
                    method="depth",
                    estimated_position=estimated_pos,
                    confidence=self.estimated_confidence,
                    inlier_ratio=None,
                    anomaly_flags=anomaly_flags
                )
                
                return estimated_pos
            
            # 如果没有深度观测，使用RANSAC三角化
            if len(self.buffer) < 2:
                rag.log_info(f"空间融合模块：观测数不足 ({len(self.buffer)} < 2)，无法进行三角化")
                return None
            
            # 提取所有观测的位置和射线（用于三角化）
            positions = []
            rays = []
            confidences = []
            rejected_rays = []  # 被拒绝的射线索引和原因
            
            for i, obs in enumerate(self.buffer):
                pos = np.array(obs["position"])
                rot_quat = obs["rotation"]
                bbox_center = obs["bbox_center"]
                conf = obs.get("confidence", 0.0)
                
                # 转换旋转矩阵
                R = self._quaternion_to_rotation_matrix(rot_quat)
                
                # 计算射线方向
                ray = self._pixel_to_ray(bbox_center[0], bbox_center[1], R)
                
                # 检查射线是否与之前的估计差异过大
                if self.last_reliable_estimate is not None:
                    # 计算射线到上一个可靠估计的距离
                    last_est = np.array(self.last_reliable_estimate)
                    distance_to_ray = self._point_to_ray_distance(last_est, pos, ray)
                    
                    # 如果距离过大（>2米），标记为可疑
                    if distance_to_ray > 2.0:
                        rejected_rays.append({
                            "index": i,
                            "reason": f"ray_diverged_from_last_estimate (distance={distance_to_ray:.3f}m > 2.0m)",
                            "distance": distance_to_ray,
                            "position": pos.tolist(),
                            "ray_direction": ray.tolist()
                        })
                        rag.log_warning(f"空间融合模块：拒绝射线 {i} - 与上一个可靠估计差异过大 "
                                      f"(distance={distance_to_ray:.3f}m > 2.0m)")
                        continue  # 跳过这条射线
                
                positions.append(pos)
                rays.append(ray)
                confidences.append(conf)
            
            # 如果所有射线都被拒绝，记录警告
            if len(positions) == 0 and len(rejected_rays) > 0:
                rag.log_warning(f"空间融合模块：所有射线都被拒绝 ({len(rejected_rays)} 条), "
                              f"拒绝原因: {[r['reason'] for r in rejected_rays]}")
                # 回退到最后一个可靠估计
                if self.last_reliable_estimate is not None:
                    rag.log_warning(f"空间融合模块：所有射线异常，回退到最后一个可靠估计")
                    self.estimated_position = np.array(self.last_reliable_estimate)
                    self.estimated_confidence = self.last_reliable_confidence * 0.7
                    
                    # 记录融合追踪日志（回退情况）
                    self._log_fusion_trace(
                        method="triangulation",
                        estimated_position=self.estimated_position,
                        confidence=self.estimated_confidence,
                        inlier_ratio=0.0,
                        anomaly_flags=["all_rays_rejected", "fallback_to_last_estimate"]
                    )
                    
                    return self.estimated_position
                
                # 记录融合追踪日志（完全失败）
                self._log_fusion_trace(
                    method="triangulation",
                    estimated_position=None,
                    confidence=0.0,
                    inlier_ratio=0.0,
                    anomaly_flags=["all_rays_rejected", "no_fallback_available"]
                )
                
                return None
            
            # 使用RANSAC三角化
            estimated_pos, inlier_ratio, ransac_confidence = self._ransac_triangulate(
                positions, rays,
                max_iterations=100,
                inlier_threshold=0.15,  # 15cm内点阈值
                min_inliers=2
            )
            
            # 失败阈值：如果内点比例太低，认为需要更多观测
            fail_threshold = 0.3  # 30%内点比例阈值
            
            if estimated_pos is not None:
                # 检查新估计是否与之前的估计或记忆位置差异过大
                is_suspect = False
                suspect_reason = ""
                
                # 检查与上一个可靠估计的距离
                if self.last_reliable_estimate is not None:
                    distance_to_last = np.linalg.norm(estimated_pos - np.array(self.last_reliable_estimate))
                    if distance_to_last > 2.0:  # 2米阈值
                        is_suspect = True
                        suspect_reason = f"diverged_from_last_estimate (distance={distance_to_last:.3f}m > 2.0m)"
                        rag.log_warning(f"空间融合模块：新估计与上一个可靠估计差异过大 - {suspect_reason}, "
                                      f"新估计: {estimated_pos.tolist()}, 上一个: {self.last_reliable_estimate}")
                
                # 检查内点比例是否足够
                if inlier_ratio < fail_threshold:
                    is_suspect = True
                    if suspect_reason:
                        suspect_reason += f", low_inlier_ratio ({inlier_ratio:.3f} < {fail_threshold:.3f})"
                    else:
                        suspect_reason = f"low_inlier_ratio ({inlier_ratio:.3f} < {fail_threshold:.3f})"
                    rag.log_warning(f"空间融合模块：RANSAC内点比例过低 ({inlier_ratio:.3f} < {fail_threshold:.3f})，"
                                  f"建议收集更多观测以提高精度")
                
                # 如果可疑，但不回退（仍然使用，但置信度会较低）
                if is_suspect:
                    rag.log_warning(f"空间融合模块：标记为可疑估计 - {suspect_reason}")
                    # 降低置信度
                    ransac_confidence *= 0.5
                
                # 融合RANSAC置信度和视觉置信度
                visual_confidence = np.mean(confidences)
                # 最终置信度 = RANSAC置信度 * 视觉置信度
                final_confidence = ransac_confidence * visual_confidence
                
                # 如果置信度足够高，更新可靠估计
                if final_confidence >= 0.5 and not is_suspect:
                    self.last_reliable_estimate = estimated_pos.tolist()
                    self.last_reliable_confidence = final_confidence
                    rag.log_info(f"空间融合模块：更新可靠估计 ({estimated_pos[0]:.3f}, {estimated_pos[1]:.3f}, {estimated_pos[2]:.3f}), "
                               f"置信度: {final_confidence:.3f}")
                
                self.estimated_position = estimated_pos
                self.estimated_confidence = final_confidence
                
                rag.log_info(f"空间融合模块：使用RANSAC三角化估计3D位置 ({estimated_pos[0]:.3f}, {estimated_pos[1]:.3f}, {estimated_pos[2]:.3f}), "
                           f"内点比例: {inlier_ratio:.3f}, RANSAC置信度: {ransac_confidence:.3f}, "
                           f"最终置信度: {self.estimated_confidence:.3f} (观测数: {len(positions)})")
                
                # 记录融合追踪日志
                anomaly_flags = []
                if is_suspect:
                    if "diverged_from_last_estimate" in suspect_reason:
                        anomaly_flags.append("diverged_from_last_estimate")
                    if "low_inlier_ratio" in suspect_reason:
                        anomaly_flags.append("low_inlier_ratio")
                if len(rejected_rays) > 0:
                    anomaly_flags.append(f"rejected_rays_count_{len(rejected_rays)}")
                
                self._log_fusion_trace(
                    method="triangulation",
                    estimated_position=estimated_pos,
                    confidence=self.estimated_confidence,
                    inlier_ratio=inlier_ratio,
                    anomaly_flags=anomaly_flags
                )
                
                return estimated_pos
            else:
                # 三角化失败，检查是否需要回退
                rag.log_warning(f"空间融合模块：RANSAC三角化失败，可能需要更多观测或检查观测质量")
                
                # 如果所有新观测都是可疑的，回退到最后一个可靠估计
                if self.last_reliable_estimate is not None and self.anomaly_count >= 3:
                    rag.log_warning(f"空间融合模块：连续 {self.anomaly_count} 个异常观测，回退到最后一个可靠估计")
                    self.estimated_position = np.array(self.last_reliable_estimate)
                    self.estimated_confidence = self.last_reliable_confidence * 0.8  # 降低置信度
                    rag.log_info(f"空间融合模块：回退到可靠估计 ({self.estimated_position[0]:.3f}, "
                               f"{self.estimated_position[1]:.3f}, {self.estimated_position[2]:.3f}), "
                               f"置信度: {self.estimated_confidence:.3f}")
                    
                    # 记录融合追踪日志（回退情况）
                    self._log_fusion_trace(
                        method="triangulation",
                        estimated_position=self.estimated_position,
                        confidence=self.estimated_confidence,
                        inlier_ratio=0.0,
                        anomaly_flags=["ransac_failed", "high_anomaly_count", "fallback_to_last_estimate"]
                    )
                    
                    return self.estimated_position
                
                # 请求多角度重新扫描
                rag.log_warning(f"空间融合模块：建议执行多角度重新扫描以提高定位精度")
                
                # 记录融合追踪日志（完全失败）
                self._log_fusion_trace(
                    method="triangulation",
                    estimated_position=None,
                    confidence=0.0,
                    inlier_ratio=0.0,
                    anomaly_flags=["ransac_failed", "no_fallback_available", "suggest_multi_angle_rescan"]
                )
                
                return None
        
        def get_fused_confidence(self, vision_confidence: float, memory_position: Optional[np.ndarray] = None) -> float:
            """
            计算融合后的置信度
            
            Args:
                vision_confidence: 视觉检测置信度
                memory_position: 记忆库中的历史位置（可选）
            
            Returns:
                融合后的置信度
            """
            if self.estimated_position is None:
                return vision_confidence
            
            # 基础融合：视觉置信度
            C_final = self.alpha * vision_confidence
            
            # 如果有历史位置，添加空间一致性项
            if memory_position is not None:
                pos_mem = np.array(memory_position)
                pos_est = self.estimated_position
                distance = np.linalg.norm(pos_est - pos_mem)
                spatial_consistency = np.exp(-distance / self.lambda_spatial)
                C_final += (1 - self.alpha) * spatial_consistency
            else:
                # 如果没有历史位置，使用估计位置的一致性（多视角一致性）
                C_final += (1 - self.alpha) * self.estimated_confidence
            
            return min(1.0, C_final)
        
        def clear_buffer(self):
            """清空缓冲区"""
            self.buffer.clear()
            self.estimated_position = None
            self.estimated_confidence = 0.0

    def _check_too_close(vision_result: Dict, img_shape: Tuple[int, int, int], max_area_frac: float = 0.75, min_objects: int = 2) -> Tuple[bool, float]:
        """
        检测物体是否过近
        返回: (是否过近, 最大物体面积占比)
        """
        objs = vision_result.get("objects", []) if isinstance(vision_result, dict) else []
        if not objs:
            return False, 0.0
        
        H = img_shape[0] if (img_shape is not None and len(img_shape) >= 2) else None
        W = img_shape[1] if (img_shape is not None and len(img_shape) >= 2) else None
        
        if not H or not W:
            return False, 0.0
        
        max_area = 0.0
        for obj in objs:
            bbox = obj.get("bbox")
            # 使用统一的bbox归一化函数
            bbox_result = normalize_and_validate_bbox(bbox, W, H)
            if bbox_result is None:
                continue
            x1, y1, x2, y2, area_frac = bbox_result
            max_area = max(max_area, area_frac)
        
        # 判断是否过近：
        # 1. 最大物体面积占比超过阈值（默认75%）
        # 2. 检测到的物体数量少于最小值（可能因为太近导致只能看到部分）
        too_close = (max_area > max_area_frac) or (len(objs) < min_objects and max_area > 0.5)
        return too_close, max_area

    def _detect_unexplored_areas(depth: np.ndarray, max_depth: float = 5.0, min_depth: float = 0.5) -> Tuple[bool, float]:
        """
        基于深度图检测未探索区域
        返回: (是否有未探索区域, 未探索区域占比)
        """
        if depth is None or depth.size == 0:
            return False, 0.0
        
        # 深度图中的有效像素（在合理范围内）
        valid_mask = (depth >= min_depth) & (depth <= max_depth)
        # 未探索区域：深度值过大（可能是开放空间）或无效
        unexplored_mask = (depth > max_depth) | (~valid_mask)
        
        unexplored_ratio = np.sum(unexplored_mask) / depth.size
        has_unexplored = unexplored_ratio > 0.15  # 如果超过15%的区域未探索
        
        return has_unexplored, unexplored_ratio

    def _save_object_crop(rgb: np.ndarray, bbox: List[float], out_path: str, annotated_out: str = None):
        try:
            from PIL import Image, ImageDraw, ImageFont
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            h, w = rgb.shape[0], rgb.shape[1]
            
            # 检查图像尺寸有效性
            if w <= 0 or h <= 0:
                rag.log_warning(f"保存物体裁剪失败：无效的图像尺寸 {w}x{h}")
                return
            
            # 使用统一的bbox归一化函数
            bbox_result = normalize_and_validate_bbox(bbox, w, h)
            if bbox_result is None:
                rag.log_warning(f"保存物体裁剪失败：无效的bbox {bbox}")
                return
            
            x1, y1, x2, y2, area_ratio = bbox_result
            
            # 计算原始尺寸
            box_w = x2 - x1
            box_h = y2 - y1
            
            # 防御性检查：确保box_w和box_h有效
            if box_w <= 0 or box_h <= 0:
                rag.log_warning(f"保存物体裁剪失败：bbox尺寸无效 {box_w}x{box_h}")
                return
            
            # 动态边距：根据物体尺寸调整，最小边距20像素，最大边距50像素
            # 边距为物体尺寸的15%，但不超过最大边距，且确保至少为最小值
            pad_w = min(max(box_w * 0.15, 20), 50) if box_w > 0 else 20
            pad_h = min(max(box_h * 0.15, 20), 50) if box_h > 0 else 20
            
            # 应用边距（在归一化后的坐标基础上）
            x1_padded = max(0, x1 - int(pad_w))
            y1_padded = max(0, y1 - int(pad_h))
            x2_padded = min(w, x2 + int(pad_w))
            y2_padded = min(h, y2 + int(pad_h))
            
            # 确保最小尺寸（但不超过图像尺寸）
            min_size = 32
            if (x2_padded - x1_padded) < min_size:
                # 如果图像宽度小于min_size，使用图像宽度
                actual_min_w = min(min_size, w)
                cx = 0.5 * (x1_padded + x2_padded)
                x1_padded = max(0, int(cx - actual_min_w / 2))
                x2_padded = min(w, int(cx + actual_min_w / 2))
            if (y2_padded - y1_padded) < min_size:
                # 如果图像高度小于min_size，使用图像高度
                actual_min_h = min(min_size, h)
                cy = 0.5 * (y1_padded + y2_padded)
                y1_padded = max(0, int(cy - actual_min_h / 2))
                y2_padded = min(h, int(cy + actual_min_h / 2))
            
            # 最终边界检查（确保有效）
            x1_final = max(0, min(w - 1, x1_padded))
            y1_final = max(0, min(h - 1, y1_padded))
            x2_final = max(x1_final + 1, min(w, x2_padded))
            y2_final = max(y1_final + 1, min(h, y2_padded))
            
            # 关键验证：确保最终坐标有效（防止空裁剪或越界）
            if x2_final <= x1_final or y2_final <= y1_final:
                rag.log_warning(f"保存物体裁剪失败：无效的最终裁剪区域 ({x1_final},{y1_final},{x2_final},{y2_final}), 图像尺寸: {w}x{h}")
                return
            
            # 使用最终坐标
            x1, y1, x2, y2 = x1_final, y1_final, x2_final, y2_final
            
            # 保存裁剪图
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                rag.log_warning(f"保存物体裁剪失败：裁剪区域为空 ({x1},{y1},{x2},{y2})")
                return
            Image.fromarray(crop).save(out_path)
            
            # 保存带框注释图（使用更粗的框线）
            if annotated_out:
                # 仅在需要时复制图像（优化内存使用）
                img = Image.fromarray(rgb.copy())
                draw = ImageDraw.Draw(img)
                
                # 绘制更粗的框（多层叠加效果）
                box_width = max(5, int(min(w, h) * 0.01))  # 框线宽度为图像尺寸的1%
                for i in range(box_width):
                    offset = i - box_width // 2
                    # 修复：clamp坐标防止越界
                    x1_draw = max(0, x1 + offset)
                    y1_draw = max(0, y1 + offset)
                    x2_draw = min(w, x2 - 1 - offset)
                    y2_draw = min(h, y2 - 1 - offset)
                    # 确保绘制区域有效
                    if x2_draw > x1_draw and y2_draw > y1_draw:
                        draw.rectangle(
                            [x1_draw, y1_draw, x2_draw, y2_draw],
                            outline=(255, 0, 0),
                            width=1
                        )
                
                # 可选：添加标签文字（如果有空间）
                if (y2 - y1) > 40 and (x2 - x1) > 80:
                    # 改进字体fallback：支持多平台
                    font = None
                    font_paths = [
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                        "/System/Library/Fonts/Helvetica.ttc",  # macOS
                        "C:/Windows/Fonts/arial.ttf",  # Windows
                    ]
                    for font_path in font_paths:
                        try:
                            font = ImageFont.truetype(font_path, 20)
                            break
                        except:
                            continue
                    
                    # 如果所有字体路径都失败，使用默认字体
                    if font is None:
                        try:
                            font = ImageFont.load_default()
                        except:
                            font = None
                    
                    if font:
                        label_text = "Target Object"
                        # 改进文字位置：确保在图像范围内
                        text_y = max(5, min(y1 - 5, h - 30))  # 至少距离顶部5像素，距离底部30像素
                        try:
                            # 绘制文字背景
                            bbox_text = draw.textbbox((x1, text_y), label_text, font=font)
                            if bbox_text and len(bbox_text) >= 4:
                                # 确保文字背景在图像范围内
                                bg_x1 = max(0, bbox_text[0] - 5)
                                bg_y1 = max(0, bbox_text[1] - 2)
                                bg_x2 = min(w, bbox_text[2] + 5)
                                bg_y2 = min(h, bbox_text[3] + 2)
                                if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                                    draw.rectangle(
                                        [bg_x1, bg_y1, bg_x2, bg_y2],
                                        fill=(255, 0, 0),
                                        outline=(255, 255, 255),
                                        width=2
                                    )
                                    # 绘制文字
                                    draw.text((x1, text_y), label_text, fill=(255, 255, 255), font=font)
                        except Exception as e:
                            # 如果文字绘制失败，只记录警告，不影响主流程
                            rag.log_warning(f"绘制标签文字失败: {e}")
                
                img.save(annotated_out)
                rag.log_info(f"保存标注图: {annotated_out}, 框尺寸: {x2-x1}x{y2-y1}, 边距: {pad_w:.1f}x{pad_h:.1f}")
                
        except Exception as e:
            rag.log_warning(f"保存目标裁剪失败: {out_path} - {e}")

    def _save_all_object_crops(rgb: np.ndarray, vision_result: Dict, save_dir: str, step: int) -> Dict[str, str]:
        """
        保存所有检测到的物体的截图
        
        Args:
            rgb: RGB图像
            vision_result: 视觉检测结果
            save_dir: 保存目录
            step: 当前步数
        
        Returns:
            物体标签到截图路径的映射 {label: crop_path}
        """
        object_crop_paths = {}
        objects = vision_result.get("objects", []) if isinstance(vision_result, dict) else []
        
        if not objects:
            return object_crop_paths
        
        # 创建物体截图目录
        objects_dir = os.path.join(save_dir, "detected_objects")
        os.makedirs(objects_dir, exist_ok=True)
        
        for idx, obj in enumerate(objects):
            label = obj.get("label", "unknown")
            bbox = obj.get("bbox", [])
            confidence = obj.get("confidence", 0.0)
            
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            
            # 生成安全的文件名（移除特殊字符）
            safe_label = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in label)
            # 修复：如果safe_label为空（全特殊字符），使用默认值
            if not safe_label or len(safe_label.strip()) == 0:
                safe_label = "unknown"
            
            # 修复：限制文件名长度，防止超过系统限制（通常255字符）
            # 保留前缀和后缀的空间，限制label部分最多100字符
            max_label_len = 100
            if len(safe_label) > max_label_len:
                safe_label = safe_label[:max_label_len]
            
            crop_filename = f"obj_{step:05d}_{idx:02d}_{safe_label}_conf{confidence:.2f}.png"
            # 额外检查：确保完整路径不超过系统限制
            if len(crop_filename) > 200:  # 留一些余量
                # 进一步缩短label
                safe_label = safe_label[:max_label_len - (len(crop_filename) - 200)]
                crop_filename = f"obj_{step:05d}_{idx:02d}_{safe_label}_conf{confidence:.2f}.png"
            
            crop_path = os.path.join(objects_dir, crop_filename)
            
            try:
                # 保存物体截图
                _save_object_crop(rgb, bbox, crop_path, annotated_out=None)
                object_crop_paths[label] = crop_path
                rag.log_info(f"保存物体截图: {label} -> {crop_path}")
            except Exception as e:
                rag.log_warning(f"保存物体截图失败: {label} - {e}")
        
        return object_crop_paths

    def _generate_and_save_plan(found_location: str, step_index: int, vision_result: Dict, estimated_3d_position: Optional[np.ndarray] = None, agent_pose: Optional[Dict] = None):
        """调用 RAG 规划器生成行动计划并保存"""
        try:
            # 获取当前 pose（如果未提供）
            if agent_pose is None:
                try:
                    agent_pose = agent.get_position_and_rotation()
                except:
                    agent_pose = {"position": None, "rotation": None}
            # 使用时间优先级检索记忆（已包含 MemoryGraph 重排序和 fused_conf）
            memory_evidences = rag.retrieve_candidates_with_time_prior(intent, current_time_context=current_time_context, topk=5)
            
            # 利用 fused_conf 优先级：选择最高置信度的记忆
            best_evidence = None
            best_fused_conf = 0.0
            for evidence in memory_evidences:
                fused_conf = evidence.get("fused_conf", 0.0)
                if fused_conf > best_fused_conf:
                    best_fused_conf = fused_conf
                    best_evidence = evidence
            
            current_status = {
                "status": "SUCCESS",
                "target_object": intent.get("object", target_object),
                "target_found": True,
                "found_location": found_location,
                "current_step": step_index,
                "visual_detection_result": "SUCCESS - Target object visually confirmed",
                "best_memory_fused_conf": best_fused_conf if best_evidence else None,
            }
            
            # 添加估计的3D位置信息
            if estimated_3d_position is not None:
                current_status["estimated_3d_position"] = estimated_3d_position.tolist() if isinstance(estimated_3d_position, np.ndarray) else estimated_3d_position
                current_status["spatial_fusion"] = "Multi-view triangulation completed"
                rag.log_info(f"计划中包含估计的3D位置: {current_status['estimated_3d_position']}")
            
            vision_evidence = rag.structure_vision_evidence(
                {
                    "detected": True,
                    "detection_status": "SUCCESS",
                    "location": found_location,
                    "object": intent.get("object", target_object),
                    "confidence": 0.9,
                    "message": "Visual detection confirmed"
                },
                intent.get("object", target_object)
            )
            fused = rag.fuse_vision_with_memory(memory_evidences, vision_evidence)
            plan_res = rag.generate_action_plan_llm(intent, fused, current_status)
            plan = plan_res.data if plan_res.is_success() else {"error": plan_res.error_msg}
            
            # 在生成 plan 前，调用 world_model.predict_effect 用于 plan scoring
            predicted_effects = []
            if rag.world_model and plan and "next_action" in plan:
                try:
                    # 构建 action candidate
                    # 尝试从 best_evidence 获取 node_id 或 entity_id
                    target_entity_id = None
                    if best_evidence:
                        target_entity_id = best_evidence.get("node_id") or best_evidence.get("entity_id") or best_evidence.get("id")
                    
                    action_candidate = {
                        "type": "execute_plan",
                        "target_entity_id": target_entity_id,
                        "parameters": {
                            "action": plan.get("next_action", ""),
                            "reasoning": plan.get("reasoning", ""),
                            "expected_observation": plan.get("expected_observation", "")
                        },
                        "agent_position": agent_pose.get("position") if agent_pose else None,
                        "agent_rotation": agent_pose.get("rotation") if agent_pose else None
                    }
                    
                    # 注意：predict_effect 方法在 WorldModel 中未实现
                    # 如果需要此功能，可以在 WorldModel 中添加该方法
                    # 暂时跳过预测功能
                    prediction = None
                    # prediction = rag.world_model.predict_effect(action_candidate)  # 未实现
                    if prediction:
                        predicted_effects.append({
                            "action": plan.get("next_action", ""),
                            "predicted_state_changes": prediction.get("predicted_state_changes", []),
                            "events": prediction.get("events", []),
                            "confidence": prediction.get("confidence", 0.0),
                            "rationale": prediction.get("rationale", ""),
                            "tentative_event_id": prediction.get("tentative_event_id")
                        })
                        rag.log_info(f"动作效果预测完成: confidence={prediction.get('confidence', 0.0):.3f}")
                except Exception as e:
                    # predict_effect 方法未实现，这是预期的
                    # rag.log_warning(f"预测动作效果失败: {e}")
                    pass
            
            # 在plan中添加估计的3D位置
            if estimated_3d_position is not None:
                plan["estimated_3d_position"] = estimated_3d_position.tolist() if isinstance(estimated_3d_position, np.ndarray) else estimated_3d_position
                plan["spatial_fusion_method"] = "multi_view_triangulation"
            
            # 在plan中添加预测效果
            if predicted_effects:
                plan["predicted_effects"] = predicted_effects
            
            plan_file = os.path.join(save_obs_dir, "plan.json")
            with open(plan_file, "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
            rag.log_info("行动计划已生成并保存", {"plan_file": plan_file})
        except Exception as e:
            rag.log_warning(f"生成/保存行动计划失败: {e}")

    # 多角度验证函数：在当前位置旋转多角度，检查目标是否稳定出现
    def _multi_angle_verify(agent, target_syns, target_name, num_angles=4, angle_step=30):
        """多角度验证：在当前位置旋转多角度，检查目标是否稳定出现"""
        verify_results = []
        original_pose = agent.get_position_and_rotation()
        for i in range(num_angles):
            rgb = agent.get_rgb()
            ok, vision_result = _vision_with_retries(rgb, target_syns, retries=2)
            if ok:
                obj = _find_matching_object(vision_result.data, target_name, rgb.shape, min_conf=0.65)
                if obj:
                    verify_results.append({
                        "found": True,
                        "confidence": float(obj.get("confidence", 0.0)),
                        "label": obj.get("label", ""),
                    })
                else:
                    verify_results.append({"found": False})
            if i < num_angles - 1:  # 最后一次不需要旋转
                agent.step("turn_left")
                time.sleep(0.1)  # 短暂停顿确保动作完成
        # 计算验证通过率
        found_count = sum(1 for r in verify_results if r.get("found", False))
        avg_conf = sum(r.get("confidence", 0.0) for r in verify_results if r.get("found", False))
        if found_count > 0:
            avg_conf /= found_count
        return found_count >= (num_angles // 2 + 1), avg_conf, found_count, num_angles

    # 初次 360° 环扫（旋转采样），以快速积累视觉记忆
    rag.log_info("开始初次旋转环扫进行感知...")
    # 初始化空间语义融合模块（用于环扫阶段）
    sweep_fusion_module = SpatialSemanticFusionModule(
        max_buffer=10,
        min_views=3,
        camera_intrinsics=agent.camera_intrinsics,
        alpha=0.7,
        lambda_spatial=1.0,
        log_dir=save_obs_dir,
        fusion_mode=fusion_mode
    )
    confirm_hits = 0
    sweep_confirm_history = []
    for i in range(rotate_sweep):
        pose = agent.get_position_and_rotation()
        rgb = agent.get_rgb()
        depth = agent.get_depth()
        
        if save_obs_dir:
            _save_rgb(rgb, os.path.join(sweep_dir, f"sweep_{i:03d}.png"))
            if depth is not None:
                _save_depth(depth, os.path.join(depth_dir, f"sweep_{i:03d}_depth.png"))
        
        ok, vision_result = _vision_with_retries(rgb, target_syns, retries=3)
        if ok:
            # 保存检测结果
            if save_obs_dir:
                _save_detection_result(vision_result.data, os.path.join(detection_dir, f"sweep_{i:03d}_detection.json"))
            # 检测物体是否过近，如果过近则后退
            too_close, max_area = _check_too_close(vision_result.data, rgb.shape)
            if too_close:
                rag.log_warning(f"环扫中检测到物体过近（最大面积占比: {max_area:.2%}），执行后退动作")
                back_steps = 0
                max_back_steps = 3  # 最多后退3步
                while back_steps < max_back_steps:
                    agent.step("move_backward")
                    back_steps += 1
                    time.sleep(0.1)
                    # 记录后退位置
                    back_pose = agent.get_position_and_rotation()
                    path_trajectory.append({
                        "timestamp": time.time(),
                        "step": i + 1,
                        "phase": "sweep",
                        "action": "move_backward",
                        "back_step": back_steps,
                        "reason": "too_close",
                        **back_pose
                    })
                    # 重新获取图像并检测
                    rgb = agent.get_rgb()
                    ok, vision_result = _vision_with_retries(rgb, target_syns, retries=2)
                    if ok:
                        too_close_new, max_area_new = _check_too_close(vision_result.data, rgb.shape)
                        if not too_close_new:
                            rag.log_info(f"后退 {back_steps} 步后获得合适观察距离（面积占比: {max_area_new:.2%}）")
                            break
                    else:
                        break
                # 如果后退后仍过近，记录警告但继续处理
                if back_steps >= max_back_steps:
                    rag.log_warning(f"已后退 {max_back_steps} 步，但物体可能仍过近，继续处理")
            
            # 保存所有检测到的物体的截图
            scene_image_path = os.path.join(sweep_dir, f"sweep_{i:03d}.png") if save_obs_dir else None
            object_crop_paths = _save_all_object_crops(rgb, vision_result.data, save_obs_dir, i + 1) if save_obs_dir else {}
            
            # 获取当前位置和旋转
            pose = agent.get_position_and_rotation()
            
            # 增强的视觉记忆存储（包含位置和截图）
            rag.add_vision_memory(
                vision_result.data, 
                image_path=scene_image_path,
                agent_position=pose["position"],
                agent_rotation=pose["rotation"],
                object_crop_paths=object_crop_paths
            )
            strict_obj = _find_matching_object(vision_result.data, intent.get("object", target_object), rgb.shape, min_conf=0.65)
            if strict_obj:
                conf = float(strict_obj.get("confidence", 0.0))
                bbox = strict_obj.get("bbox", [])
                label = strict_obj.get("label", "")
                
                # 添加到空间语义融合模块（传入深度图）
                sweep_fusion_module.add_observation(
                    position=pose["position"],
                    rotation=pose["rotation"],
                    bbox=bbox,
                    confidence=conf,
                    label=label,
                    img_shape=(rgb.shape[0], rgb.shape[1]),
                    depth=depth
                )
                
                # 尝试估计3D位置
                estimated_3d_pos = sweep_fusion_module.estimate_object_position()
                
                # 计算 depth_stats（从 depth 中提取）
                depth_stats = None
                if depth is not None and bbox:
                    try:
                        depth_h, depth_w = depth.shape[:2]
                        x1, y1, x2, y2 = bbox
                        x1_clamped = max(0, min(depth_w - 1, int(x1)))
                        y1_clamped = max(0, min(depth_h - 1, int(y1)))
                        x2_clamped = max(x1_clamped + 1, min(depth_w, int(x2)))
                        y2_clamped = max(y1_clamped + 1, min(depth_h, int(y2)))
                        
                        depth_roi = depth[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
                        valid_depths = depth_roi[(depth_roi >= 0.1) & (depth_roi <= 10.0) & 
                                                 (~np.isnan(depth_roi)) & (~np.isinf(depth_roi))]
                        
                        if len(valid_depths) > 0:
                            depth_stats = {
                                "min": float(np.min(valid_depths)),
                                "max": float(np.max(valid_depths)),
                                "mean": float(np.mean(valid_depths)),
                                "valid_count": len(valid_depths)
                            }
                    except Exception as e:
                        rag.log_warning(f"计算 depth_stats 失败: {e}")
                
                # 获取 memnode_id（从最近的 memory_evidences 中获取，如果有）
                memnode_id = None
                try:
                    if rag.memory_graph:
                        # 尝试从最近的 add_vision_memory 调用中获取 node_id
                        # 这里简化处理，可以后续优化
                        pass
                except:
                    pass
                
                # 使用统一的 process_fused_result 处理融合结果
                if estimated_3d_pos is not None:
                    fused_result = {
                        "label": label,
                        "fused_3d_pos": estimated_3d_pos.tolist() if isinstance(estimated_3d_pos, np.ndarray) else estimated_3d_pos,
                        "confidence": conf,
                        "bbox": bbox,
                        "image_path": scene_image_path,
                        "agent_position": pose["position"],
                        "agent_rotation": pose["rotation"],
                        "ts": time.time(),
                        "metadata": {
                            "depth_stats": depth_stats,
                            "phase": "sweep"
                        }
                    }
                    
                    # 调用统一的处理函数
                    process_result = rag.process_fused_result(fused_result)
                    if process_result:
                        rag.log_info(f"融合结果已处理: memory_node_id={process_result.get('memory_node_id')}, entity_id={process_result.get('entity_id')}, action={process_result.get('action')}")
                
                sweep_confirm_history.append({
                    "step": i + 1,
                    "confidence": conf,
                    "label": label,
                    "estimated_3d_position": estimated_3d_pos.tolist() if estimated_3d_pos is not None else None
                })
                confirm_hits += 1
                
                # WorldModel: 检查是否需要注册移动
                try:
                    if estimated_3d_pos is not None and rag.world_model:
                        # 尝试从 memory 中找到候选实体 ID
                        memories = rag.retrieve_candidates_by_intent(intent, topk=5)
                        candidate_eid = None
                        for m in memories:
                            meta = m.get("meta", {})
                            # 尝试从多个可能的字段获取 entity_id
                            entity_id = meta.get("entity_id") or meta.get("world_model_entity_id")
                            if entity_id:
                                candidate_eid = entity_id
                                break
                        
                        # 如果找到候选实体，检查位置差异
                        if candidate_eid:
                            old_ent = rag.world_model.get_entity(candidate_eid)
                            if old_ent and old_ent.get("position") is not None:
                                old_pos = np.array(old_ent["position"])
                                new_pos = np.array(estimated_3d_pos)
                                dist = np.linalg.norm(new_pos - old_pos)
                                
                                if dist > 0.5:  # move_th = 0.5m
                                    fused_conf = sweep_fusion_module.get_fused_confidence(conf, old_ent["position"])
                                    rag.world_model_register_movement(
                                        candidate_eid,
                                        estimated_3d_pos.tolist() if isinstance(estimated_3d_pos, np.ndarray) else estimated_3d_pos,
                                        method="triangulate",
                                        confidence=fused_conf
                                    )
                                    rag.log_info(f"WorldModel: 检测到实体移动 (entity_id={candidate_eid}, distance={dist:.3f}m)")
                except Exception as e:
                    rag.log_warning(f"WorldModel 注册移动失败: {e}")
            else:
                confirm_hits = max(0, confirm_hits - 1)
                
                # WorldModel: 当 strict_obj 未命中时，检查是否有历史实体在视野内
                try:
                    if rag.world_model:
                        # 尝试从 memory 中找到候选实体 ID
                        memories = rag.retrieve_candidates_by_intent(intent, topk=5)
                        candidate_eid = None
                        for m in memories:
                            meta = m.get("meta", {})
                            # 尝试从多个可能的字段获取 entity_id
                            entity_id = meta.get("entity_id") or meta.get("world_model_entity_id")
                            if entity_id:
                                candidate_eid = entity_id
                                break
                        
                        # 如果找到候选实体，检查是否在视野内
                        if candidate_eid:
                            old_ent = rag.world_model.get_entity(candidate_eid)
                            if old_ent and old_ent.get("position") is not None:
                                in_view, vis_score = _is_position_in_view(
                                    agent,
                                    old_ent["position"],
                                    agent.camera_intrinsics,
                                    depth
                                )
                                
                                if in_view and vis_score > 0.4:
                                    evidence = {
                                        "ts": time.time(),
                                        "agent_pose": pose,
                                        "vis_score": vis_score,
                                        "frame_id": f"sweep_step_{i+1}",
                                        "depth_stats": None
                                    }
                                    
                                    # 计算 depth_stats（如果有深度图）
                                    if depth is not None:
                                        try:
                                            depth_h, depth_w = depth.shape[:2]
                                            depth_center = depth[depth_h//2, depth_w//2]
                                            if 0.1 <= depth_center <= 10.0:
                                                evidence["depth_stats"] = {
                                                    "mean": float(depth_center),
                                                    "valid_count": 1
                                                }
                                        except:
                                            pass
                                    
                                    rag.world_model_mark_no_detection(candidate_eid, evidence)
                                    rag.log_info(f"WorldModel: 标记未检测 (entity_id={candidate_eid}, vis_score={vis_score:.3f})")
                                    
                                    # 评估 missing 状态
                                    rag.world_model_evaluate_missing(candidate_eid, n_no_detects=3, vis_th=0.6)
                except Exception as e:
                    rag.log_warning(f"WorldModel 标记未检测失败: {e}")
            # 提升阈值：需要连续3次确认，且平均置信度≥0.7
            if confirm_hits >= 3:
                avg_conf = sum(c["confidence"] for c in sweep_confirm_history[-3:]) / min(3, len(sweep_confirm_history))
                if avg_conf >= 0.7:
                    rag.log_info(f"初次环扫检测到目标，执行多角度验证... (平均置信度: {avg_conf:.2f})")
                    # 执行多角度验证
                    verify_ok, verify_conf, found_count, total_angles = _multi_angle_verify(
                        agent, target_syns, intent.get("object", target_object), num_angles=4
                    )
                    if verify_ok and verify_conf >= 0.65:
                        rag.log_info(f"多角度验证通过！{found_count}/{total_angles} 角度检测到目标，平均置信度: {verify_conf:.2f}")
                        # 记录终点位置
                        end_pose = agent.get_position_and_rotation()
                        
                        # 获取最终估计的3D位置
                        final_estimated_3d_pos = sweep_fusion_module.estimated_position
                        if final_estimated_3d_pos is None:
                            # 如果还没有估计位置，尝试再次估计
                            final_estimated_3d_pos = sweep_fusion_module.estimate_object_position()
                        
                        # 获取最后一次检测的bbox用于保存
                        final_rgb = agent.get_rgb()
                        final_ok, final_vision = _vision_with_retries(final_rgb, target_syns, retries=2)
                        if final_ok:
                            final_obj = _find_matching_object(final_vision.data, intent.get("object", target_object), final_rgb.shape)
                            if final_obj and isinstance(final_obj.get("bbox"), (list, tuple)) and len(final_obj.get("bbox")) == 4:
                                _save_object_crop(
                                    final_rgb,
                                    final_obj.get("bbox"),
                                    os.path.join(save_obs_dir, "target_object.png"),
                                    annotated_out=os.path.join(save_obs_dir, "target_object_annotated.png"),
                                )
                        # 生成计划（附带估计的3D位置）
                        _generate_and_save_plan("initial_sweep", i + 1, vision_result.data, estimated_3d_position=final_estimated_3d_pos, agent_pose=end_pose)
                        
                        # 如果有估计的3D位置，记录日志
                        if final_estimated_3d_pos is not None:
                            rag.log_info(f"环扫阶段：将估计的3D位置 {final_estimated_3d_pos.tolist()} 更新到视觉记忆中")
                        
                        path_trajectory.append({
                            "timestamp": time.time(),
                            "step": i + 1,
                            "phase": "sweep",
                            "action": "detect_target_verified",
                            "target_found": True,
                            "verify_confidence": verify_conf,
                            "verify_ratio": f"{found_count}/{total_angles}",
                            "estimated_3d_position": final_estimated_3d_pos.tolist() if final_estimated_3d_pos is not None else None,
                            **end_pose
                        })
                        # 保存路径日志
                        _save_path_log(path_trajectory, save_obs_dir, start_time)
                        
                        # 后处理：应用衰减和评估 missing/removed 状态
                        _post_process_memory_world()
                        
                        # 打印摘要
                        rag.print_memory_world_summary()
                        
                        return {
                            "status": "SUCCESS",
                            "found": True,
                            "steps": i + 1,
                            "location": "initial_sweep",
                            "start_position": start_pose["position"],
                            "end_position": end_pose["position"],
                            "estimated_3d_position": final_estimated_3d_pos.tolist() if final_estimated_3d_pos is not None else None,
                            "verify_confidence": verify_conf,
                            "verify_ratio": f"{found_count}/{total_angles}",
                        }
                    else:
                        rag.log_warning(f"多角度验证未通过，继续环扫... (通过率: {found_count}/{total_angles}, 置信度: {verify_conf:.2f})")
                        confirm_hits = max(0, confirm_hits - 2)
        # 记录环扫位置
        path_trajectory.append({
            "timestamp": time.time(),
            "step": i + 1,
            "phase": "sweep",
            "action": "turn_left",
            **pose
        })
        agent.step("turn_left")

    # 探索主循环
    rag.log_info("进入探索循环...")
    # 初始化空间语义融合模块
    fusion_module = SpatialSemanticFusionModule(
        max_buffer=10,
        min_views=3,
        camera_intrinsics=agent.camera_intrinsics,
        alpha=0.7,
        lambda_spatial=1.0,
        log_dir=save_obs_dir,
        fusion_mode=fusion_mode
    )
    rag.log_info("空间语义融合模块已初始化")
    
    confirm_hits = 0
    confirm_history = []  # 记录确认历史，包含置信度和位置
    last_confirm_step = -1
    # 运动缓冲：连续失败计数
    consecutive_failures = 0
    failure_threshold = 15  # 连续15次失败后执行全向扫描
    last_full_scan_step = -1
    # 卡住检测：窗口位置分布过小则认为在同一房间打转
    pos_window = deque(maxlen=60)
    last_relocate_ts = 0.0
    relocate_cooldown = 10.0  # 秒
    stuck_spread_thresh = 1.2  # 以米估计的平面范围阈值
    relocate_moves = 24  # 脱困直行步数
    # 碰撞检测
    last_position = None
    collision_count = 0
    for step in range(1, max_steps + 1):
        pose = agent.get_position_and_rotation()
        rgb = agent.get_rgb()
        depth = agent.get_depth()
        
        if save_obs_dir:
            _save_rgb(rgb, os.path.join(explore_dir, f"step_{step:05d}.png"))
            if depth is not None:
                _save_depth(depth, os.path.join(depth_dir, f"step_{step:05d}_depth.png"))
        
        ok, vision_result = _vision_with_retries(rgb, target_syns, retries=3)
        if ok:
            # 保存检测结果
            if save_obs_dir:
                _save_detection_result(vision_result.data, os.path.join(detection_dir, f"step_{step:05d}_detection.json"))
            # 检测物体是否过近，如果过近则后退
            too_close, max_area = _check_too_close(vision_result.data, rgb.shape)
            if too_close:
                rag.log_warning(f"探索中检测到物体过近（最大面积占比: {max_area:.2%}），执行后退动作")
                back_steps = 0
                max_back_steps = 3  # 最多后退3步
                while back_steps < max_back_steps:
                    agent.step("move_backward")
                    back_steps += 1
                    time.sleep(0.1)
                    # 记录后退位置
                    back_pose = agent.get_position_and_rotation()
                    path_trajectory.append({
                        "timestamp": time.time(),
                        "step": step,
                        "phase": "explore",
                        "action": "move_backward",
                        "back_step": back_steps,
                        "reason": "too_close",
                        **back_pose
                    })
                    # 重新获取图像并检测
                    rgb = agent.get_rgb()
                    ok, vision_result = _vision_with_retries(rgb, target_syns, retries=2)
                    if ok:
                        too_close_new, max_area_new = _check_too_close(vision_result.data, rgb.shape)
                        if not too_close_new:
                            rag.log_info(f"后退 {back_steps} 步后获得合适观察距离（面积占比: {max_area_new:.2%}）")
                            break
                    else:
                        break
                # 如果后退后仍过近，记录警告但继续处理
                if back_steps >= max_back_steps:
                    rag.log_warning(f"已后退 {max_back_steps} 步，但物体可能仍过近，继续处理")
            
            # 保存所有检测到的物体的截图
            scene_image_path = os.path.join(explore_dir, f"step_{step:05d}.png") if save_obs_dir else None
            object_crop_paths = _save_all_object_crops(rgb, vision_result.data, save_obs_dir, step) if save_obs_dir else {}
            
            # 获取当前位置和旋转
            pose = agent.get_position_and_rotation()
            
            # 增强的视觉记忆存储（包含位置和截图）
            rag.add_vision_memory(
                vision_result.data, 
                image_path=scene_image_path,
                agent_position=pose["position"],
                agent_rotation=pose["rotation"],
                object_crop_paths=object_crop_paths
            )
            strict_obj = _find_matching_object(vision_result.data, intent.get("object", target_object), rgb.shape, min_conf=0.65)
            if strict_obj:
                conf = float(strict_obj.get("confidence", 0.0))
                bbox = strict_obj.get("bbox", [])
                label = strict_obj.get("label", "")
                
                # 添加到空间语义融合模块（传入深度图）
                fusion_module.add_observation(
                    position=pose["position"],
                    rotation=pose["rotation"],
                    bbox=bbox,
                    confidence=conf,
                    label=label,
                    img_shape=(rgb.shape[0], rgb.shape[1]),
                    depth=depth
                )
                
                # 尝试估计3D位置
                estimated_3d_pos = fusion_module.estimate_object_position()
                fused_conf = conf
                
                # 使用统一的 process_fused_result 处理融合结果
                if estimated_3d_pos is not None:
                    fused_result = {
                        "label": label,
                        "fused_3d_pos": estimated_3d_pos.tolist() if isinstance(estimated_3d_pos, np.ndarray) else estimated_3d_pos,
                        "confidence": conf,
                        "bbox": bbox,
                        "image_path": scene_image_path,
                        "agent_position": pose["position"],
                        "agent_rotation": pose["rotation"],
                        "ts": time.time(),
                        "metadata": {
                            "phase": "explore"
                        }
                    }
                    
                    # 调用统一的处理函数
                    process_result = rag.process_fused_result(fused_result)
                    if process_result:
                        rag.log_info(f"融合结果已处理: memory_node_id={process_result.get('memory_node_id')}, entity_id={process_result.get('entity_id')}, action={process_result.get('action')}")
                
                if estimated_3d_pos is not None:
                    # 查询记忆库中的历史位置（使用时间优先级和 MemoryGraph 重排序）
                    memory_evidences = rag.retrieve_candidates_with_time_prior(intent, current_time_context=current_time_context, topk=3)
                    memory_position = None
                    best_fused_conf = 0.0
                    for evidence in memory_evidences:
                        # 优先使用 fused_conf 高的记忆
                        evidence_fused_conf = evidence.get("fused_conf", 0.0)
                        if evidence_fused_conf > best_fused_conf:
                            meta = evidence.get("meta", {})
                            spatial = meta.get("spatial", {})
                            # 尝试从新的格式获取位置
                            if isinstance(spatial, dict):
                                world_pos = spatial.get("world_position") or spatial.get("position")
                            else:
                                world_pos = None
                            
                            # 如果新格式没有，尝试从 node_data 获取
                            if world_pos is None:
                                node_data = meta.get("node_data", {})
                                world_pos = node_data.get("position")
                            
                            if world_pos is not None:
                                memory_position = world_pos
                                best_fused_conf = evidence_fused_conf
                                rag.log_info(f"使用高置信度记忆位置 (fused_conf={best_fused_conf:.3f}): {memory_position}")
                    
                    # 计算融合置信度
                    fused_conf = fusion_module.get_fused_confidence(conf, memory_position)
                    rag.log_info(f"空间融合：估计3D位置 {estimated_3d_pos.tolist()}, 融合置信度: {fused_conf:.3f}")
                    
                    # WorldModel: 检查是否需要注册移动
                    try:
                        if rag.world_model:
                            # 尝试从 memory 中找到候选实体 ID
                            candidate_eid = None
                            for evidence in memory_evidences:
                                meta = evidence.get("meta", {})
                                entity_id = meta.get("entity_id") or meta.get("world_model_entity_id")
                                if entity_id:
                                    candidate_eid = entity_id
                                    break
                            
                            # 如果找到候选实体，检查位置差异
                            if candidate_eid:
                                old_ent = rag.world_model.get_entity(candidate_eid)
                                if old_ent and old_ent.get("position") is not None:
                                    old_pos = np.array(old_ent["position"])
                                    new_pos = np.array(estimated_3d_pos)
                                    dist = np.linalg.norm(new_pos - old_pos)
                                    
                                    if dist > 0.5:  # move_th = 0.5m
                                        rag.world_model_register_movement(
                                            candidate_eid,
                                            estimated_3d_pos.tolist() if isinstance(estimated_3d_pos, np.ndarray) else estimated_3d_pos,
                                            method="triangulate",
                                            confidence=fused_conf
                                        )
                                        rag.log_info(f"WorldModel: 检测到实体移动 (entity_id={candidate_eid}, distance={dist:.3f}m)")
                    except Exception as e:
                        rag.log_warning(f"WorldModel 注册移动失败: {e}")
                
                # 检测到目标，减少失败计数
                consecutive_failures = max(0, consecutive_failures - 2)
                # 记录确认信息（使用融合置信度）
                confirm_history.append({
                    "step": step,
                    "confidence": fused_conf,
                    "label": label,
                    "position": pose["position"],
                    "estimated_3d_position": estimated_3d_pos.tolist() if estimated_3d_pos is not None else None
                })
                # 如果连续检测到，累积计数
                if step - last_confirm_step <= 3:  # 3步内连续检测
                    confirm_hits += 1
                else:
                    confirm_hits = 1  # 重置计数
                last_confirm_step = step
                
                # 提升阈值：需要连续3次确认，且平均置信度≥0.7
                if confirm_hits >= 3:
                    avg_conf = sum(c["confidence"] for c in confirm_history[-3:]) / min(3, len(confirm_history))
                    if avg_conf >= 0.7:
                        rag.log_info(f"在第 {step} 步检测到目标，执行多角度验证... (平均置信度: {avg_conf:.2f})")
                        # 执行多角度验证
                        verify_ok, verify_conf, found_count, total_angles = _multi_angle_verify(
                            agent, target_syns, intent.get("object", target_object), num_angles=4
                        )
                        if verify_ok and verify_conf >= 0.65:
                            rag.log_info(f"多角度验证通过！{found_count}/{total_angles} 角度检测到目标，平均置信度: {verify_conf:.2f}")
                            # 重置连续失败计数
                            consecutive_failures = 0
                            # 记录终点位置
                            end_pose = agent.get_position_and_rotation()
                            
                            # 获取最终估计的3D位置
                            final_estimated_3d_pos = fusion_module.estimated_position
                            if final_estimated_3d_pos is None:
                                # 如果还没有估计位置，尝试再次估计
                                final_estimated_3d_pos = fusion_module.estimate_object_position()
                            
                            # 获取最后一次检测的bbox用于保存
                            final_rgb = agent.get_rgb()
                            final_ok, final_vision = _vision_with_retries(final_rgb, target_syns, retries=2)
                            if final_ok:
                                final_obj = _find_matching_object(final_vision.data, intent.get("object", target_object), final_rgb.shape)
                                if final_obj and isinstance(final_obj.get("bbox"), (list, tuple)) and len(final_obj.get("bbox")) == 4:
                                    _save_object_crop(
                                        final_rgb,
                                        final_obj.get("bbox"),
                                        os.path.join(save_obs_dir, "target_object.png"),
                                        annotated_out=os.path.join(save_obs_dir, "target_object_annotated.png"),
                                    )
                            
                            # 生成计划（附带估计的3D位置）
                            current_pose = agent.get_position_and_rotation()
                            _generate_and_save_plan("habitat", step, vision_result.data, estimated_3d_position=final_estimated_3d_pos, agent_pose=current_pose)
                            
                            # 如果有估计的3D位置，更新到视觉记忆中
                            if final_estimated_3d_pos is not None:
                                # 更新最后一次视觉记忆的metadata，添加估计的3D位置
                                rag.log_info(f"将估计的3D位置 {final_estimated_3d_pos.tolist()} 更新到视觉记忆中")
                                # 注意：这里可以扩展add_vision_memory函数来支持更新，或者单独存储
                            
                            path_trajectory.append({
                                "timestamp": time.time(),
                                "step": step,
                                "phase": "explore",
                                "action": "detect_target_verified",
                                "target_found": True,
                                "verify_confidence": verify_conf,
                                "verify_ratio": f"{found_count}/{total_angles}",
                                "estimated_3d_position": final_estimated_3d_pos.tolist() if final_estimated_3d_pos is not None else None,
                                **end_pose
                            })
                        # 保存路径日志
                        _save_path_log(path_trajectory, save_obs_dir, start_time)
                        
                        # 后处理：应用衰减和评估 missing/removed 状态
                        _post_process_memory_world()
                        
                        # 打印摘要
                        rag.print_memory_world_summary()
                        
                        return {
                            "status": "SUCCESS",
                            "found": True,
                            "steps": i + 1,
                            "location": "initial_sweep",
                            "start_position": start_pose["position"],
                            "end_position": end_pose["position"],
                            "estimated_3d_position": final_estimated_3d_pos.tolist() if final_estimated_3d_pos is not None else None,
                            "verify_confidence": verify_conf,
                            "verify_ratio": f"{found_count}/{total_angles}",
                        }
                    else:
                        rag.log_warning(f"多角度验证未通过，继续搜索... (通过率: {found_count}/{total_angles}, 置信度: {verify_conf:.2f})")
            else:
                # strict_obj 未命中时的处理
                # WorldModel: 检查是否有历史实体在视野内
                try:
                    if rag.world_model:
                        # 尝试从 memory 中找到候选实体 ID
                        memories = rag.retrieve_candidates_by_intent(intent, topk=5)
                        candidate_eid = None
                        for m in memories:
                            meta = m.get("meta", {})
                            entity_id = meta.get("entity_id") or meta.get("world_model_entity_id")
                            if entity_id:
                                candidate_eid = entity_id
                                break
                        
                        # 如果找到候选实体，检查是否在视野内
                        if candidate_eid:
                            old_ent = rag.world_model.get_entity(candidate_eid)
                            if old_ent and old_ent.get("position") is not None:
                                in_view, vis_score = _is_position_in_view(
                                    agent,
                                    old_ent["position"],
                                    agent.camera_intrinsics,
                                    depth
                                )
                                
                                if in_view and vis_score > 0.4:
                                    evidence = {
                                        "ts": time.time(),
                                        "agent_pose": pose,
                                        "vis_score": vis_score,
                                        "frame_id": f"explore_step_{step}",
                                        "depth_stats": None
                                    }
                                    
                                    # 计算 depth_stats（如果有深度图）
                                    if depth is not None:
                                        try:
                                            depth_h, depth_w = depth.shape[:2]
                                            depth_center = depth[depth_h//2, depth_w//2]
                                            if 0.1 <= depth_center <= 10.0:
                                                evidence["depth_stats"] = {
                                                    "mean": float(depth_center),
                                                    "valid_count": 1
                                                }
                                        except:
                                            pass
                                    
                                    rag.world_model_mark_no_detection(candidate_eid, evidence)
                                    rag.log_info(f"WorldModel: 标记未检测 (entity_id={candidate_eid}, vis_score={vis_score:.3f})")
                                    
                                    # 评估 missing 状态
                                    rag.world_model_evaluate_missing(candidate_eid, n_no_detects=3, vis_th=0.6)
                except Exception as e:
                    rag.log_warning(f"WorldModel 标记未检测失败: {e}")
            
            # 未检测到目标时的处理（无论是否找到候选实体）
            if not strict_obj:
                # 未检测到目标，降低确认计数并增加失败计数
                if confirm_hits > 0:
                    confirm_hits = max(0, confirm_hits - 1)
                consecutive_failures += 1
                # 如果连续多次未检测到目标，清空融合模块缓冲区（避免累积错误观测）
                if consecutive_failures >= 5:
                    fusion_module.clear_buffer()
                    rag.log_info("连续多次未检测到目标，清空空间融合模块缓冲区")
        else:
            rag.log_warning(f"视觉检测失败: {vision_result}")
            consecutive_failures += 1
            # 视觉检测失败时也清空缓冲区
            if consecutive_failures >= 5:
                fusion_module.clear_buffer()
                rag.log_info("连续多次视觉检测失败，清空空间融合模块缓冲区")

        # 运动缓冲：连续多次失败后执行全向旋转扫描
        if consecutive_failures >= failure_threshold and (step - last_full_scan_step) > failure_threshold:
            rag.log_info(f"连续 {consecutive_failures} 次检测失败，执行全向旋转扫描...")
            last_full_scan_step = step
            scan_angles = 12
            for scan_i in range(scan_angles):
                rgb_scan = agent.get_rgb()
                depth_scan = agent.get_depth()
                scan_image_path = os.path.join(explore_dir, f"buffer_scan_{step:05d}_{scan_i:03d}.png") if save_obs_dir else None
                if save_obs_dir:
                    _save_rgb(rgb_scan, scan_image_path)
                    if depth_scan is not None:
                        _save_depth(depth_scan, os.path.join(depth_dir, f"buffer_scan_{step:05d}_{scan_i:03d}_depth.png"))
                ok_scan, vision_scan = _vision_with_retries(rgb_scan, target_syns, retries=2)
                if ok_scan:
                    # 保存检测结果
                    if save_obs_dir:
                        _save_detection_result(vision_scan.data, os.path.join(detection_dir, f"buffer_scan_{step:05d}_{scan_i:03d}_detection.json"))
                    # 保存物体截图
                    scan_object_crops = _save_all_object_crops(rgb_scan, vision_scan.data, save_obs_dir, step * 1000 + scan_i) if save_obs_dir else {}
                    scan_pose = agent.get_position_and_rotation()
                    
                    # 增强的视觉记忆存储
                    rag.add_vision_memory(
                        vision_scan.data,
                        image_path=scan_image_path,
                        agent_position=scan_pose["position"],
                        agent_rotation=scan_pose["rotation"],
                        object_crop_paths=scan_object_crops
                    )
                    strict_obj_scan = _find_matching_object(vision_scan.data, intent.get("object", target_object), rgb_scan.shape, min_conf=0.65)
                    if strict_obj_scan:
                        rag.log_info(f"全向扫描中检测到目标，重置失败计数")
                        consecutive_failures = 0
                        # 继续正常流程
                        break
                if scan_i < scan_angles - 1:
                    agent.step("turn_left")
                    time.sleep(0.1)
            # 重置失败计数（即使没找到也重置，避免频繁扫描）
            consecutive_failures = 0

        # 获取深度图并检测未探索区域
        depth = agent.get_depth()
        has_unexplored, unexplored_ratio = _detect_unexplored_areas(depth) if depth is not None else (False, 0.0)
        
        # 策略决定动作
        state = agent.get_agent_state()
        pos = np.array(state.position)
        
        # 根据策略类型调用不同的plan方法
        if explore_policy == "frontier":
            # 获取记忆证据用于知识引导（使用时间优先级）
            memory_evidences_for_planning = rag.retrieve_candidates_with_time_prior(intent, current_time_context=current_time_context, topk=5)
            action = policy.plan(
                current_pos=pos,
                depth=depth,
                target_object=intent.get("object", target_object),
                memory_evidences=memory_evidences_for_planning,
                step_count=step
            )
            # 保存前沿信息（如果前沿已更新）
            if save_obs_dir and hasattr(policy, 'frontiers') and policy.frontiers:
                _save_frontier_info(policy.frontiers, pos, os.path.join(frontier_dir, f"step_{step:05d}_frontiers.json"))
        else:
            # 简单策略只需要位置
            action = policy.plan(pos)
        
        # 基于未探索区域调整动作：如果有未探索区域，优先前进（作为补充）
        if has_unexplored and action == "turn_left":
            # 有未探索区域时，减少转向，增加前进
            if random.random() < 0.4:  # 40%概率改为前进（降低概率，因为前沿探索已经考虑了）
                action = "move_forward"
                rag.log_info(f"检测到未探索区域（占比: {unexplored_ratio:.2%}），调整动作为前进")
        
        # 记录探索位置（在执行动作前记录当前位置）
        path_trajectory.append({
            "timestamp": time.time(),
            "step": step,
            "phase": "explore",
            "action": action,
            "has_unexplored": has_unexplored,
            "unexplored_ratio": unexplored_ratio,
            "consecutive_failures": consecutive_failures,
            **pose
        })
        
        # 使用碰撞检测执行动作
        rgb_after, is_collision = agent.step_with_collision_check(action)
        if is_collision:
            collision_count += 1
            rag.log_warning(f"检测到碰撞（第 {collision_count} 次），执行避障动作")
            # 避障：随机转向
            if random.random() < 0.5:
                avoid_action = "turn_left"
            else:
                avoid_action = "turn_right"
            # 执行多次转向确保避开障碍
            for _ in range(3):
                agent.step(avoid_action)
                time.sleep(0.1)
            rag.log_info(f"避障完成，已转向 {avoid_action}")
            # 记录避障轨迹
            avoid_pose = agent.get_position_and_rotation()
            path_trajectory.append({
                "timestamp": time.time(),
                "step": step,
                "phase": "explore",
                "action": f"avoid_{avoid_action}",
                "reason": "collision",
                **avoid_pose
            })
        else:
            # 没有碰撞，重置碰撞计数
            if collision_count > 0:
                collision_count = 0

        # 更新卡住检测窗口
        pos_window.append(np.array(agent.get_agent_state().position))
        if len(pos_window) >= pos_window.maxlen:
            window_arr = np.stack(pos_window, axis=0)
            spread_xz = (window_arr[:, 0].max() - window_arr[:, 0].min()) + (window_arr[:, 2].max() - window_arr[:, 2].min())
            now_ts = time.time()
            if spread_xz < stuck_spread_thresh and (now_ts - last_relocate_ts) > relocate_cooldown:
                rag.log_warning(f"检测到疑似在同一房间打转，执行脱困探索序列。spread_xz={spread_xz:.2f}")
                # 脱困：直行一段并间隔右转，尽量跨越门口进入新空间
                for k in range(relocate_moves):
                    agent.step("move_forward")
                    if (k + 1) % 6 == 0:
                        agent.step("turn_right")
                    # 保存观测与轨迹
                    rgb2 = agent.get_rgb()
                    if save_obs_dir:
                        _save_rgb(rgb2, os.path.join(explore_dir, f"relocate_{step:05d}_{k:03d}.png"))
                    pose2 = agent.get_position_and_rotation()
                    path_trajectory.append({
                        "timestamp": time.time(),
                        "step": step,
                        "phase": "relocate",
                        "action": "move_forward/turn_right_seq",
                        **pose2
                    })
                last_relocate_ts = now_ts
                # 清空窗口，重新评估
                pos_window.clear()

    # 达到最大步数，记录终点位置并保存日志
    end_pose = agent.get_position_and_rotation()
    path_trajectory.append({
        "timestamp": time.time(),
        "step": max_steps,
        "phase": "end",
        "action": "max_steps_reached",
        "target_found": False,
        **end_pose
    })
    _save_path_log(path_trajectory, save_obs_dir, start_time)
    rag.log_warning("达到最大步数未找到目标")
    
    # 后处理：应用衰减和评估 missing/removed 状态
    _post_process_memory_world()
    
    # 打印摘要
    rag.print_memory_world_summary()
    
    return {
        "status": "FAIL",
        "found": False,
        "steps": max_steps,
        "start_position": start_pose["position"],
        "end_position": end_pose["position"],
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Habitat HM3D + RAG 目标物品查找 Demo")
    p.add_argument("--scene", required=True, help="HM3D 场景 .glb 或 .json 路径")
    p.add_argument("--target", required=True, help="目标物品名称，如 'watercup' 或中文名")
    p.add_argument("--max_steps", type=int, default=1000, help="最大探索步数")
    p.add_argument("--rotate_sweep", type=int, default=12, help="初始环扫旋转次数(每次~30度)")
    p.add_argument("--dataset_config", type=str, default=None, help="Habitat 场景数据集配置文件(scene_dataset_config.json) 路径，如 hm3d/mp3d 对应的配置")
    # 不再要求手动指定保存目录：若未提供，将自动创建 run_logs/obs_<ts>_<scene>_<target>
    p.add_argument("--save_obs_dir", type=str, default=None, help="可选：保存观测/路径/计划的根目录；若不提供将自动生成")
    # 可选调参：脱困参数
    p.add_argument("--stuck_spread_thresh", type=float, default=1.2, help="判定在同一房间打转的平面范围阈值(米)")
    p.add_argument("--relocate_moves", type=int, default=24, help="脱困时直行的步数（每6步右转一次）")
    p.add_argument("--relocate_cooldown", type=float, default=10.0, help="两次脱困之间的最小时间间隔(秒)")
    p.add_argument("--explore_policy", type=str, default="frontier", choices=["simple", "frontier"], 
                   help="探索策略：'simple'=简单策略, 'frontier'=前沿探索+知识引导(默认)")
    p.add_argument("--fusion_mode", type=str, default="triangulate", 
                   choices=["triangulate", "depth", "voxel_vote"],
                   help="融合模式：'triangulate'=RANSAC三角化(默认), 'depth'=深度反投影, 'voxel_vote'=微体素投票")
    return p


def main():
    args = build_argparser().parse_args()
    rag.log_info("启动 Habitat HM3D RAG 目标查找", {
        "scene": args.scene,
        "target": args.target,
        "max_steps": args.max_steps,
        "rotate_sweep": args.rotate_sweep,
        "dataset_config": args.dataset_config,
        "save_obs_dir": args.save_obs_dir or "<auto>",
        "stuck_spread_thresh": args.stuck_spread_thresh,
        "relocate_moves": args.relocate_moves,
        "relocate_cooldown": args.relocate_cooldown,
        "explore_policy": args.explore_policy,
        "fusion_mode": args.fusion_mode,
    })

    # 将数据集配置放到全局，供 Agent 包装读取（简化参数传递）
    global _dataset_config_path
    _dataset_config_path = args.dataset_config

    result = habitat_rag_search(
        scene_path=args.scene,
        target_object=args.target,
        max_steps=args.max_steps,
        rotate_sweep=args.rotate_sweep,
        save_obs_dir=args.save_obs_dir,
        explore_policy=args.explore_policy,
        fusion_mode=args.fusion_mode,
    )

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
