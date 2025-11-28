# 空间语义融合模块（Spatial-Semantic Fusion Module）说明

## 概述

空间语义融合模块通过多视角几何三角化实现小物体的精确3D定位，融合视觉检测、机器人位姿和记忆库信息，提供更准确的空间定位能力。

## 功能特性

### 1. 多帧检测融合（Multi-view Fusion）

- **缓冲区管理**：维护最多10帧的检测结果
- **观测信息**：每帧包含：
  - 机器人位置和旋转（世界坐标）
  - 物体边界框中心（归一化图像坐标）
  - 检测置信度
  - 物体标签
  - 时间戳

### 2. 多视几何三角化定位

- **射线反投影**：将像素坐标转换为世界坐标系下的射线
- **三角化算法**：使用SVD求解最小二乘问题，找到多条射线的最接近交点
- **验证机制**：
  - 检查条件数（避免共线情况）
  - 验证估计位置是否在合理范围内（距离观测点不超过10米）

### 3. 空间语义加权更新

- **融合公式**：
  ```
  C_final = α * C_vision + (1 - α) * exp(-||P* - P_memory|| / λ)
  ```
  - `α = 0.7`：视觉置信度权重
  - `λ = 1.0`：空间距离衰减系数（米）
  - `P*`：估计的3D位置
  - `P_memory`：记忆库中的历史位置

### 4. 自动内参调整

- 根据实际图像尺寸自动调整相机内参
- 默认内参（640x480）：`fx=fy=320, cx=320, cy=240`
- 支持任意图像尺寸

## 类接口

### `SpatialSemanticFusionModule`

```python
class SpatialSemanticFusionModule:
    def __init__(self, max_buffer=10, min_views=3, 
                 fx=320.0, fy=320.0, cx=320.0, cy=240.0,
                 alpha=0.7, lambda_spatial=1.0):
        """
        初始化空间语义融合模块
        
        Args:
            max_buffer: 缓冲区最大帧数（默认10）
            min_views: 进行三角化所需的最少视角数（默认3）
            fx, fy, cx, cy: 相机内参（默认适配640x480）
            alpha: 视觉置信度权重（默认0.7）
            lambda_spatial: 空间距离衰减系数（默认1.0米）
        """
    
    def add_observation(self, position, rotation, bbox, confidence, label, img_shape=None):
        """添加一次观测到缓冲区"""
    
    def estimate_object_position(self) -> Optional[np.ndarray]:
        """估计目标物体的3D位置，返回 [x, y, z] 或 None"""
    
    def get_fused_confidence(self, vision_confidence, memory_position=None) -> float:
        """计算融合后的置信度"""
    
    def clear_buffer(self):
        """清空缓冲区"""
```

## 集成位置

### 1. 环扫阶段

- 在环扫开始前初始化 `sweep_fusion_module`
- 每次检测到目标物体时调用 `add_observation`
- 当有≥3个观测时，自动估计3D位置
- 找到目标后，将估计的3D位置添加到返回结果和计划中

### 2. 探索主循环

- 在探索循环开始前初始化 `fusion_module`
- 每次检测到目标物体时调用 `add_observation`
- 当有≥3个观测时，自动估计3D位置
- 融合视觉置信度和记忆库位置信息
- 找到目标后，将估计的3D位置添加到返回结果和计划中

### 3. 缓冲区管理

- 连续5次未检测到目标时，自动清空缓冲区
- 避免累积错误的观测数据

## 输出信息

### 1. 日志输出

- 每次添加观测：`空间融合模块：添加观测 (缓冲区大小: X/10)`
- 估计3D位置：`空间融合模块：估计3D位置 (x, y, z), 平均置信度: X.XXX`
- 融合置信度：`空间融合：估计3D位置 [...], 融合置信度: X.XXX`

### 2. 返回结果

在找到目标后，返回结果包含：
```python
{
    "status": "SUCCESS",
    "found": True,
    "estimated_3d_position": [x, y, z],  # 估计的3D位置
    ...
}
```

### 3. 路径轨迹

路径轨迹日志中包含：
```json
{
    "estimated_3d_position": [x, y, z],
    ...
}
```

### 4. 行动计划

`plan.json` 中包含：
```json
{
    "estimated_3d_position": [x, y, z],
    "spatial_fusion_method": "multi_view_triangulation",
    ...
}
```

## 算法细节

### 1. 射线反投影

```python
# 归一化像素坐标 -> 像素坐标
u = u_norm * img_width
v = v_norm * img_height

# 相机坐标系下的射线方向
x_cam = (u - cx) / fx
y_cam = (v - cy) / fy
z_cam = 1.0
ray_cam = normalize([x_cam, y_cam, z_cam])

# 转换到世界坐标系
ray_world = R @ ray_cam
```

### 2. 三角化算法

使用投影矩阵方法：
```python
# 对于每条射线，构建投影矩阵
proj_i = I - ray_i @ ray_i^T

# 最小化 sum_i ||proj_i * (P - pos_i)||^2
# 转换为线性系统 A * P = b
A = [proj_1; proj_2; ...]
b = [proj_1 @ pos_1; proj_2 @ pos_2; ...]

# 使用SVD求解
U, s, Vt = svd(A)
P = Vt^T @ (U^T @ b / s)
```

### 3. 置信度融合

```python
if memory_position exists:
    distance = ||estimated_pos - memory_position||
    spatial_consistency = exp(-distance / lambda_spatial)
    C_final = alpha * C_vision + (1 - alpha) * spatial_consistency
else:
    C_final = alpha * C_vision + (1 - alpha) * C_multi_view
```

## 使用示例

模块会自动在检测到目标物体时工作，无需手动调用。但可以通过日志查看工作状态：

```bash
# 查看日志中的融合信息
grep "空间融合" run_logs/obs_*/path_trajectory.json

# 查看估计的3D位置
cat run_logs/obs_*/plan.json | jq .estimated_3d_position
```

## 参数调优

### 相机内参

如果知道实际的相机内参，可以在初始化时指定：
```python
fusion_module = SpatialSemanticFusionModule(
    fx=实际fx值,
    fy=实际fy值,
    cx=实际cx值,
    cy=实际cy值
)
```

### 融合权重

调整 `alpha` 参数可以改变视觉置信度和空间一致性的权重：
- `alpha=0.9`：更依赖视觉置信度
- `alpha=0.5`：平衡视觉和空间信息
- `alpha=0.3`：更依赖空间一致性

### 缓冲区大小

- `max_buffer=5`：更快的响应，但可能精度较低
- `max_buffer=20`：更高的精度，但需要更多观测

## 注意事项

1. **最少视角数**：需要至少3个不同视角的观测才能进行三角化
2. **观测质量**：观测点应该分散在不同位置，避免共线
3. **距离限制**：估计位置距离观测点不应超过10米（可调整）
4. **缓冲区清理**：连续5次未检测到目标时自动清空，避免累积错误

## 技术细节

- **坐标系**：使用Habitat的世界坐标系（右手系）
- **旋转表示**：四元数 [x, y, z, w]
- **图像坐标**：归一化到 [0, 1] 范围
- **数值稳定性**：使用SVD求解，检查条件数

## 未来改进

1. **自适应内参**：从Habitat-Sim获取实际相机内参
2. **不确定性估计**：计算估计位置的置信区间
3. **动态权重**：根据观测质量动态调整融合权重
4. **多物体跟踪**：支持同时跟踪多个物体的3D位置

