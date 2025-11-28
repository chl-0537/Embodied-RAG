# 微体素定位器（Micro-Voxel Localizer）实现说明

## 概述

本次实现了一个创新的微体素定位器，用于鲁棒地融合多个深度/三角化观测结果。该模块使用小尺寸的3D体素网格，通过投票机制来确定物体的3D位置，并提供不确定性度量（熵）。

## 核心特性

### 1. 体素投票机制

- **体素网格**：使用小尺寸的3D体素网格（默认0.4m × 0.4m × 0.4m）
- **体素大小**：默认0.02m（2厘米），提供高精度定位
- **投票机制**：每个候选点根据其置信度对相应的体素进行投票
- **位置估计**：最高投票数的体素中心作为估计位置

### 2. 候选点收集

体素投票模式收集两种类型的候选点：

1. **深度反投影点**：从深度观测中直接反投影得到的3D点
2. **三角化结果点**：通过RANSAC三角化得到的3D点

### 3. 不确定性度量

- **熵计算**：使用直方图熵来衡量位置估计的不确定性
- **置信度调整**：高熵（高不确定性）会降低最终置信度
- **归一化**：熵值归一化到[0, 1]范围

### 4. 融合模式切换

支持三种融合模式：

- **`triangulate`**（默认）：使用RANSAC三角化
- **`depth`**：优先使用深度反投影
- **`voxel_vote`**：使用微体素投票融合所有候选点

## 实现细节

### 1. 体素网格配置

```python
voxel_size = 0.02  # 体素大小（米）
grid_width = 0.4   # 网格宽度（米）
grid_height = 0.4  # 网格高度（米）
```

**网格尺寸计算**：
- X方向体素数：`n_voxels_x = ceil(grid_width / voxel_size) = 20`
- Y方向体素数：`n_voxels_y = ceil(grid_height / voxel_size) = 20`
- Z方向体素数：`n_voxels_z = ceil(grid_height / voxel_size) = 20`
- 总体素数：`20 × 20 × 20 = 8000`

### 2. 体素投票算法

```python
def _voxel_vote_localize(candidate_points, candidate_confidences, reference_point):
    # 1. 确定网格中心（使用参考点或候选点加权中心）
    # 2. 初始化体素直方图
    # 3. 对每个候选点：
    #    - 计算体素索引
    #    - 使用置信度作为权重投票
    # 4. 找到最高投票的体素
    # 5. 计算体素中心位置
    # 6. 计算置信度（归一化的最大投票数）
    # 7. 计算熵（不确定性）
    return estimated_position, confidence, entropy
```

### 3. 候选点收集流程

```python
def _estimate_with_voxel_vote():
    candidate_points = []
    candidate_confidences = []
    
    # 1. 收集深度反投影点
    for obs in buffer:
        if obs.type == "depth" and obs.world_position:
            candidate_points.append(obs.world_position)
            candidate_confidences.append(obs.confidence)
    
    # 2. 收集三角化结果点
    if len(buffer) >= 2:
        triangulated_pos, inlier_ratio, ransac_confidence = _ransac_triangulate(...)
        if triangulated_pos:
            candidate_points.append(triangulated_pos)
            candidate_confidences.append(ransac_confidence * visual_confidence)
    
    # 3. 使用体素投票融合
    estimated_pos, confidence, entropy = _voxel_vote_localize(...)
    
    return estimated_pos
```

### 4. 熵计算

```python
# 归一化直方图
normalized_counts = voxel_counts / total_votes

# 避免log(0)
normalized_counts = normalized_counts + 1e-10

# 计算熵: H = -sum(p * log(p))
entropy = -sum(normalized_counts * log(normalized_counts))

# 归一化到[0, 1]
max_entropy = log(n_voxels_x * n_voxels_y * n_voxels_z)
normalized_entropy = entropy / max_entropy
```

### 5. 置信度调整

```python
# 使用熵调整置信度（高熵 = 低置信度）
entropy_penalty = 1.0 - entropy * 0.5  # 最多降低50%置信度
final_confidence = voxel_confidence * entropy_penalty
```

## 使用示例

### 基本使用

```python
# 创建融合模块，使用体素投票模式
fusion_module = SpatialSemanticFusionModule(
    max_buffer=10,
    min_views=3,
    camera_intrinsics=agent.camera_intrinsics,
    alpha=0.7,
    lambda_spatial=1.0,
    log_dir=save_obs_dir,
    fusion_mode="voxel_vote",  # 使用体素投票模式
    voxel_size=0.02,           # 体素大小（米）
    grid_width=0.4,            # 网格宽度（米）
    grid_height=0.4            # 网格高度（米）
)

# 估计位置
estimated_pos = fusion_module.estimate_object_position()
```

### 命令行使用

```bash
# 使用默认的三角化模式
python habitat_hm3d_rag_search.py --scene ... --target ... --fusion_mode triangulate

# 使用深度反投影模式
python habitat_hm3d_rag_search.py --scene ... --target ... --fusion_mode depth

# 使用微体素投票模式
python habitat_hm3d_rag_search.py --scene ... --target ... --fusion_mode voxel_vote
```

## 日志记录

体素投票模式会在`fusion_trace.json`中记录以下信息：

```json
{
  "timestamp": "2024-01-01T12:00:00.123456+00:00",
  "n_observations": 5,
  "observations": [...],
  "method": "voxel_vote",
  "estimated_position": [1.2, 2.3, 3.4],
  "confidence": 0.82,
  "inlier_ratio": null,
  "entropy": 0.35,
  "anomaly_flags": []
}
```

**新增字段**：
- `entropy`：体素投票的熵值（不确定性度量）

## 优势与特点

### 1. 鲁棒性

- **多源融合**：同时利用深度和三角化结果
- **异常处理**：自动过滤异常射线和候选点
- **投票机制**：通过多数投票减少异常值的影响

### 2. 精度

- **高分辨率**：2厘米的体素大小提供高精度定位
- **加权投票**：使用置信度作为权重，提高准确性
- **不确定性度量**：熵值帮助评估估计的可靠性

### 3. 灵活性

- **模式切换**：支持三种融合模式，便于对比实验
- **参数可调**：体素大小和网格尺寸可配置
- **易于集成**：与现有系统无缝集成

## 参数配置

### 体素大小（voxel_size）

- **默认值**：0.02m（2厘米）
- **调整建议**：
  - 小物体（<10cm）：0.01m
  - 中等物体（10-50cm）：0.02m
  - 大物体（>50cm）：0.05m

### 网格尺寸（grid_width, grid_height）

- **默认值**：0.4m × 0.4m
- **调整建议**：
  - 小物体：0.2m × 0.2m
  - 中等物体：0.4m × 0.4m
  - 大物体：0.8m × 0.8m

### 融合模式（fusion_mode）

- **`triangulate`**：适合多视角观测场景
- **`depth`**：适合深度数据可靠的场景
- **`voxel_vote`**：适合需要融合多种观测的场景

## 性能考虑

### 计算复杂度

- **体素投票**：O(N × M)，其中N是候选点数，M是体素数
- **典型值**：N ≈ 10，M ≈ 8000，总计算量 ≈ 80,000次操作
- **实际性能**：在现代CPU上，单次投票 < 1ms

### 内存使用

- **体素直方图**：8000个float32 ≈ 32KB
- **候选点**：10个点 × 3个float64 ≈ 240字节
- **总内存**：< 50KB

## 实验建议

### 消融实验

1. **模式对比**：比较三种融合模式的性能
2. **体素大小**：测试不同体素大小对精度的影响
3. **网格尺寸**：测试不同网格尺寸对鲁棒性的影响

### 评估指标

- **定位精度**：估计位置与真实位置的误差
- **置信度准确性**：置信度与实际精度的相关性
- **熵值有效性**：熵值与实际不确定性的相关性

## 未来改进

1. **自适应体素大小**：根据物体大小自动调整体素大小
2. **多尺度投票**：使用多个尺度的体素网格进行投票
3. **时间融合**：融合历史估计结果，提高稳定性
4. **GPU加速**：使用GPU加速体素投票计算

## 技术细节

### 体素索引计算

```python
# 计算相对于网格中心的偏移
offset = point - grid_center

# 计算体素索引
voxel_x = int((offset[0] + grid_half_width) / voxel_size)
voxel_y = int((offset[1] + grid_half_height) / voxel_size)
voxel_z = int((offset[2] + grid_half_depth) / voxel_size)
```

### 体素中心位置计算

```python
voxel_center_x = grid_center[0] - grid_half_width + (voxel_x + 0.5) * voxel_size
voxel_center_y = grid_center[1] - grid_half_height + (voxel_y + 0.5) * voxel_size
voxel_center_z = grid_center[2] - grid_half_depth + (voxel_z + 0.5) * voxel_size
```

### 异常检测

体素投票模式集成了以下异常检测：

- **射线发散检测**：拒绝与上一个可靠估计差异过大的射线
- **低内点比例**：标记低内点比例的三角化结果
- **估计发散**：检测估计位置与上一个可靠估计的差异

## 注意事项

1. **参考点选择**：参考点用于确定体素网格中心，优先使用最后一个可靠估计
2. **候选点质量**：确保候选点质量，过滤异常观测
3. **网格范围**：确保所有候选点都在网格范围内
4. **熵值解释**：低熵（<0.3）表示高置信度，高熵（>0.7）表示低置信度

