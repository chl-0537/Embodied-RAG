# RANSAC三角化实现说明

## 概述

本次实现添加了基于RANSAC（Random Sample Consensus）的鲁棒三角化方法，用于提高多视角物体定位的鲁棒性，特别是在存在噪声和异常值的情况下。

## 主要功能

### 1. 核心方法

#### `_point_to_ray_distance()`

计算点到射线的距离（残差）：

```python
distance = ||point - (ray_origin + t * ray_direction)||
```

用于判断一个3D点是否与某条射线一致。

#### `_triangulate_two_rays()`

使用线性最小二乘法三角化两条射线：

- 构建投影矩阵约束
- 使用SVD求解
- 返回估计的3D点

#### `_ransac_triangulate()`

RANSAC三角化主方法：

**算法流程：**
1. 随机选择2条射线
2. 三角化这两条射线得到候选点
3. 计算所有射线到候选点的残差
4. 统计内点（残差 < 阈值）
5. 重复多次，选择内点数最多的候选点
6. 使用所有内点重新三角化（提高精度）

**参数：**
- `max_iterations`: 最大迭代次数（默认100）
- `inlier_threshold`: 内点阈值（默认0.15米）
- `min_inliers`: 最少内点数（默认2）

**返回：**
- `best_point`: 最佳估计点
- `inlier_ratio`: 内点比例
- `confidence`: 置信度分数

### 2. 置信度计算

置信度公式：
```python
view_factor = 1.0 - exp(-num_views / 3.0)
confidence = inlier_ratio * view_factor
```

- **内点比例**：反映观测的一致性
- **视角因子**：反映观测数量的充分性
- **最终置信度**：两者的乘积

### 3. 集成到估计流程

**`estimate_object_position()` 更新：**

1. **优先级**：
   - 深度反投影（如果有）
   - RANSAC三角化（如果没有深度观测）

2. **失败阈值**：
   - 内点比例 < 30%：发出警告，建议收集更多观测
   - 仍然返回结果，但置信度较低

3. **最终置信度**：
   ```python
   final_confidence = ransac_confidence * visual_confidence
   ```
   - 融合RANSAC置信度和视觉检测置信度

## 实现细节

### RANSAC循环

```python
for iteration in range(max_iterations):
    # 1. 随机选择2条射线
    indices = random.choice(n_rays, size=2)
    
    # 2. 三角化
    candidate = triangulate_two_rays(positions[idx1], rays[idx1], ...)
    
    # 3. 计算内点
    inliers = [i for i in range(n_rays) 
               if distance(candidate, positions[i], rays[i]) < threshold]
    
    # 4. 更新最佳结果
    if len(inliers) > best_inlier_count:
        best_point = candidate
        best_inliers = inliers
```

### 内点精化

找到最佳内点集后，使用所有内点重新三角化：

```python
inlier_positions = [positions[i] for i in best_inliers]
inlier_rays = [rays[i] for i in best_inliers]
refined_point = triangulate_rays(inlier_positions, inlier_rays)
```

这提高了最终估计的精度。

### 错误处理

- **观测数不足**：返回 `(None, 0.0, 0.0)`
- **内点数不足**：返回 `(None, 0.0, 0.0)` 并记录警告
- **数值异常**：检查NaN/Inf值

## 使用示例

### 在估计位置时自动使用

```python
# 自动使用RANSAC三角化（如果没有深度观测）
estimated_pos, inlier_ratio, confidence = fusion_module.estimate_object_position()

if estimated_pos is not None:
    if inlier_ratio < 0.3:
        # 内点比例过低，建议收集更多观测
        print("警告：内点比例过低，建议收集更多观测")
```

## 单元测试

测试文件：`test_ransac_triangulation.py`

### 测试用例

1. **完美匹配**（无噪声）
   - 5个观测，所有射线精确指向真实点
   - 期望：误差 < 1cm，内点比例 ≥ 90%

2. **有噪声匹配**
   - 8个观测，射线方向有5%噪声
   - 期望：误差 < 20cm，内点比例 ≥ 50%

3. **异常值较多**
   - 10个观测，其中5个是异常值（50%异常率）
   - 期望：误差 < 50cm，内点比例 ≥ 40%

4. **最少观测数**
   - 2个观测（最少要求）
   - 期望：能够正常工作

5. **观测数不足**
   - 1个观测
   - 期望：正确返回None

### 运行测试

```bash
python test_ransac_triangulation.py
```

## 优势

1. **鲁棒性**：能够处理噪声和异常值
2. **自适应性**：自动识别内点和异常值
3. **置信度评估**：提供内点比例和置信度分数
4. **精度提升**：使用内点精化提高最终精度

## 参数调优

### 内点阈值 (`inlier_threshold`)

- **默认值**：0.15米（15cm）
- **较小值**（如0.05m）：更严格，适合高精度场景
- **较大值**（如0.3m）：更宽松，适合噪声较大的场景

### 最大迭代次数 (`max_iterations`)

- **默认值**：100
- **计算复杂度**：O(max_iterations * n_rays)
- **建议**：根据观测数量调整
  - 少量观测（<5）：50次足够
  - 中等观测（5-10）：100次
  - 大量观测（>10）：200次

### 失败阈值 (`fail_threshold`)

- **默认值**：0.3（30%内点比例）
- **调整建议**：
  - 严格场景：0.5（50%）
  - 宽松场景：0.2（20%）

## 日志输出

### 成功情况

```
RANSAC三角化：找到 8/10 内点，内点比例: 0.800, 置信度: 0.750
空间融合模块：使用RANSAC三角化估计3D位置 (2.123, 3.456, 5.789), 
内点比例: 0.800, RANSAC置信度: 0.750, 最终置信度: 0.675 (观测数: 10)
```

### 警告情况

```
RANSAC三角化：内点数不足 (1 < 2)
空间融合模块：RANSAC内点比例过低 (0.200 < 0.300)，建议收集更多观测以提高精度
```

## 性能考虑

1. **计算复杂度**：
   - RANSAC循环：O(max_iterations * n_rays)
   - 内点精化：O(n_inliers^2)
   - 总体：O(max_iterations * n_rays + n_inliers^2)

2. **优化建议**：
   - 对于大量观测，可以限制最大迭代次数
   - 可以提前终止（如果找到足够好的内点集）

## 未来改进

1. **自适应迭代次数**：根据观测数量动态调整
2. **并行化**：并行计算多个候选点
3. **加权RANSAC**：根据观测置信度加权
4. **多模型RANSAC**：处理多个物体的情况
5. **不确定性估计**：计算估计位置的协方差矩阵

## 技术细节

### 点到射线距离计算

使用向量投影方法：
```python
vec_to_point = point - ray_origin
t = dot(vec_to_point, ray_direction)  # 投影长度
closest_point = ray_origin + t * ray_direction
distance = ||point - closest_point||
```

### 两条射线三角化

使用投影矩阵方法：
```python
proj = I - ray @ ray^T
minimize ||proj * (P - pos)||^2
```

### 多条射线三角化

使用最小二乘法：
```python
minimize sum_i ||(I - ray_i @ ray_i^T) * (P - pos_i)||^2
```

