# 融合过程追踪日志实现说明

## 概述

本次实现为`SpatialSemanticFusionModule`添加了详细的融合过程追踪日志功能。每次调用`estimate_object_position()`时，都会将完整的融合过程信息追加到`fusion_trace.json`文件中，便于实验分析和调试。

## 功能特性

### 1. 日志文件管理

- **文件位置**：`{save_obs_dir}/fusion_trace.json`
- **格式**：JSON数组，每个元素代表一次融合估计
- **编码**：UTF-8，支持中文
- **原子性写入**：使用临时文件确保写入的原子性

### 2. 日志条目结构

每个日志条目包含以下字段：

```json
{
  "timestamp": "2024-01-01T12:00:00.000000+00:00",
  "n_observations": 5,
  "observations": [
    {
      "ray_origin": [1.0, 2.0, 3.0],
      "ray_dir": [0.1, 0.2, 0.9],
      "depth": 2.5,
      "bbox": [100, 200, 150, 250],
      "type": "depth",
      "confidence": 0.85,
      "world_position": [1.5, 2.5, 3.5]
    }
  ],
  "method": "depth",
  "estimated_position": [1.2, 2.3, 3.4],
  "confidence": 0.82,
  "inlier_ratio": 0.75,
  "anomaly_flags": []
}
```

### 3. 字段说明

| 字段 | 类型 | 说明 |
|-----|------|------|
| `timestamp` | string | ISO 8601格式的时间戳（UTC） |
| `n_observations` | int | 参与融合的观测数量 |
| `observations` | array | 观测数据列表 |
| `method` | string | 融合方法：`"depth"`、`"triangulation"`、`"voxel_vote"` |
| `estimated_position` | array/null | 估计的3D位置 `[x, y, z]`，失败时为`null` |
| `confidence` | float | 融合后的置信度（0.0-1.0） |
| `inlier_ratio` | float/null | 内点比例（仅用于triangulation），其他方法为`null` |
| `anomaly_flags` | array | 异常标志列表，如`["diverged_from_last_estimate", "low_inlier_ratio"]` |

#### 观测数据字段

| 字段 | 类型 | 说明 |
|-----|------|------|
| `ray_origin` | array | 射线起点（agent位置）`[x, y, z]` |
| `ray_dir` | array/null | 射线方向`[dx, dy, dz]`，深度方法为`null` |
| `depth` | float/null | 深度值（米），无深度时为`null` |
| `bbox` | array/null | 边界框`[x1, y1, x2, y2]`，无bbox时为`null` |
| `type` | string | 观测类型：`"depth"`、`"triangulation"` |
| `confidence` | float | 该观测的置信度 |
| `world_position` | array/null | 世界坐标位置（深度方法），其他为`null` |

### 4. 异常标志

异常标志用于标记融合过程中的异常情况：

| 标志 | 说明 |
|-----|------|
| `diverged_from_last_estimate` | 估计位置与上一个可靠估计差异过大（>2m） |
| `low_inlier_ratio` | RANSAC内点比例过低（<0.3） |
| `rejected_rays_count_N` | 有N条射线被拒绝 |
| `all_rays_rejected` | 所有射线都被拒绝 |
| `fallback_to_last_estimate` | 回退到上一个可靠估计 |
| `no_fallback_available` | 没有可用的回退估计 |
| `ransac_failed` | RANSAC三角化失败 |
| `high_anomaly_count` | 异常观测计数过高（>=3） |
| `suggest_multi_angle_rescan` | 建议执行多角度重新扫描 |

## 实现细节

### 1. 初始化

在`SpatialSemanticFusionModule.__init__()`中：

```python
def __init__(self, ..., log_dir: str = None):
    self.log_dir = log_dir
    self.fusion_trace_file = None
    
    if self.log_dir:
        os.makedirs(self.log_dir, exist_ok=True)
        self.fusion_trace_file = os.path.join(self.log_dir, "fusion_trace.json")
        # 如果文件不存在，创建空数组
        if not os.path.exists(self.fusion_trace_file):
            with open(self.fusion_trace_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
```

### 2. 日志记录方法

`_log_fusion_trace()`方法负责：

1. **读取现有日志**：从文件读取所有现有条目
2. **准备观测数据**：从buffer提取或使用提供的观测数据
3. **构建日志条目**：创建包含所有必需字段的字典
4. **原子性写入**：使用临时文件确保写入的原子性

### 3. 观测数据提取

从`buffer`中提取观测数据时：

- **射线方向**：对于triangulation类型，计算射线方向
- **深度值**：从world_position计算深度（agent到物体的距离）
- **边界框**：从bbox字段提取，或从bbox_center推断

### 4. 日志记录点

在`estimate_object_position()`的以下位置记录日志：

1. **深度反投影成功**：记录深度方法的结果
2. **RANSAC三角化成功**：记录三角化方法的结果
3. **所有射线被拒绝（有回退）**：记录回退情况
4. **所有射线被拒绝（无回退）**：记录失败情况
5. **RANSAC失败（有回退）**：记录回退情况
6. **RANSAC失败（无回退）**：记录失败情况

## 使用示例

### 基本使用

```python
# 在habitat_rag_search函数中
fusion_module = SpatialSemanticFusionModule(
    max_buffer=10,
    min_views=3,
    camera_intrinsics=agent.camera_intrinsics,
    alpha=0.7,
    lambda_spatial=1.0,
    log_dir=save_obs_dir  # 传递日志目录
)

# 每次调用estimate_object_position()时自动记录
estimated_pos = fusion_module.estimate_object_position()
```

### 读取日志

```python
import json

# 读取融合追踪日志
with open("run_logs/obs_xxx/fusion_trace.json", "r", encoding="utf-8") as f:
    trace_entries = json.load(f)

# 分析日志
for entry in trace_entries:
    print(f"时间: {entry['timestamp']}")
    print(f"方法: {entry['method']}")
    print(f"位置: {entry['estimated_position']}")
    print(f"置信度: {entry['confidence']}")
    print(f"观测数: {entry['n_observations']}")
    if entry['anomaly_flags']:
        print(f"异常标志: {entry['anomaly_flags']}")
    print()
```

## 日志分析

### 1. 统计信息

```python
# 统计不同方法的使用次数
method_counts = {}
for entry in trace_entries:
    method = entry['method']
    method_counts[method] = method_counts.get(method, 0) + 1

print("方法使用统计:", method_counts)
```

### 2. 置信度分析

```python
# 分析置信度分布
confidences = [entry['confidence'] for entry in trace_entries if entry['estimated_position']]
avg_confidence = sum(confidences) / len(confidences)
print(f"平均置信度: {avg_confidence:.3f}")
```

### 3. 异常分析

```python
# 统计异常情况
anomaly_types = {}
for entry in trace_entries:
    for flag in entry['anomaly_flags']:
        anomaly_types[flag] = anomaly_types.get(flag, 0) + 1

print("异常类型统计:", anomaly_types)
```

### 4. 位置轨迹

```python
# 提取位置轨迹
positions = []
for entry in trace_entries:
    if entry['estimated_position']:
        positions.append(entry['estimated_position'])

# 可视化位置轨迹
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
positions = np.array(positions)
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'o-')
plt.show()
```

## 文件格式示例

```json
[
  {
    "timestamp": "2024-01-01T12:00:00.123456+00:00",
    "n_observations": 3,
    "observations": [
      {
        "ray_origin": [1.0, 2.0, 3.0],
        "ray_dir": [0.1, 0.2, 0.9],
        "depth": null,
        "bbox": [100, 200, 150, 250],
        "type": "triangulation",
        "confidence": 0.85,
        "world_position": null
      },
      {
        "ray_origin": [1.1, 2.1, 3.1],
        "ray_dir": null,
        "depth": 2.5,
        "bbox": [110, 210, 160, 260],
        "type": "depth",
        "confidence": 0.90,
        "world_position": [1.5, 2.5, 3.5]
      }
    ],
    "method": "depth",
    "estimated_position": [1.2, 2.3, 3.4],
    "confidence": 0.82,
    "inlier_ratio": null,
    "anomaly_flags": []
  },
  {
    "timestamp": "2024-01-01T12:00:01.234567+00:00",
    "n_observations": 5,
    "observations": [...],
    "method": "triangulation",
    "estimated_position": [1.3, 2.4, 3.5],
    "confidence": 0.75,
    "inlier_ratio": 0.65,
    "anomaly_flags": ["low_inlier_ratio"]
  }
]
```

## 性能考虑

### 1. 文件I/O

- **追加写入**：每次只追加一个条目，不重写整个文件
- **原子性**：使用临时文件确保写入的原子性
- **缓冲**：可以考虑批量写入以提高性能（未来优化）

### 2. 内存使用

- **观测数据**：每个观测包含完整信息，可能占用较多内存
- **建议**：对于长时间运行，可以考虑定期归档或压缩旧日志

### 3. 文件大小

- **单次估计**：约1-5KB（取决于观测数量）
- **100次估计**：约100-500KB
- **1000次估计**：约1-5MB

## 错误处理

### 1. 文件读取失败

如果读取现有日志失败，会创建新的日志文件：

```python
try:
    with open(self.fusion_trace_file, "r", encoding="utf-8") as f:
        trace_entries = json.load(f)
except (json.JSONDecodeError, IOError) as e:
    rag.log_warning(f"读取融合追踪日志失败，创建新文件: {e}")
    trace_entries = []
```

### 2. 写入失败

如果写入失败，会记录警告但不影响主流程：

```python
try:
    # 写入逻辑
    ...
except Exception as e:
    rag.log_warning(f"记录融合追踪日志失败: {e}")
```

## 未来改进

1. **批量写入**：累积多个条目后批量写入，减少I/O次数
2. **日志压缩**：定期压缩旧日志，节省存储空间
3. **日志轮转**：当日志文件过大时自动轮转
4. **实时可视化**：提供实时可视化工具查看融合过程
5. **统计分析**：自动生成统计报告

## 注意事项

1. **日志目录**：确保`log_dir`参数正确传递，否则不会记录日志
2. **文件权限**：确保有写入权限
3. **磁盘空间**：长时间运行可能产生大量日志，注意磁盘空间
4. **性能影响**：日志记录会略微影响性能，但通常可以忽略

