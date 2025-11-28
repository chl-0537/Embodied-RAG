# 调试信息保存说明

本文档说明程序运行过程中保存的中间信息，便于后续调试和分析。

## 目录结构

运行程序后，会在 `run_logs/obs_<timestamp>_<scene>_<target>/` 目录下创建以下子目录：

```
run_logs/
└── obs_<timestamp>_<scene>_<target>/
    ├── sweep/                    # 初始环扫阶段的RGB图像
    ├── explore/                  # 探索阶段的RGB图像
    ├── depth/                    # 深度图（归一化版本，便于可视化）
    ├── detections/               # 视觉检测结果（JSON格式）
    ├── frontiers/                # 前沿点信息（JSON格式，仅前沿探索策略）
    ├── detected_objects/         # 检测到的物体截图
    ├── path_trajectory.json      # 路径轨迹日志
    ├── plan.json                 # 找到目标后的行动计划
    ├── target_object.png         # 目标物体截图
    └── target_object_annotated.png  # 目标物体标注图
```

## 文件说明

### 1. RGB图像 (`sweep/`, `explore/`)

- **格式**: PNG图像
- **命名**:
  - 环扫: `sweep_000.png`, `sweep_001.png`, ...
  - 探索: `step_00001.png`, `step_00002.png`, ...
  - 缓冲扫描: `buffer_scan_00060_000.png`, ...
- **用途**: 查看机器人每个步骤观察到的场景

### 2. 深度图 (`depth/`)

- **格式**: PNG图像（8位灰度图，归一化到0-255）
- **命名**: 
  - 环扫: `sweep_000_depth.png`, ...
  - 探索: `step_00001_depth.png`, ...
  - 缓冲扫描: `buffer_scan_00060_000_depth.png`, ...
- **深度范围**: 0-10米（归一化）
- **用途**: 
  - 分析深度信息
  - 调试前沿检测
  - 分析未探索区域检测

### 3. 检测结果 (`detections/`)

- **格式**: JSON文件
- **命名**: 
  - 环扫: `sweep_000_detection.json`, ...
  - 探索: `step_00001_detection.json`, ...
  - 缓冲扫描: `buffer_scan_00060_000_detection.json`, ...
- **内容结构**:
```json
{
  "objects": [
    {
      "label": "水杯",
      "bbox": [100, 200, 150, 250],
      "confidence": 0.95,
      "description": "蓝色水杯在书桌上"
    }
  ],
  "scene_description": "整体场景描述",
  "detection_summary": "检测总结"
}
```
- **用途**: 
  - 分析检测准确性
  - 调试物体匹配逻辑
  - 分析误检和漏检

### 4. 前沿点信息 (`frontiers/`, 仅前沿探索策略)

- **格式**: JSON文件
- **命名**: `step_00001_frontiers.json`, ...
- **内容结构**:
```json
{
  "current_position": [1.23, 0.0, 4.56],
  "frontier_count": 5,
  "frontiers": [
    {"x": 2.34, "z": 5.67},
    {"x": 3.45, "z": 6.78}
  ],
  "timestamp": 1234567890.123
}
```
- **用途**: 
  - 分析前沿检测算法
  - 调试路径规划
  - 分析探索策略效果

### 5. 物体截图 (`detected_objects/`)

- **格式**: PNG图像
- **命名**: `obj_00001_00_水杯_cup_conf0.95.png`
  - 格式: `obj_{step:05d}_{idx:02d}_{label}_conf{confidence:.2f}.png`
- **用途**: 
  - 查看检测到的每个物体
  - 分析检测质量
  - 调试物体匹配

### 6. 路径轨迹 (`path_trajectory.json`)

- **格式**: JSON文件
- **内容结构**:
```json
{
  "start_time": 1234567890.0,
  "end_time": 1234568000.0,
  "duration_seconds": 110.0,
  "total_steps": 500,
  "start_position": [0.0, 0.0, 0.0],
  "end_position": [5.0, 0.0, 3.0],
  "trajectory": [
    {
      "timestamp": 1234567890.0,
      "step": 0,
      "phase": "start",
      "action": "reset",
      "position": [0.0, 0.0, 0.0],
      "rotation": [0.0, 0.0, 0.0, 1.0]
    }
  ]
}
```
- **用途**: 
  - 分析机器人运动轨迹
  - 调试路径规划
  - 分析探索效率

## 调试建议

### 1. 分析检测准确性

1. 查看 `detections/` 目录中的检测结果
2. 对比对应的RGB图像 (`sweep/` 或 `explore/`)
3. 检查物体截图 (`detected_objects/`)
4. 分析误检和漏检的原因

### 2. 调试前沿探索

1. 查看 `frontiers/` 目录中的前沿点信息
2. 对比对应的深度图 (`depth/`)
3. 分析前沿点是否合理
4. 检查路径规划是否正确

### 3. 分析探索效率

1. 查看 `path_trajectory.json` 中的轨迹
2. 分析机器人是否在重复探索
3. 检查是否卡在某个区域
4. 优化探索策略参数

### 4. 调试物体匹配

1. 查看 `detected_objects/` 中的物体截图
2. 对比检测结果 (`detections/`)
3. 分析匹配逻辑是否正确
4. 检查同义词映射是否准确

## 文件大小估算

- RGB图像: ~500KB/张
- 深度图: ~200KB/张
- 检测结果JSON: ~5KB/个
- 前沿信息JSON: ~1KB/个
- 物体截图: ~50KB/个

对于1000步的探索，预计总大小约：
- RGB: 500MB
- 深度图: 200MB
- 其他: ~10MB
- **总计**: ~710MB

## 注意事项

1. **存储空间**: 长时间运行会产生大量文件，注意磁盘空间
2. **性能影响**: 保存中间信息会略微影响运行速度
3. **选择性保存**: 可以通过修改代码选择性地保存某些信息
4. **清理旧日志**: 定期清理 `run_logs/` 目录中的旧日志

## 快速查看脚本

可以编写简单的脚本来快速查看保存的信息：

```python
import json
import os
from PIL import Image

# 查看检测结果
with open("run_logs/obs_xxx/detections/step_00001_detection.json") as f:
    detection = json.load(f)
    print(f"检测到 {len(detection['objects'])} 个物体")

# 查看前沿点
with open("run_logs/obs_xxx/frontiers/step_00001_frontiers.json") as f:
    frontiers = json.load(f)
    print(f"检测到 {frontiers['frontier_count']} 个前沿点")

# 查看深度图
depth_img = Image.open("run_logs/obs_xxx/depth/step_00001_depth.png")
depth_img.show()
```


