# 相机内参重构说明

## 概述

本次重构移除了所有硬编码的相机内参（fx, fy, cx, cy），改为从Habitat-Sim的传感器配置中动态提取真实的相机内参。

## 主要更改

### 1. HabitatAgentWrapper 类

#### 新增方法：`_extract_camera_intrinsics`

从sensor_spec提取相机内参，包括：
- 分辨率（width, height）
- 水平视场角（HFOV）
- 计算的内参（fx, fy, cx, cy）

**内参计算公式：**
```python
fx = (width / 2) / tan(hfov / 2)
fy = fx  # 针孔模型通常 fy = fx
cx = width / 2
cy = height / 2
```

**HFOV获取策略（按优先级）：**
1. 从sensor_spec直接获取
2. 从simulator的agent配置中获取
3. 使用默认值60度（如果前两种方法都失败）

#### 新增属性：`camera_intrinsics`

在初始化后自动提取并存储相机内参字典：
```python
{
    "fx": float,
    "fy": float,
    "cx": float,
    "cy": float,
    "width": int,
    "height": int,
    "hfov_rad": float,
    "hfov_deg": float
}
```

### 2. SpatialSemanticFusionModule 类

#### 构造函数更改

**之前：**
```python
SpatialSemanticFusionModule(
    max_buffer=10,
    min_views=3,
    fx=320.0, fy=320.0, cx=320.0, cy=240.0,  # 硬编码
    alpha=0.7,
    lambda_spatial=1.0
)
```

**现在：**
```python
SpatialSemanticFusionModule(
    max_buffer=10,
    min_views=3,
    camera_intrinsics=agent.camera_intrinsics,  # 从agent获取
    alpha=0.7,
    lambda_spatial=1.0
)
```

#### 内部改进

- 从`camera_intrinsics`字典中提取所有内参值
- 移除了`add_observation`中根据图像尺寸动态调整内参的逻辑
- 添加了图像尺寸一致性检查（如果实际图像尺寸与内参不一致，发出警告）

### 3. 使用位置更新

所有创建`SpatialSemanticFusionModule`实例的地方都已更新：

1. **环扫阶段**（`habitat_rag_search`函数中）：
   ```python
   sweep_fusion_module = SpatialSemanticFusionModule(
       camera_intrinsics=agent.camera_intrinsics,
       ...
   )
   ```

2. **探索主循环**（`habitat_rag_search`函数中）：
   ```python
   fusion_module = SpatialSemanticFusionModule(
       camera_intrinsics=agent.camera_intrinsics,
       ...
   )
   ```

## 优势

1. **准确性**：使用真实的相机内参，而不是硬编码的近似值
2. **灵活性**：支持不同的图像分辨率和视场角配置
3. **可维护性**：内参集中管理，避免多处硬编码
4. **一致性**：确保内参与实际传感器配置一致

## 向后兼容性

- 如果`camera_intrinsics`为`None`，`SpatialSemanticFusionModule`会使用默认值（640x480，fx=fy=320）
- 这确保了代码的向后兼容性

## 日志输出

重构后的代码会在以下情况输出日志：

1. **内参提取成功**：
   ```
   提取相机内参 {'fx': 554.26, 'fy': 554.26, 'cx': 320.0, 'cy': 240.0, ...}
   ```

2. **空间融合模块初始化**：
   ```
   空间融合模块：使用相机内参 fx=554.26, fy=554.26, cx=320.00, cy=240.00, 分辨率=640x480
   ```

3. **HFOV未找到警告**：
   ```
   sensor_spec中未找到hfov，使用默认值60度 (传感器: rgb)
   ```

4. **图像尺寸不一致警告**：
   ```
   空间融合模块：图像尺寸 (800x600) 与内参尺寸 (640x480) 不一致，使用内参尺寸进行归一化
   ```

## 测试建议

1. **验证内参提取**：检查日志中的内参值是否合理
2. **不同分辨率**：测试不同图像分辨率下的内参计算
3. **不同HFOV**：测试不同视场角下的内参计算
4. **三角化精度**：验证使用真实内参后的3D定位精度是否提高

## 注意事项

1. **HFOV单位**：Habitat-Sim中的HFOV可能是弧度或度数，代码会自动处理
2. **图像尺寸一致性**：如果实际图像尺寸与内参中的尺寸不一致，会使用内参尺寸进行归一化，确保内参一致性
3. **默认值**：如果无法获取HFOV，使用60度作为默认值，这在大多数情况下是合理的

## 未来改进

1. **从实际传感器读取**：尝试从simulator的实际传感器对象中读取内参
2. **支持非针孔模型**：扩展支持其他相机模型（如鱼眼相机）
3. **内参验证**：添加内参合理性检查（例如，fx/fy应该接近width/height）

