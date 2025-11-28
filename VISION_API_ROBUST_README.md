# 视觉API增强包装器实现说明

## 概述

本次实现创建了一个鲁棒的视觉API包装器，包含错误分类、指数退避、结构化输出和安全JSON解析，大幅提高了视觉API调用的可靠性和容错能力。

## 主要功能

### 1. 错误分类系统 (`_classify_vision_error`)

自动分类视觉API错误：

| 错误类型 | 说明 | 重试策略 |
|---------|------|---------|
| `network` | 网络连接错误 | 重试 |
| `timeout` | 请求超时 | 重试 |
| `5xx` | 服务器错误（500-599） | 重试（带退避） |
| `4xx` | 客户端错误（400-499） | 不重试 |
| `malformed_json` | JSON格式错误 | 重试1次 |
| `empty_response` | 空响应 | 重试 |
| `unknown` | 未知错误 | 不重试 |

### 2. 指数退避策略

**退避时间序列**：`[0.5, 1.0, 2.0, 4.0]` 秒

- **第1次重试**：等待 0.5 秒
- **第2次重试**：等待 1.0 秒
- **第3次重试**：等待 2.0 秒
- **第4次重试**：等待 4.0 秒

**公式**：`sleep_time = backoff_base * (2 ^ attempt_index)`

### 3. 安全JSON解析 (`_safe_parse_json`)

多层次的JSON解析策略：

1. **直接解析**：尝试直接解析原始文本
2. **提取JSON块**：使用正则表达式提取JSON部分
   - 支持嵌套JSON
   - 移除markdown代码块标记（```json ... ```）
3. **修复常见错误**：
   - 移除注释（// 和 /* */）
   - 修复单引号（' → "）
   - 修复尾随逗号
   - 提取并解析修复后的JSON

### 4. 增强的视觉API包装器 (`_call_chatglm_vision_robust`)

**功能特性：**

- **自动重试**：根据错误类型智能重试
- **指数退避**：避免对服务器造成压力
- **结构化输出**：始终返回统一的结构
- **详细日志**：记录所有错误和重试信息

**返回结构：**
```python
{
    "success": bool,           # 是否成功
    "error": str or None,      # 错误信息
    "error_type": str or None, # 错误类型
    "raw": str or None,        # 原始响应文本
    "boxes": List[Dict],       # 边界框列表
    "labels": List[str],       # 标签列表
    "confidence": float,      # 平均置信度
    "data": Dict or None,     # 解析后的数据
    "retry_count": int         # 重试次数
}
```

### 5. Fallback机制

#### 5.1 JSON修复Fallback (`_fallback_vision_json_rectification`)

当视觉API返回的JSON无法解析时：

1. **提取原始文本**：从失败的响应中提取文本
2. **LLM修复**：使用LLM修复JSON格式
3. **验证结构**：验证修复后的JSON是否符合要求
4. **返回结果**：返回修复后的数据或错误

**Fallback提示词特点：**
- 明确要求"严格输出JSON格式"
- 要求移除markdown标记
- 要求修复常见JSON错误

#### 5.2 集成到现有函数

- **`call_chatglm_multi_object_detection`**：自动使用增强包装器和fallback
- **`call_chatglm_vision_json`**：自动使用增强包装器和JSON修复

## 实现细节

### 错误处理流程

```
API调用
  ↓
成功？
  ↓ 是
解析JSON
  ↓
成功？
  ↓ 是
验证结构
  ↓
返回结果
  ↓ 否
分类错误
  ↓
可重试？
  ↓ 是
执行退避
  ↓
重试
  ↓ 否
尝试Fallback
  ↓
返回结果/错误
```

### 重试决策逻辑

```python
if error_type == "timeout" or error_type == "network":
    # 重试所有尝试
    retry = True
elif error_type == "5xx":
    # 重试所有尝试（带退避）
    retry = True
elif error_type == "4xx":
    # 不重试（客户端错误）
    retry = False
elif error_type == "malformed_json":
    # 只重试1次
    retry = (attempt == 0)
else:
    # 不重试
    retry = False
```

### JSON解析策略

1. **直接解析**：最快，适用于标准JSON
2. **正则提取**：处理包含额外文本的响应
3. **错误修复**：处理格式错误的JSON

## 使用示例

### 基本使用

```python
# 自动使用增强包装器
result = call_chatglm_multi_object_detection(image_url, target_objects=["杯子", "桌子"])

if result.is_success():
    objects = result.data.get("objects", [])
    print(f"检测到 {len(objects)} 个对象")
else:
    print(f"检测失败: {result.error_msg}")
```

### 检查重试信息

```python
result = call_chatglm_multi_object_detection(image_url, target_objects=["杯子"])

if result.metadata:
    retry_count = result.metadata.get("retry_count", 1)
    print(f"重试次数: {retry_count}")
    
    if result.metadata.get("fallback_used"):
        print("使用了fallback机制")
```

## 日志输出

### 成功情况

```
[INFO] 视觉API调用成功 {"attempt": 1, "objects_count": 5, "boxes_count": 5}
[INFO] Multi-object detection successful {"objects_count": 5, "target_objects": ["杯子"], "retry_count": 1}
```

### 重试情况

```
[INFO] 视觉API重试 1/3，等待 0.50 秒...
[WARN] vision_error {"error_type": "timeout", "attempt": 2, "max_retries": 4, "error": "Request timeout: ..."}
```

### Fallback情况

```
[WARN] Vision API call failed, attempting fallback {"error_type": "malformed_json", "error": "JSON parsing failed: ...", "retry_count": 2}
[INFO] JSON rectification successful via LLM fallback {"objects_count": 3}
```

## 错误类型详解

### 1. Network/Timeout错误

**特点**：临时性错误，通常可以重试成功

**处理**：
- 自动重试所有尝试
- 使用指数退避
- 记录详细日志

### 2. 5xx服务器错误

**特点**：服务器端问题，可能需要等待

**处理**：
- 自动重试所有尝试
- 使用指数退避（给服务器恢复时间）
- 记录状态码和错误信息

### 3. 4xx客户端错误

**特点**：请求格式错误或认证失败，重试无意义

**处理**：
- 不重试
- 立即返回错误
- 记录详细错误信息

### 4. Malformed JSON错误

**特点**：响应格式问题，可能需要修复

**处理**：
- 重试1次（可能只是临时格式问题）
- 如果仍然失败，尝试fallback修复
- 记录原始文本用于调试

## 性能考虑

### 重试开销

- **最大重试次数**：4次
- **总等待时间**：最多 0.5 + 1.0 + 2.0 + 4.0 = 7.5秒
- **总请求时间**：最多 60秒 × 4 = 240秒（如果每次都超时）

### 优化建议

1. **减少重试次数**：对于非关键调用，可以减少到2-3次
2. **缩短超时时间**：对于快速响应场景，可以减少到30秒
3. **并行重试**：对于独立请求，可以考虑并行处理

## 参数配置

### `max_retries`

- **默认值**：4
- **调整建议**：
  - 关键调用：4-5次
  - 一般调用：2-3次
  - 快速失败：1次

### `backoff_base`

- **默认值**：0.5秒
- **调整建议**：
  - 快速恢复场景：0.25秒
  - 标准场景：0.5秒
  - 慢速恢复场景：1.0秒

## 测试建议

1. **网络错误测试**：模拟网络中断
2. **超时测试**：模拟慢速响应
3. **JSON格式测试**：测试各种格式错误的JSON
4. **Fallback测试**：验证fallback机制是否正常工作
5. **重试测试**：验证重试逻辑和退避时间

## 未来改进

1. **自适应退避**：根据错误类型动态调整退避时间
2. **熔断器模式**：在连续失败时暂时停止请求
3. **请求去重**：避免重复请求相同内容
4. **缓存机制**：缓存成功的响应
5. **监控和告警**：集成监控系统，及时发现问题

## 技术细节

### JSON解析正则表达式

```python
# 嵌套JSON匹配
r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'

# 简单JSON匹配
r'\{.*?\}'
```

### 错误分类逻辑

```python
if isinstance(error, requests.Timeout):
    return "timeout"
elif isinstance(error, requests.ConnectionError):
    return "network"
elif response.status_code >= 500:
    return "5xx"
elif response.status_code >= 400:
    return "4xx"
elif isinstance(error, json.JSONDecodeError):
    return "malformed_json"
```

### 结构化输出提取

```python
objects = parsed_result.get("objects", [])
for obj in objects:
    boxes.append(obj.get("bbox", []))
    labels.append(obj.get("label", ""))
    confidences.append(obj.get("confidence", 0.0))
avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
```

