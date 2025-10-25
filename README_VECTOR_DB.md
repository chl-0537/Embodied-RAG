# RAG Robot Framework with Vector Database

## 概述

本项目已升级为使用Chroma向量数据库，提供更强大的记忆存储和检索能力。

## 主要改进

### 1. 向量数据库集成
- **ChromaDB**: 轻量级、高性能的向量数据库
- **持久化存储**: 记忆数据自动保存到磁盘
- **高维索引**: 支持1536维向量的高效检索
- **元数据过滤**: 支持按位置、时间等条件过滤

### 2. 智能检索策略
- **组合查询**: `"蓝色水杯 in 书房"` 形式的增强查询
- **位置过滤**: 仅检索特定区域的物体
- **语义理解**: 更好的对象-位置关联

### 3. 新增功能
- **数据库统计**: 查看记忆总数和存储信息
- **位置搜索**: 按位置查找所有物体
- **元数据管理**: 丰富的物体属性存储

## 安装要求

### 1. 安装ChromaDB
```bash
pip install chromadb
```

或者使用提供的安装脚本：
```bash
python install_chroma.py
```

### 2. 验证安装
```python
import chromadb
print("ChromaDB安装成功！")
```

## 使用方法

### 1. 基本使用
```python
python rag_robot_framework.py
```

### 2. 程序输出
程序会显示：
- 向量数据库初始化状态
- 记忆数据统计信息
- 位置搜索演示结果
- 增强检索过程
- 详细的日志记录

### 3. 日志文件
- 位置: `logs/rag_robot_YYYYMMDD_HHMMSS.log`
- 内容: 完整的执行过程和调试信息
- 格式: 时间戳 + 日志级别 + 详细信息

## 核心功能

### 1. 记忆存储
```python
# 添加记忆条目
memory_id = add_memory_entry_simple(
    text_desc="蓝色水杯在书桌上",
    meta={
        "item_label": "watercup",
        "spatial": {"location_semantic": "study_desk"},
        "temporal": {"timestamp_utc": "2024-01-15T10:00:00+00:00"},
        "confidence": 0.9
    }
)
```

### 2. 智能检索
```python
# 基于意图检索
intent = {
    "object": "蓝色水杯",
    "location": "书房"
}
candidates = retrieve_candidates_by_intent(intent)
```

### 3. 位置搜索
```python
# 查找特定位置的所有物体
objects = search_by_location("study_desk")
```

### 4. 数据库统计
```python
# 获取数据库统计信息
stats = get_memory_stats()
print(f"总记忆数: {stats['total_memories']}")
```

## 文件结构

```
├── rag_robot_framework.py    # 主程序
├── install_chroma.py         # ChromaDB安装脚本
├── logs/                     # 日志文件目录
│   └── rag_robot_*.log
├── vector_db/                # 向量数据库存储
│   └── robot_memory/         # 记忆集合
└── picture/                  # 图片资源
    └── study_room.png
```

## 性能优势

### 1. 检索精度提升
- **组合查询**: 对象+位置组合查询提高准确性
- **元数据过滤**: 减少无关结果
- **语义理解**: 更好的上下文匹配

### 2. 存储效率
- **持久化**: 数据不会因程序重启丢失
- **压缩存储**: 高效的向量压缩算法
- **增量更新**: 支持增量添加记忆

### 3. 扩展性
- **大规模数据**: 支持百万级记忆存储
- **并发访问**: 支持多进程并发查询
- **分布式**: 可扩展到分布式部署

## 故障排除

### 1. ChromaDB安装失败
```bash
# 更新pip
pip install --upgrade pip

# 安装ChromaDB
pip install chromadb

# 如果仍有问题，尝试
pip install chromadb --no-cache-dir
```

### 2. 数据库初始化失败
- 检查 `vector_db/` 目录权限
- 确保有足够的磁盘空间
- 查看日志文件中的详细错误信息

### 3. 检索结果为空
- 确认记忆数据已正确添加
- 检查查询文本是否匹配
- 验证元数据过滤条件

## 开发说明

### 1. 添加新的检索策略
```python
def custom_retrieve(query_params):
    # 自定义检索逻辑
    return chroma_collection.query(**query_params)
```

### 2. 扩展元数据字段
```python
metadata = {
    "item_label": "watercup",
    "location_semantic": "study_desk",
    "color": "blue",           # 新增颜色字段
    "size": "medium",          # 新增尺寸字段
    "timestamp_utc": "...",
    "confidence": 0.9
}
```

### 3. 自定义重排序算法
```python
def custom_rerank(candidates, weights):
    # 自定义重排序逻辑
    return sorted_candidates
```

## 更新日志

- **v2.0**: 集成ChromaDB向量数据库
- **v2.1**: 添加智能检索策略
- **v2.2**: 支持元数据过滤
- **v2.3**: 完善日志记录系统
