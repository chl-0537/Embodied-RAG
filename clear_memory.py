#!/usr/bin/env python3
"""
清除机器人记忆数据库的脚本
支持两种模式：
1. 清空集合中的所有记忆（保留集合结构）
2. 完全删除集合或数据库目录
"""

import os
import sys
import argparse
import shutil

# 导入 RAG 框架组件
import rag_robot_framework as rag


def clear_collection_only():
    """仅清空集合中的所有记忆条目，保留集合结构"""
    try:
        # 初始化数据库
        db_result = rag.init_chroma_db()
        if db_result is None:
            rag.log_error("无法初始化数据库")
            return False
        
        client, collection = db_result
        
        # 获取当前记忆数量
        count_before = collection.count()
        rag.log_info(f"清空前记忆数量: {count_before}")
        
        if count_before == 0:
            rag.log_info("记忆数据库已为空，无需清除")
            return True
        
        # 获取所有记忆ID
        results = collection.get()
        ids = results.get("ids", [])
        
        if ids:
            # 删除所有记忆条目
            collection.delete(ids=ids)
            rag.log_info(f"已删除 {len(ids)} 条记忆")
        
        # 验证清空结果
        count_after = collection.count()
        if count_after == 0:
            rag.log_info("记忆数据库已成功清空")
            return True
        else:
            rag.log_warning(f"清空后仍有 {count_after} 条记忆，可能未完全清除")
            return False
            
    except Exception as e:
        rag.log_error("清空记忆集合失败", e)
        return False


def delete_collection():
    """删除整个集合"""
    try:
        db_result = rag.init_chroma_db()
        if db_result is None:
            rag.log_error("无法初始化数据库")
            return False
        
        client, collection = db_result
        
        # 获取当前记忆数量
        count_before = collection.count()
        rag.log_info(f"删除前记忆数量: {count_before}")
        
        # 删除集合
        client.delete_collection(name=rag.COLLECTION_NAME)
        rag.log_info(f"已删除集合: {rag.COLLECTION_NAME}")
        return True
        
    except Exception as e:
        rag.log_error("删除集合失败", e)
        return False


def delete_database_directory():
    """完全删除数据库目录"""
    try:
        db_dir = rag.VECTOR_DB_DIR
        
        if not os.path.exists(db_dir):
            rag.log_info(f"数据库目录不存在: {db_dir}")
            return True
        
        # 获取目录大小（用于日志）
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(db_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        rag.log_info(f"数据库目录大小: {total_size / 1024 / 1024:.2f} MB")
        
        # 删除整个目录
        shutil.rmtree(db_dir)
        rag.log_info(f"已删除数据库目录: {db_dir}")
        return True
        
    except Exception as e:
        rag.log_error("删除数据库目录失败", e)
        return False


def get_memory_stats():
    """获取记忆统计信息"""
    try:
        db_result = rag.init_chroma_db()
        if db_result is None:
            return {"error": "无法初始化数据库"}
        
        client, collection = db_result
        
        count = collection.count()
        stats = rag.get_memory_stats()
        
        return {
            "total_memories": count,
            "database_path": rag.VECTOR_DB_DIR,
            "collection_name": rag.COLLECTION_NAME,
            **stats
        }
        
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="清除机器人记忆数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 查看当前记忆统计
  python clear_memory.py --stats
  
  # 仅清空所有记忆（保留集合结构）
  python clear_memory.py --clear
  
  # 删除整个集合（集合结构也会删除）
  python clear_memory.py --delete-collection
  
  # 完全删除数据库目录（最彻底）
  python clear_memory.py --delete-all
        """
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="显示记忆统计信息"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="清空集合中的所有记忆（保留集合结构）"
    )
    
    parser.add_argument(
        "--delete-collection",
        action="store_true",
        help="删除整个记忆集合"
    )
    
    parser.add_argument(
        "--delete-all",
        action="store_true",
        help="完全删除数据库目录（最彻底的方式）"
    )
    
    args = parser.parse_args()
    
    # 如果没有指定任何操作，显示统计信息
    if not any([args.stats, args.clear, args.delete_collection, args.delete_all]):
        args.stats = True
    
    # 显示统计信息
    if args.stats:
        rag.log_info("=" * 60)
        rag.log_info("记忆数据库统计信息")
        rag.log_info("=" * 60)
        stats = get_memory_stats()
        if "error" in stats:
            rag.log_error(f"获取统计信息失败: {stats['error']}")
        else:
            for key, value in stats.items():
                rag.log_info(f"{key}: {value}")
        rag.log_info("=" * 60)
    
    # 执行清除操作
    if args.clear:
        rag.log_info("开始清空记忆集合...")
        if clear_collection_only():
            rag.log_info("✓ 清空操作完成")
        else:
            rag.log_error("✗ 清空操作失败")
            sys.exit(1)
    
    if args.delete_collection:
        rag.log_info("开始删除记忆集合...")
        confirm = input("确认删除整个集合吗？(yes/no): ")
        if confirm.lower() == "yes":
            if delete_collection():
                rag.log_info("✓ 删除集合操作完成")
            else:
                rag.log_error("✗ 删除集合操作失败")
                sys.exit(1)
        else:
            rag.log_info("已取消删除操作")
    
    if args.delete_all:
        rag.log_info("开始删除整个数据库目录...")
        confirm = input("确认完全删除数据库目录吗？这将删除所有数据！(yes/no): ")
        if confirm.lower() == "yes":
            if delete_database_directory():
                rag.log_info("✓ 删除数据库目录操作完成")
            else:
                rag.log_error("✗ 删除数据库目录操作失败")
                sys.exit(1)
        else:
            rag.log_info("已取消删除操作")


if __name__ == "__main__":
    main()

