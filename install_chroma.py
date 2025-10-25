#!/usr/bin/env python3
"""
ChromaDB安装脚本
用于安装和配置Chroma向量数据库
"""

import subprocess
import sys
import os

def install_chroma():
    """安装ChromaDB"""
    try:
        print("正在安装ChromaDB...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chromadb"])
        print("✅ ChromaDB安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ChromaDB安装失败: {e}")
        return False

def test_chroma():
    """测试ChromaDB是否正常工作"""
    try:
        import chromadb
        from chromadb.config import Settings
        
        # 创建测试客户端
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        # 创建测试集合
        collection = client.create_collection("test_collection")
        
        # 添加测试数据
        collection.add(
            documents=["This is a test document"],
            embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]],
            metadatas=[{"test": True}],
            ids=["test_id"]
        )
        
        # 查询测试数据
        results = collection.query(
            query_texts=["test document"],
            n_results=1
        )
        
        if results['ids'] and results['ids'][0]:
            print("✅ ChromaDB测试成功！")
            return True
        else:
            print("❌ ChromaDB测试失败：查询无结果")
            return False
            
    except Exception as e:
        print(f"❌ ChromaDB测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("ChromaDB 安装和测试脚本")
    print("=" * 50)
    
    # 检查是否已安装
    try:
        import chromadb
        print("✅ ChromaDB已安装")
    except ImportError:
        print("❌ ChromaDB未安装，开始安装...")
        if not install_chroma():
            print("安装失败，请手动运行: pip install chromadb")
            return
    
    # 测试ChromaDB
    print("\n正在测试ChromaDB...")
    if test_chroma():
        print("\n🎉 ChromaDB安装和测试完成！")
        print("现在可以运行 rag_robot_framework.py 了")
    else:
        print("\n❌ ChromaDB测试失败，请检查安装")

if __name__ == "__main__":
    main()
