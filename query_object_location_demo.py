#!/usr/bin/env python3
"""
演示如何使用增强的视觉记忆位置查询功能

示例用法：
    python query_object_location_demo.py "刚刚找到的那个杯子在哪个桌子上？"
    python query_object_location_demo.py "水杯在哪里？" --object_label "水杯"
"""

import argparse
import sys
import json
import rag_robot_framework as rag

def main():
    parser = argparse.ArgumentParser(description="查询物体位置信息")
    parser.add_argument("query", help="查询文本，例如：'刚刚找到的那个杯子在哪个桌子上？' 或 '水杯在哪里？'")
    parser.add_argument("--object_label", type=str, default=None, help="可选的物体标签，用于精确匹配")
    parser.add_argument("--topk", type=int, default=5, help="返回前k个结果")
    args = parser.parse_args()
    
    # 初始化数据库
    db = rag.init_chroma_db()
    if db is None:
        print("错误：Chroma 向量库初始化失败")
        sys.exit(1)
    rag.chroma_client, rag.chroma_collection = db
    
    # 查询位置
    result = rag.query_object_location(
        query_text=args.query,
        object_label=args.object_label,
        topk=args.topk
    )
    
    # 输出结果
    print("\n" + "="*60)
    print("位置查询结果")
    print("="*60)
    
    if result.get("error"):
        print(f"错误: {result['error']}")
        sys.exit(1)
    
    if result.get("found"):
        print(f"\n✓ {result['answer']}\n")
        print("详细信息：")
        best_match = result.get("best_match", {})
        print(f"  物体标签: {best_match.get('item_label', 'N/A')}")
        print(f"  描述: {best_match.get('description', 'N/A')}")
        print(f"  置信度: {best_match.get('confidence', 0.0):.2f}")
        
        if best_match.get("world_position"):
            pos = best_match["world_position"]
            print(f"  世界坐标: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        if best_match.get("location_semantic"):
            print(f"  语义位置: {best_match['location_semantic']}")
        
        if best_match.get("object_crop_path"):
            print(f"  物体截图: {best_match['object_crop_path']}")
        
        if best_match.get("scene_image_path"):
            print(f"  场景图像: {best_match['scene_image_path']}")
        
        if best_match.get("timestamp"):
            print(f"  时间戳: {best_match['timestamp']}")
        
        print(f"\n找到 {result.get('count', 0)} 个匹配结果")
        
        # 显示所有匹配结果
        all_matches = result.get("all_matches", [])
        if len(all_matches) > 1:
            print("\n所有匹配结果：")
            for i, match in enumerate(all_matches[1:], 1):
                print(f"  {i+1}. {match.get('item_label', 'N/A')} - 置信度: {match.get('confidence', 0.0):.2f}")
    else:
        print(f"\n✗ {result['answer']}")
    
    print("\n" + "="*60)
    
    # 可选：输出JSON格式
    if args.topk > 1:
        print("\n完整JSON结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

