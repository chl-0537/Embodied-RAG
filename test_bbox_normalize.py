#!/usr/bin/env python3
"""
测试 normalize_and_validate_bbox 函数

运行方式：
    python test_bbox_normalize.py
"""

import sys
import numpy as np

def normalize_and_validate_bbox(bbox, img_w, img_h):
    """
    统一的边界框归一化和验证工具函数
    
    功能：
    1. 自动检测bbox格式（归一化[0,1]或像素坐标）
    2. 转换为像素坐标 (x1, y1, x2, y2)
    3. 裁剪到图像边界
    4. 确保 x1 < x2, y1 < y2
    5. 计算边界框面积占比
    
    Args:
        bbox: 边界框 [x1, y1, x2, y2]，可能是归一化坐标[0,1]或像素坐标
        img_w: 图像宽度（像素）
        img_h: 图像高度（像素）
    
    Returns:
        (x1, y1, x2, y2, area_ratio) 或 None（如果bbox无效）
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (ValueError, TypeError):
        return None
    
    # 检测是否为归一化坐标
    is_normalized = (
        (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1) or
        (max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5)
    )
    
    if is_normalized:
        # 归一化坐标 -> 像素坐标
        x1, x2 = x1 * img_w, x2 * img_w
        y1, y2 = y1 * img_h, y2 * img_h
    
    # 确保顺序正确：x1 < x2, y1 < y2
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    
    # 裁剪到图像边界
    x1 = max(0, min(img_w - 1, int(round(x1))))
    y1 = max(0, min(img_h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(img_w, int(round(x2))))
    y2 = max(y1 + 1, min(img_h, int(round(y2))))
    
    # 计算面积占比
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    area_ratio = bbox_area / img_area if img_area > 0 else 0.0
    
    # 验证：确保bbox有效（面积>0）
    if bbox_area <= 0:
        return None
    
    return (x1, y1, x2, y2, area_ratio)


def test_normalize_and_validate_bbox():
    """测试 normalize_and_validate_bbox 函数"""
    print("=" * 60)
    print("测试 normalize_and_validate_bbox 函数")
    print("=" * 60)
    
    img_w, img_h = 640, 480
    tests_passed = 0
    tests_failed = 0
    
    # 测试1: 归一化坐标 [0, 1]
    print("\n测试1: 归一化坐标 [0.1, 0.2, 0.5, 0.6]")
    result = normalize_and_validate_bbox([0.1, 0.2, 0.5, 0.6], img_w, img_h)
    if result:
        x1, y1, x2, y2, area_ratio = result
        expected_x1, expected_y1 = int(0.1 * img_w), int(0.2 * img_h)
        expected_x2, expected_y2 = int(0.5 * img_w), int(0.6 * img_h)
        if x1 == expected_x1 and y1 == expected_y1 and x2 == expected_x2 and y2 == expected_y2:
            print(f"  ✓ 通过: ({x1}, {y1}, {x2}, {y2}), area_ratio={area_ratio:.4f}")
            tests_passed += 1
        else:
            print(f"  ✗ 失败: 期望 ({expected_x1}, {expected_y1}, {expected_x2}, {expected_y2}), 得到 ({x1}, {y1}, {x2}, {y2})")
            tests_failed += 1
    else:
        print("  ✗ 失败: 返回 None")
        tests_failed += 1
    
    # 测试2: 像素坐标
    print("\n测试2: 像素坐标 [100, 200, 300, 400]")
    result = normalize_and_validate_bbox([100, 200, 300, 400], img_w, img_h)
    if result:
        x1, y1, x2, y2, area_ratio = result
        if x1 == 100 and y1 == 200 and x2 == 300 and y2 == 400:
            print(f"  ✓ 通过: ({x1}, {y1}, {x2}, {y2}), area_ratio={area_ratio:.4f}")
            tests_passed += 1
        else:
            print(f"  ✗ 失败: 期望 (100, 200, 300, 400), 得到 ({x1}, {y1}, {x2}, {y2})")
            tests_failed += 1
    else:
        print("  ✗ 失败: 返回 None")
        tests_failed += 1
    
    # 测试3: 顺序错误的坐标（应该自动纠正）
    print("\n测试3: 顺序错误的坐标 [0.5, 0.6, 0.1, 0.2]")
    result = normalize_and_validate_bbox([0.5, 0.6, 0.1, 0.2], img_w, img_h)
    if result:
        x1, y1, x2, y2, area_ratio = result
        if x1 < x2 and y1 < y2:
            print(f"  ✓ 通过: 已自动纠正顺序 ({x1}, {y1}, {x2}, {y2}), area_ratio={area_ratio:.4f}")
            tests_passed += 1
        else:
            print(f"  ✗ 失败: 顺序未纠正 ({x1}, {y1}, {x2}, {y2})")
            tests_failed += 1
    else:
        print("  ✗ 失败: 返回 None")
        tests_failed += 1
    
    # 测试4: 超出边界的坐标（应该裁剪）
    print("\n测试4: 超出边界的坐标 [0.9, 0.9, 1.2, 1.2]")
    result = normalize_and_validate_bbox([0.9, 0.9, 1.2, 1.2], img_w, img_h)
    if result:
        x1, y1, x2, y2, area_ratio = result
        if x2 <= img_w and y2 <= img_h:
            print(f"  ✓ 通过: 已裁剪到边界 ({x1}, {y1}, {x2}, {y2}), area_ratio={area_ratio:.4f}")
            tests_passed += 1
        else:
            print(f"  ✗ 失败: 未正确裁剪 ({x1}, {y1}, {x2}, {y2})")
            tests_failed += 1
    else:
        print("  ✗ 失败: 返回 None")
        tests_failed += 1
    
    # 测试5: 无效的bbox（长度不对）
    print("\n测试5: 无效的bbox [100, 200]")
    result = normalize_and_validate_bbox([100, 200], img_w, img_h)
    if result is None:
        print("  ✓ 通过: 正确返回 None")
        tests_passed += 1
    else:
        print(f"  ✗ 失败: 应该返回 None，但得到 {result}")
        tests_failed += 1
    
    # 测试6: 无效的bbox（面积为0）
    print("\n测试6: 面积为0的bbox [100, 200, 100, 200]")
    result = normalize_and_validate_bbox([100, 200, 100, 200], img_w, img_h)
    if result is None:
        print("  ✓ 通过: 正确返回 None（面积为0）")
        tests_passed += 1
    else:
        print(f"  ✗ 失败: 应该返回 None，但得到 {result}")
        tests_failed += 1
    
    # 测试7: 负坐标（应该裁剪到0）
    print("\n测试7: 负坐标 [-10, -20, 100, 200]")
    result = normalize_and_validate_bbox([-10, -20, 100, 200], img_w, img_h)
    if result:
        x1, y1, x2, y2, area_ratio = result
        if x1 >= 0 and y1 >= 0:
            print(f"  ✓ 通过: 已裁剪到0 ({x1}, {y1}, {x2}, {y2}), area_ratio={area_ratio:.4f}")
            tests_passed += 1
        else:
            print(f"  ✗ 失败: 未正确裁剪 ({x1}, {y1}, {x2}, {y2})")
            tests_failed += 1
    else:
        print("  ✗ 失败: 返回 None")
        tests_failed += 1
    
    # 测试8: 面积占比计算
    print("\n测试8: 面积占比计算 [0.0, 0.0, 0.5, 0.5]")
    result = normalize_and_validate_bbox([0.0, 0.0, 0.5, 0.5], img_w, img_h)
    if result:
        x1, y1, x2, y2, area_ratio = result
        expected_ratio = 0.25  # 0.5 * 0.5 = 0.25
        if abs(area_ratio - expected_ratio) < 0.01:
            print(f"  ✓ 通过: area_ratio={area_ratio:.4f} (期望 {expected_ratio:.4f})")
            tests_passed += 1
        else:
            print(f"  ✗ 失败: area_ratio={area_ratio:.4f}, 期望 {expected_ratio:.4f}")
            tests_failed += 1
    else:
        print("  ✗ 失败: 返回 None")
        tests_failed += 1
    
    # 总结
    print("\n" + "=" * 60)
    print(f"测试结果: {tests_passed} 通过, {tests_failed} 失败")
    print("=" * 60)
    
    return tests_failed == 0


if __name__ == "__main__":
    success = test_normalize_and_validate_bbox()
    sys.exit(0 if success else 1)

