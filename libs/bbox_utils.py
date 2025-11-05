# -*- coding: utf-8 -*-
"""
边界框相关的工具函数
"""


def calculate_bbox_overlap(bbox1, bbox2):
    """
    计算两个矩形框的重叠度（IoU - Intersection over Union）
    
    参数:
        bbox1: [x_min, y_min, x_max, y_max] 第一个边界框
        bbox2: [x_min, y_min, x_max, y_max] 第二个边界框
    
    返回:
        float: IoU值，范围 [0.0, 1.0]
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # 计算交集
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    
    # 计算并集
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

