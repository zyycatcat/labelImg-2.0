# -*- coding: utf-8 -*-
"""
智能吸附模块 - 自动检测构件边界并优化矩形框
使用图像处理技术（边缘检测、轮廓检测等）来找到构件的最小包围矩形
"""
import cv2
import numpy as np
import os
from libs.label_template import cn_imread
from libs.bbox_utils import calculate_bbox_overlap


def detect_object_boundary(img_path, bbox, method='adaptive'):
    """
    检测矩形框内构件的边界，返回优化的最小包围矩形
    
    参数:
        img_path (str): 图像文件路径
        bbox (list): [xmin, ymin, xmax, ymax] 原始矩形框坐标
        method (str): 检测方法
            - 'adaptive': 自适应阈值（适合大多数情况）
            - 'canny': Canny边缘检测
            - 'contour': 轮廓检测（适合有明显边界的构件）
            - 'otsu': Otsu阈值（适合对比度较好的图像）
    
    返回:
        tuple: (xmin, ymin, xmax, ymax) 优化后的矩形框，如果检测失败返回 None
    """
    try:
        # 读取图像（支持中文路径）
        img = cn_imread(img_path)
        if img is None:
            return None
        
        # 确保坐标在图像范围内
        xmin, ymin, xmax, ymax = map(int, bbox)
        h, w = img.shape[:2]
        
        # 边界检查
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(xmin + 1, min(xmax, w))
        ymax = max(ymin + 1, min(ymax, h))
        
        # 提取矩形区域（稍微扩大一点以便更好检测边界）
        margin = 1
        roi_x1 = max(0, xmin - margin)
        roi_y1 = max(0, ymin - margin)
        roi_x2 = min(w, xmax + margin)
        roi_y2 = min(h, ymax + margin)
        
        roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi.size == 0:
            return None
        
        # 转换为灰度图
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # 根据方法选择检测算法
        if method == 'adaptive':
            mask = _adaptive_threshold_detection(gray)
        elif method == 'canny':
            mask = _canny_detection(gray)
        elif method == 'contour':
            mask = _contour_detection(gray)
        elif method == 'otsu':
            mask = _otsu_threshold_detection(gray)
        elif method == 'grabcut':
            # GrabCut 需要彩色 ROI 与初始矩形（ROI 坐标系）
            init_rect = (0, 0, roi.shape[1], roi.shape[0])
            mask = _grabcut_detection(roi, init_rect)
        else:
            mask = _adaptive_threshold_detection(gray)
        
        # 从mask中找到非零区域的最小包围矩形
        optimized_bbox = _find_min_bounding_rect(mask)
        
        if optimized_bbox is None:
            return None
        
        # 将坐标转换回原图坐标系
        local_x1, local_y1, local_x2, local_y2 = optimized_bbox
        global_x1 = local_x1 + roi_x1
        global_y1 = local_y1 + roi_y1
        global_x2 = local_x2 + roi_x1
        global_y2 = local_y2 + roi_y1
        
        # 限制在原始框的一定范围内（防止扩展过大）
        max_expansion = 0.3  # 最多扩展30%
        orig_w = xmax - xmin
        orig_h = ymax - ymin
        max_w = int(orig_w * (1 + max_expansion))
        max_h = int(orig_h * (1 + max_expansion))
        
        # 计算新框相对于原始框中心的位置
        orig_cx = (xmin + xmax) / 2
        orig_cy = (ymin + ymax) / 2
        
        new_w = global_x2 - global_x1
        new_h = global_y2 - global_y1
        
        # 如果新框过大，限制其大小
        if new_w > max_w:
            new_w = max_w
        if new_h > max_h:
            new_h = max_h
        
        # 以原始框中心为基准，计算新框位置
        new_x1 = int(orig_cx - new_w / 2)
        new_y1 = int(orig_cy - new_h / 2)
        new_x2 = int(orig_cx + new_w / 2)
        new_y2 = int(orig_cy + new_h / 2)
        
        # 确保坐标在图像范围内
        new_x1 = max(0, min(new_x1, w - 1))
        new_y1 = max(0, min(new_y1, h - 1))
        new_x2 = max(new_x1 + 1, min(new_x2, w))
        new_y2 = max(new_y1 + 1, min(new_y2, h))
        
        # 如果新框和原框差异太小（小于5%），返回None表示不需要更新
        overlap_ratio = calculate_bbox_overlap([xmin, ymin, xmax, ymax], 
                                                [new_x1, new_y1, new_x2, new_y2])
        if overlap_ratio > 0.95:  # 重叠度超过95%，认为没有明显改进
            return None
        
        return [new_x1, new_y1, new_x2, new_y2]
        
    except Exception as e:
        print(f"智能吸附检测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def _adaptive_threshold_detection(gray):
    """自适应阈值检测"""
    # 使用自适应阈值来分离前景和背景
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 形态学操作去除噪声
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def _canny_detection(gray):
    """Canny边缘检测"""
    # 先进行高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 膨胀操作连接边缘
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges


def _contour_detection(gray):
    """轮廓检测方法 - 优化版，更适合主接线图构件检测"""
    # 尝试多种阈值方法，选择最佳结果
    # 方法1: Otsu阈值（适合对比度好的图像）
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 方法2: 自适应阈值（适合光照不均的图像）
    thresh2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 形态学操作去除噪声（闭运算填充小孔，开运算去除小噪声）
    kernel = np.ones((3, 3), np.uint8)
    
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    
    thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
    
    # 对两种阈值结果都查找轮廓，选择更合适的
    best_mask = None
    best_area = 0
    
    for thresh in [thresh1, thresh2]:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 找到最大的轮廓（假设是主要构件）
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            # 选择面积最大的（更可能是完整构件）
            if area > best_area:
                best_area = area
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [largest_contour], 255)
                best_mask = mask
    
    if best_mask is not None:
        return best_mask
    
    # 如果都失败，返回空mask
    return np.zeros(gray.shape, dtype=np.uint8)


def _otsu_threshold_detection(gray):
    """Otsu阈值检测"""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def _grabcut_detection(bgr_roi, init_rect):
    """使用 GrabCut 分离前景，返回二值 mask (uint8: 0/255)。

    参数:
        bgr_roi: 彩色图像 ROI (H,W,3)
        init_rect: 初始矩形 (x, y, w, h) 相对 ROI 坐标
    """
    h, w = bgr_roi.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # 构造 GrabCut 所需的 mask 和模型
    gc_mask = np.zeros((h, w), dtype=np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 适当收缩初始化矩形，避免把边界噪声当作前景
    x, y, rw, rh = init_rect
    shrink = 1
    x = max(0, x + shrink)
    y = max(0, y + shrink)
    rw = max(1, rw - 2 * shrink)
    rh = max(1, rh - 2 * shrink)
    rect = (x, y, rw, rh)

    try:
        cv2.grabCut(bgr_roi, gc_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        # 前景类别: GC_FGD(1), GC_PR_FGD(3)
        mask_bin = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype('uint8')
        # 形态学细化
        kernel = np.ones((3, 3), np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
        return mask_bin
    except Exception:
        # 失败则返回空 mask，调用方会处理为 None
        return np.zeros((h, w), dtype=np.uint8)


def _find_min_bounding_rect(mask):
    """
    从二值化mask中找到非零区域的最小包围矩形
    
    返回:
        tuple: (x1, y1, x2, y2) 或 None
    """
    # 找到非零像素的位置
    points = cv2.findNonZero(mask)
    
    if points is None or len(points) == 0:
        return None
    
    # 找到最小包围矩形
    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    
    return (x, y, x + w, y + h)


def try_multiple_methods(img_path, bbox, use_sam_if_available=True, preferred_method=None):
    """
    尝试多种检测方法，返回最佳结果
    优先尝试SAM（如果可用），否则使用传统方法
    
    参数:
        img_path: 图像路径
        bbox: 边界框 [xmin, ymin, xmax, ymax]
        use_sam_if_available: 是否优先尝试SAM
        preferred_method: 指定的方法名称
            - 'sam': 使用SAM（如果可用）
            - 'adaptive': 自适应阈值
            - 'contour': 轮廓检测
            - 'otsu': Otsu阈值
            - 'canny': Canny边缘检测
            - 'grabcut': GrabCut
    
    返回:
        tuple: (最佳bbox, 使用的method名称) 或 (None, None)
    """
    # 如果指定了单一方法，直接使用该方法
    if preferred_method:
        # SAM相关方法
        if preferred_method == 'sam' and use_sam_if_available:
            try:
                from libs.smart_snap_sam import detect_with_sam_auto
                sam_result, sam_method = detect_with_sam_auto(img_path, bbox)
                if sam_result is not None:
                    return sam_result, f"sam_{sam_method}"
            except ImportError:
                return None, None
            except Exception as e:
                print(f"SAM检测失败: {e}")
                return None, None

        # 传统CV方法
        if preferred_method in ['adaptive', 'contour', 'otsu', 'canny', 'grabcut']:
            result = detect_object_boundary(img_path, bbox, method=preferred_method)
            if result is not None:
                return result, preferred_method
            return None, None
    
    # 没有指定方法时，尝试所有方法并返回最佳结果
    # 优先尝试SAM（如果可用且启用）
    if use_sam_if_available:
        try:
            from libs.smart_snap_sam import detect_with_sam_auto
            sam_result, sam_method = detect_with_sam_auto(img_path, bbox)
            if sam_result is not None:
                return sam_result, f"sam_{sam_method}"
        except ImportError:
            # SAM未安装，使用传统方法
            pass
        except Exception as e:
            print(f"SAM检测失败，回退到传统方法: {e}")

    # 传统方法
    # 传统方法尝试次序：GrabCut 往前（对复杂前景更稳），其后为阈值/轮廓/边缘
    methods = ['grabcut', 'adaptive', 'contour', 'otsu', 'canny']
    
    best_bbox = None
    best_method = None
    best_score = -1
    
    for method in methods:
        result = detect_object_boundary(img_path, bbox, method=method)
        if result is not None:
            # 计算改进程度（新框应该比原框更紧密地包围构件）
            orig_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            new_area = (result[2] - result[0]) * (result[3] - result[1])
            
            # 如果新框面积更小但重叠度高，说明改进更好
            overlap = calculate_bbox_overlap(bbox, result)
            area_ratio = new_area / orig_area if orig_area > 0 else 1.0
            
            # 评分：重叠度高且面积适度缩小
            score = overlap * (1.0 / max(area_ratio, 0.5))  # 面积缩小越多越好，但不要太小
            
            if score > best_score:
                best_score = score
                best_bbox = result
                best_method = method
    
    return best_bbox, best_method

