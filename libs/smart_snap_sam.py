# -*- coding: utf-8 -*-
"""
基于 MobileSAM 服务器的智能吸附模块
"""
import cv2
import numpy as np
from libs.label_template import cn_imread


def _mask_to_bbox(mask, offset=[0, 0]):
    """将mask转换为最小包围矩形"""
    try:
        if isinstance(mask, np.ndarray):
            if mask.dtype != bool:
                mask = mask > 0.5
        else:
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            mask = mask > 0.5
        
        y_coords, x_coords = np.where(mask)
        
        if len(x_coords) == 0 or len(y_coords) == 0:
            return None
        
        x_min = int(x_coords.min()) + offset[0]
        y_min = int(y_coords.min()) + offset[1]
        x_max = int(x_coords.max()) + offset[0]
        y_max = int(y_coords.max()) + offset[1]
        
        return [x_min, y_min, x_max, y_max]
        
    except Exception as e:
        print(f"mask转bbox失败: {e}")
        return None


def detect_with_sam_auto(img_path, bbox, device='auto'):
    """使用 MobileSAM 服务器进行检测"""
    try:
        result = detect_object_boundary_sam_server(img_path, bbox)
        if result is not None:
            return result, 'mobilesam_server'
    except Exception as e:
        print(f"MobileSAM 服务器调用失败: {e}")
    
    return None, None


def detect_object_boundary_sam_server(img_path, bbox):
    """调用 MobileSAM 服务器API进行智能吸附
    
    参数:
        img_path: 图像文件路径
        bbox: [x1, y1, x2, y2] 原始矩形框坐标
    
    返回:
        [x1, y1, x2, y2] 优化后的矩形框，失败返回 None
    """
    try:
        import requests
    except ImportError:
        print("缺少 requests 库，请安装: pip install requests")
        return None
    
    import os
    import base64
    import time
    
    api_url = os.environ.get('MOBILE_SAM_API_URL', 'http://10.226.2.1:31265/segment/box')
    if not api_url:
        return None
    
    t_start = time.time()
    
    try:
        img = cn_imread(img_path)
        if img is None:
            return None
        
        H, W = img.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        
        # 边界检查
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(x1 + 1, min(x2, W))
        y2 = max(y1 + 1, min(y2, H))
        
        # 裁剪ROI区域，减少传输数据量
        roi_margin_ratio = float(os.environ.get('SAM_ROI_MARGIN', '0.25'))
        bw = x2 - x1
        bh = y2 - y1
        margin_x = max(15, int(bw * roi_margin_ratio))
        margin_y = max(15, int(bh * roi_margin_ratio))
        
        roi_x1 = max(0, x1 - margin_x)
        roi_y1 = max(0, y1 - margin_y)
        roi_x2 = min(W, x2 + margin_x)
        roi_y2 = min(H, y2 + margin_y)
        
        roi_img = img[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        roi_h, roi_w = roi_img.shape[:2]
        
        # 将bbox坐标转换到ROI坐标系
        roi_box_x1 = x1 - roi_x1
        roi_box_y1 = y1 - roi_y1
        roi_box_x2 = x2 - roi_x1
        roi_box_y2 = y2 - roi_y1
        
        # 编码图像
        jpeg_quality = int(os.environ.get('SAM_JPEG_QUALITY', '85'))
        ok, buf = cv2.imencode('.jpg', roi_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            return None
        
        image_bytes = buf.tobytes() if hasattr(buf, 'tobytes') else bytes(buf)
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # 准备请求
        payload = {
            'image': image_base64,
            'box': [roi_box_x1, roi_box_y1, roi_box_x2, roi_box_y2],
            'input_size': 1024
        }
        
        # 发送请求
        t_request_start = time.time()
        resp = requests.post(api_url, json=payload, timeout=30)
        t_request_end = time.time()
        
        if resp.status_code != 200:
            print(f"服务器返回错误: {resp.status_code}")
            try:
                error_data = resp.json()
                print(f"错误详情: {error_data}")
            except:
                print(f"响应文本: {resp.text[:200]}")
            return None
        
        data = resp.json()
        network_time = t_request_end - t_request_start
        
        # 获取mask和score
        mask_data = data.get('mask')
        score = data.get('score', 0.0)
        image_size = data.get('image_size', {})
        
        if mask_data is None:
            return None
        
        # 将mask转换为numpy数组
        mask_np = np.array(mask_data, dtype=bool)
        
        if mask_np.size == 0:
            return None
        
        # 从mask计算最小包围矩形
        y_coords, x_coords = np.where(mask_np)
        
        if len(x_coords) == 0 or len(y_coords) == 0:
            return None
        
        mask_x1 = int(x_coords.min())
        mask_y1 = int(y_coords.min())
        mask_x2 = int(x_coords.max())
        mask_y2 = int(y_coords.max())
        
        # 坐标转换：mask -> ROI -> 原图
        mask_h, mask_w = mask_np.shape
        server_roi_w = image_size.get('width', roi_w)
        server_roi_h = image_size.get('height', roi_h)
        
        # 计算缩放比例
        scale_mask_to_roi_x = server_roi_w / mask_w if mask_w > 0 else 1.0
        scale_mask_to_roi_y = server_roi_h / mask_h if mask_h > 0 else 1.0
        
        # 转换mask坐标到ROI坐标系
        roi_result_x1 = int(mask_x1 * scale_mask_to_roi_x)
        roi_result_y1 = int(mask_y1 * scale_mask_to_roi_y)
        roi_result_x2 = int(mask_x2 * scale_mask_to_roi_x)
        roi_result_y2 = int(mask_y2 * scale_mask_to_roi_y)
        
        # ROI坐标系 -> 原图坐标系
        ex1 = roi_result_x1 + roi_x1
        ey1 = roi_result_y1 + roi_y1
        ex2 = roi_result_x2 + roi_x1
        ey2 = roi_result_y2 + roi_y1
        
        # 确保结果框在图像范围内
        ex1 = max(0, min(ex1, W - 1))
        ey1 = max(0, min(ey1, H - 1))
        ex2 = max(ex1 + 1, min(ex2, W))
        ey2 = max(ey1 + 1, min(ey2, H))
        
        # 建议框必须在原始框范围内
        ex1 = max(x1, ex1)
        ey1 = max(y1, ey1)
        ex2 = min(x2, ex2)
        ey2 = min(y2, ey2)
        
        # 检查有效性
        if ex2 <= ex1 or ey2 <= ey1:
            return None
        
        # 计算改进程度
        orig_w = x2 - x1
        orig_h = y2 - y1
        final_w = ex2 - ex1
        final_h = ey2 - ey1
        orig_area = orig_w * orig_h
        new_area = final_w * final_h
        shrink_ratio = (orig_area - new_area) / orig_area if orig_area > 0 else 0
        
        # 至少1%改进才接受
        if shrink_ratio < 0.01:
            return None
        
        t_end = time.time()
        total_time = (t_end - t_start) * 1000
        print(f"智能吸附成功: 面积收缩 {shrink_ratio*100:.1f}% ({orig_w}x{orig_h} → {final_w}x{final_h}), 耗时 {total_time:.0f}ms")
        
        return [ex1, ey1, ex2, ey2]
        
    except Exception as e:
        print(f"MobileSAM 服务器调用异常: {e}")
        import traceback
        traceback.print_exc()
        return None
