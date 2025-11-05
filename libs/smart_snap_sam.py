# -*- coding: utf-8 -*-
"""
基于 Segment Anything Model (SAM) 的智能吸附模块
提供更精确的边界检测能力，特别适合复杂工程图纸
"""
import cv2
import numpy as np
from libs.label_template import cn_imread
from libs.bbox_utils import calculate_bbox_overlap


# SAM 相关导入（仅保留 MobileSAM）
MOBILE_SAM_AVAILABLE = False
try:
    from mobile_sam import sam_model_registry as MOBILESAM_REGISTRY
    from mobile_sam import SamPredictor as MOBILESAM_PREDICTOR
    MOBILE_SAM_AVAILABLE = True
except Exception:
    MOBILE_SAM_AVAILABLE = False
    


# 全局 MobileSAM 模型实例（延迟加载）
_sam_model = None
_sam_predictor = None
_sam_model_type = None


def _load_sam_model(model_type='mobilesam', device='auto', checkpoint_path=None):
    """
    加载SAM模型（延迟加载，只加载一次）
    
    参数:
        model_type: 固定为 'mobilesam'（其他值将被忽略）
        device: 'cpu', 'cuda', 或 'auto'（自动选择，优先GPU）
        checkpoint_path: 模型权重路径
    """
    global _sam_model, _sam_predictor, _sam_model_type
    
    if _sam_model is not None and _sam_model_type == model_type:
        return _sam_model, _sam_predictor
    
    # 智能设备选择（优先GPU）
    import os
    dev = str(device or 'auto')
    
    # 支持环境变量强制指定设备
    env_device = os.environ.get('SAM_DEVICE', '').lower()
    if env_device in ['cpu', 'cuda']:
        dev = env_device
    
    try:
        import torch
        if dev == 'auto' or dev == 'cpu':
            if torch.cuda.is_available():
                dev = 'cuda'
                print(f"✅ 检测到CUDA可用，使用GPU加速 (设备: {torch.cuda.get_device_name(0)})")
            else:
                dev = 'cpu'
                print("ℹ️ 未检测到CUDA，使用CPU模式")
        elif dev == 'cuda':
            if not torch.cuda.is_available():
                print("⚠️ 指定了CUDA但不可用，回退到CPU")
                dev = 'cpu'
            else:
                print(f"✅ 使用GPU加速 (设备: {torch.cuda.get_device_name(0)})")
    except ImportError:
        print("⚠️ PyTorch未安装，使用CPU模式")
        dev = 'cpu'
    except Exception as e:
        print(f"⚠️ 设备检测失败: {e}，使用CPU模式")
        dev = 'cpu'
    
    # 路径解析辅助
    def resolve_ckpt(default_env_key_list, fallback_path):
        for k in default_env_key_list:
            v = os.environ.get(k)
            if v and os.path.exists(v):
                return v
        return fallback_path
    
    try:
        # 仅保留 MobileSAM 路径
        model_type = 'mobilesam'

        if not MOBILE_SAM_AVAILABLE:
            raise ImportError("MobileSAM 未安装。请运行: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")

        if checkpoint_path is None:
            checkpoint_path = resolve_ckpt([
                'MOBILE_SAM_CHECKPOINT', 'SAM_CHECKPOINT', 'SAM_CKPT'
            ], None)
        
        # 检查权重文件是否存在
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            searched_paths = []
            for env_key in ['MOBILE_SAM_CHECKPOINT', 'SAM_CHECKPOINT', 'SAM_CKPT']:
                env_val = os.environ.get(env_key)
                if env_val:
                    searched_paths.append(f"  {env_key}={env_val}")
            
            if not searched_paths:
                searched_paths.append("  （未设置任何环境变量）")
            
            error_msg = (
                "MobileSAM 权重文件未找到！\n"
                "请按以下步骤配置：\n"
                "1. 下载权重文件：https://github.com/ChaoningZhang/MobileSAM/releases\n"
                "   或访问：https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt\n"
                "2. 设置环境变量（选择一个）：\n"
                "   Windows (PowerShell): $env:MOBILE_SAM_CHECKPOINT=\"C:\\path\\to\\mobile_sam.pt\"\n"
                "   Windows (CMD): setx MOBILE_SAM_CHECKPOINT \"C:\\path\\to\\mobile_sam.pt\"\n"
                "   Linux/macOS: export MOBILE_SAM_CHECKPOINT=\"/path/to/mobile_sam.pt\"\n"
                f"\n当前检查的环境变量：\n" + "\n".join(searched_paths)
            )
            if checkpoint_path:
                error_msg += f"\n检查的文件路径：{checkpoint_path}（文件不存在）"
            raise FileNotFoundError(error_msg)

        # MobileSAM 注册表健壮选择
        try:
            model_key = "vit_t" if "vit_t" in MOBILESAM_REGISTRY else list(MOBILESAM_REGISTRY.keys())[0]
        except Exception:
            model_key = "vit_t"

        _sam_model = MOBILESAM_REGISTRY[model_key](checkpoint=checkpoint_path)
        _sam_model.to(device=dev)
        _sam_predictor = MOBILESAM_PREDICTOR(_sam_model)
        _sam_model_type = 'mobilesam'
            
    except Exception as e:
        print(f"加载SAM模型失败: {e}")
        raise
    
    return _sam_model, _sam_predictor


def detect_object_boundary_sam(img_path, bbox, model_type='mobilesam', device='auto', checkpoint_path=None):
    """
    使用SAM模型检测矩形框内构件的精确边界
    
    参数:
        img_path (str): 图像文件路径
        bbox (list): [xmin, ymin, xmax, ymax] 原始矩形框坐标
        model_type (str): 固定为 'mobilesam'
        device (str): 'cpu' 或 'cuda'
        checkpoint_path (str): 模型权重路径
    
    返回:
        list: [xmin, ymin, xmax, ymax] 优化后的矩形框，失败返回 None
    """
    global _sam_model, _sam_predictor
    
    # 仅检查 MobileSAM
    if not MOBILE_SAM_AVAILABLE:
        print("警告: MobileSAM未安装，回退到其他方法")
        return None
    
    try:
        # 加载模型（如果还未加载）
        _load_sam_model(model_type, device, checkpoint_path)
        
        # 读取图像
        img = cn_imread(img_path)
        if img is None:
            return None
        
        h, w = img.shape[:2]
        xmin, ymin, xmax, ymax = map(int, bbox)
        
        # 边界检查
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(xmin + 1, min(xmax, w))
        ymax = max(ymin + 1, min(ymax, h))
        
        # 准备输入框（SAM需要的格式：左上角和右下角）
        input_box = np.array([xmin, ymin, xmax, ymax])
        
        # 根据模型类型进行预测（仅使用 MobileSAM 预测器）
        if _sam_predictor is not None:
            mask = _predict_sam(_sam_predictor, img, input_box)
        else:
            return None
        
        if mask is None or mask.sum() == 0:
            return None
        
        # 计算mask的最小包围矩形
        optimized_bbox = _mask_to_bbox(mask, offset=[xmin, ymin])
        
        if optimized_bbox is None:
            return None
        
        # 确保坐标在图像范围内
        new_x1, new_y1, new_x2, new_y2 = optimized_bbox
        new_x1 = max(0, min(new_x1, w - 1))
        new_y1 = max(0, min(new_y1, h - 1))
        new_x2 = max(new_x1 + 1, min(new_x2, w))
        new_y2 = max(new_y1 + 1, min(new_y2, h))
        
        # 检查改进程度（放宽阈值，让更多优化生效）
        overlap = calculate_bbox_overlap([xmin, ymin, xmax, ymax], 
                                         [new_x1, new_y1, new_x2, new_y2])
        if overlap > 0.98:  # 从0.95改为0.98，更宽松
            return None  # 改进不明显
        
        return [new_x1, new_y1, new_x2, new_y2]
        
    except Exception as e:
        print(f"SAM检测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def _predict_sam(predictor, img, input_box):
    """使用 MobileSAM 进行预测（优化版）"""
    try:
        # 设置图像
        predictor.set_image(img)
        
        # 方案1：使用多mask输出，选择最佳的
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,  # 输出3个mask，选最好的
        )
        
        # 选择得分最高的mask
        best_mask_idx = scores.argmax()
        mask = masks[best_mask_idx]
        
        # 方案2：添加中心点提示来辅助分割（提升精度）
        # 计算框的中心点作为前景提示
        center_x = (input_box[0] + input_box[2]) / 2
        center_y = (input_box[1] + input_box[3]) / 2
        point_coords = np.array([[center_x, center_y]])
        point_labels = np.array([1])  # 1 = 前景点
        
        # 使用框+点进行第二次预测（更精确）
        masks2, scores2, logits2 = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box[None, :],
            multimask_output=True,
        )
        
        # 选择第二次预测中得分最高的mask
        best_mask_idx2 = scores2.argmax()
        mask2 = masks2[best_mask_idx2]
        
        # 比较两次预测，选择更好的（面积更合理的）
        area1 = mask.sum()
        area2 = mask2.sum()
        box_area = (input_box[2] - input_box[0]) * (input_box[3] - input_box[1])
        
        # 选择面积比例更接近合理范围（20%-95%）的mask
        ratio1 = abs(area1 / box_area - 0.6)  # 目标：60%填充
        ratio2 = abs(area2 / box_area - 0.6)
        
        final_mask = mask if ratio1 < ratio2 else mask2
        
        # 后处理：去除小噪点和填充小孔洞
        final_mask = _postprocess_mask(final_mask)
        
        return final_mask
        
    except Exception as e:
        print(f"SAM预测失败: {e}")
        return None


def _postprocess_mask(mask):
    """对mask进行后处理，去除噪点和填充孔洞"""
    try:
        import cv2
        # 转换为uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 形态学操作：先闭运算填充小孔，再开运算去除小噪点
        kernel = np.ones((5, 5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # 转换回布尔型
        return mask_uint8 > 127
    except Exception:
        # 如果后处理失败，返回原始mask
        return mask




def _mask_to_bbox(mask, offset=[0, 0]):
    """将mask转换为最小包围矩形"""
    try:
        # 找到非零像素
        if isinstance(mask, np.ndarray):
            if mask.dtype != bool:
                mask = mask > 0.5
        else:
            # 如果是tensor，转换为numpy
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


# 便捷函数：使用 MobileSAM 进行检测
def detect_with_sam_auto(img_path, bbox, device='auto'):
    """
    使用 MobileSAM 进行检测
    """
    # 优先尝试服务器 API（如果配置了）
    try:
        import os
        api_url = os.environ.get('MOBILE_SAM_API_URL')
        if api_url:
            result = detect_object_boundary_sam_server(img_path, bbox)
            if result is not None:
                return result, 'mobilesam_server'
    except Exception as e:
        print(f"MobileSAM 服务器调用失败: {e}")
    
    # 回退到本地 MobileSAM（如果已安装）
    if not MOBILE_SAM_AVAILABLE:
        return None, None
    try:
        result = detect_object_boundary_sam(
            img_path, bbox, 
            model_type='mobilesam', 
            device=device
        )
        if result is not None:
            return result, 'mobilesam'
    except Exception as e:
        print(f"尝试 mobilesam 失败: {e}")
        
    
    return None, None


def detect_object_boundary_sam_server(img_path, bbox):
    """调用 MobileSAM 服务器进行图像分割，从返回结果中提取最小包围矩形。
    
    需设置环境变量 MOBILE_SAM_API_URL，例如: http://166.111.80.235:9903/mobile_sam
    服务器预期接口：
        POST /mobile_sam
        请求: 
            - files: {'img': image_bytes} 或 {'img': ('image.jpg', bytes, 'image/jpeg')}
            - data: {'bbox': '[x1,y1,x2,y2]'} 或 JSON格式
        响应:
            - JSON格式，包含 'mask' (base64编码) 或 'bbox' ([x1,y1,x2,y2])
            或包含 'res' 字段包装上述内容
    
    返回: [x1,y1,x2,y2] 或 None
    """
    try:
        import requests
    except ImportError:
        print("MobileSAM 服务器: 缺少 requests 库，无法调用服务器")
        return None
    
    import os
    import json as _json
    
    api_url = os.environ.get('MOBILE_SAM_API_URL')
    if not api_url:
        return None  # 静默失败，回退到本地或传统方法
    
    try:
        # 读取图像
        img = cn_imread(img_path)
        if img is None:
            return None
        H, W = img.shape[:2]
        
        # 裁剪 ROI（围绕用户框，带边距）
        x1, y1, x2, y2 = map(int, bbox)
        margin_ratio = float(os.environ.get('MOBILE_SAM_ROI_MARGIN', '0.10'))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        mx = int(round(bw * margin_ratio))
        my = int(round(bh * margin_ratio))
        rx1 = max(0, x1 - mx)
        ry1 = max(0, y1 - my)
        rx2 = min(W, x2 + mx)
        ry2 = min(H, y2 + my)
        if rx2 <= rx1 or ry2 <= ry1:
            return None
        
        # 发送整图还是 ROI？（默认整图，但发送原始bbox坐标）
        send_roi = os.environ.get('MOBILE_SAM_SEND_ROI', '0') == '1'
        if send_roi:
            roi = img[ry1:ry2, rx1:rx2]
            scale = 1.0
            # 限制尺寸
            max_size = int(os.environ.get('MOBILE_SAM_MAX_SIZE', '1024'))
            rh, rw = roi.shape[:2]
            if max(rh, rw) > max_size:
                scale = max_size / float(max(rh, rw))
                new_w = max(1, int(round(rw * scale)))
                new_h = max(1, int(round(rh * scale)))
                roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # 调整bbox坐标为ROI坐标系
                bbox_roi = [
                    int((x1 - rx1) * scale),
                    int((y1 - ry1) * scale),
                    int((x2 - rx1) * scale),
                    int((y2 - ry1) * scale)
                ]
            else:
                bbox_roi = [x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1]
            send_img = roi
        else:
            send_img = img
            bbox_roi = [x1, y1, x2, y2]
            scale = 1.0
        
        # 编码为 JPEG
        ok, buf = cv2.imencode('.jpg', send_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return None
        file_bytes = np.array(buf).tobytes()
        
        # 准备请求
        raw_files_mode = os.environ.get('MOBILE_SAM_RAW_FILES', '1') == '1'
        if raw_files_mode:
            files_payload = {'img': file_bytes}
        else:
            files_payload = {'img': ('image.jpg', file_bytes, 'image/jpeg')}
        
        # 发送 bbox 作为提示
        data_payload = {'bbox': _json.dumps(bbox_roi)}
        
        # 发送请求
        resp = requests.post(api_url, files=files_payload, data=data_payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        # 兼容返回结构
        if isinstance(data, dict) and 'res' in data:
            data = data['res']
        
        # 解析返回：优先使用 bbox，否则从 mask 计算
        result_bbox = None
        
        if 'bbox' in data:
            # 服务器直接返回 bbox
            bbox_data = data['bbox']
            if isinstance(bbox_data, list) and len(bbox_data) >= 4:
                ex1, ey1, ex2, ey2 = map(float, bbox_data[:4])
                # 如果是 ROI 模式，需要映射回原图坐标
                if send_roi:
                    ex1 = ex1 / scale + rx1
                    ey1 = ey1 / scale + ry1
                    ex2 = ex2 / scale + rx1
                    ey2 = ey2 / scale + ry1
                result_bbox = [int(ex1), int(ey1), int(ex2), int(ey2)]
        
        elif 'mask' in data:
            # 服务器返回 mask（base64 或数组），需要计算最小包围矩形
            mask_data = data['mask']
            # TODO: 解析 base64 mask 或 numpy 数组
            # 这里简化处理，假设服务器已经返回了 bbox
            pass
        
        if result_bbox:
            # 验证改进程度
            overlap = calculate_bbox_overlap([x1, y1, x2, y2], result_bbox)
            if overlap > 0.95:
                return None  # 改进不明显
            return result_bbox
        
        return None
        
    except Exception as e:
        print(f"MobileSAM 服务器调用异常: {e}")
        return None

