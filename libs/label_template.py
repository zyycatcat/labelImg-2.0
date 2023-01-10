import cv2
from xml.dom.minidom import parse, Document
import xml.etree.ElementTree as ET
import xml.dom.minidom
import numpy as np
import os
from jinja2 import Environment, PackageLoader


class Writer:
    def __init__(self, path, width, height, depth=3, database='Unknown', segmented=0):
        environment = Environment(loader=PackageLoader('pascal_voc_writer', 'templates'), keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')

        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }

    def addObject(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0):
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
        })

    def save(self, annotation_path):
        with open(annotation_path, 'w', encoding='utf-8') as file:  # 否则含有中文路径的情况会有乱码
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)


# 获取待匹配的模板位置
def get_xml_data(xml_file_path):
    res = []
    try:
        doc = xml.dom.minidom.parse(xml_file_path)
        root = doc.documentElement

        objects = root.getElementsByTagName('object')
        for obj in objects:
            if int(obj.getElementsByTagName('difficult')[0].firstChild.data) == 1:
                bbox = obj.getElementsByTagName('bndbox')[0]
                res.append([obj.getElementsByTagName('name')[0].firstChild.data, [
                    int(bbox.getElementsByTagName('xmin')[0].firstChild.data),
                    int(bbox.getElementsByTagName('ymin')[0].firstChild.data),
                    int(bbox.getElementsByTagName('xmax')[0].firstChild.data),
                    int(bbox.getElementsByTagName('ymax')[0].firstChild.data)], 1])
                print(obj.getElementsByTagName('name')[0].firstChild.data)

        # file = open(xml_file_path, 'w', encoding='utf-8')
        # doc.writexml(file, indent='', addindent='', newl='', encoding='utf-8')
        # file.close()
    except Exception as e:
        print(e)
    finally:
        return res


# 写生成的图片对应的xml
def write_xml(img_file_path, xml_file_path, width, height, tag_cood_tuple_list):
    writer = Writer(img_file_path, width, height)
    for tuple in tag_cood_tuple_list:
        tag_name = tuple[0]
        coor = tuple[1]
        x_min, y_min, x_max, y_max = coor[0], coor[1], coor[2], coor[3]
        writer.addObject(tag_name, x_min, y_min, x_max, y_max, difficult=tuple[2])
    writer.save(xml_file_path)


def bbox_IOU(bbox_a, bbox_b):
    a_xmin, a_ymin, a_xmax, a_ymax = bbox_a[0], bbox_a[1], bbox_a[2], bbox_a[3]
    b_xmin, b_ymin, b_xmax, b_ymax = bbox_b[0], bbox_b[1], bbox_b[2], bbox_b[3]
    if b_xmin > a_xmax or a_xmin > b_xmax or b_ymin > a_ymax or a_ymin > b_ymax:
        return 0
    a_area = (a_xmax - a_xmin) * (a_ymax - a_ymin)
    b_area = (b_xmax - b_xmin) * (b_ymax - b_ymin)
    x_min = max(a_xmin, b_xmin)
    x_max = min(a_xmax, b_xmax)
    y_min = max(a_ymin, b_ymin)
    y_max = min(a_ymax, b_ymax)
    overlap_area = (x_max - x_min) * (y_max - y_min)
    if overlap_area <= 0:
        print('ERROR:0-MINUS OVERLAP AREA')
        return 0
    return overlap_area / (a_area + b_area - overlap_area)


# 根据模板匹配，获取其它同类型的标注
def get_tag_cood_tuple_list(template_list: list, img_path: str, dump_flag=False, filter_list=None, thresh=0.8):
    if filter_list is None:
        filter_list = template_list
    tag_cood_tuple_list = template_list.copy()
    img = cn_imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 获取灰度图
    for temp in template_list:
        tag_name = temp[0]
        coor = temp[1]
        template = img_gray[coor[1]: coor[3], coor[0]: coor[2]]
        template = cv2.resize(template, dsize=None, fx=1, fy=1)
        result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= thresh)
        pts = []
        for pt in zip(*loc[::-1]):
            pts.append([pt[0], pt[1]])
        for pt in pts:
            bottom_right = (pt[0] + template.shape[1], pt[1] + template.shape[0])
            target_bbox = [pt[0], pt[1], bottom_right[0], bottom_right[1]]
            is_duplicate = False
            for old_tag in filter_list + tag_cood_tuple_list:
                if bbox_IOU(old_tag[1], target_bbox) > 0.9:
                    is_duplicate = True
                    break
            if not is_duplicate:  # 跳过重复的检测框
                tag_cood_tuple_list.append([tag_name, target_bbox, 0])
                if dump_flag:
                    cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 3)
        if dump_flag:
            cn_imwrite(img_path[:-4] + '_debug.jpg', img)
    return tag_cood_tuple_list

def cn_imread(cn_path: str, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(cn_path, dtype=np.uint8), flags)


def cn_imwrite(save_path: str, img_save):
    suffix = save_path[save_path.index('.'):]
    cv2.imencode(suffix, img_save)[1].tofile(save_path)


if __name__ == '__main__':
    xml_path = r'D:\个人\研二\fasterrcnn_train\background\pdf_test\1.xml'
    template_list = get_xml_data(xml_path)
    print(template_list)
    img_path = r'D:\个人\研二\fasterrcnn_train\background\pdf_test\1.jpg'
    tag_cood_tuple_list = get_tag_cood_tuple_list(template_list, img_path, dump_flag=True)
    img = cn_imread(img_path)
    h, w, _ = img.shape
    write_xml(img_path, xml_path, w, h, tag_cood_tuple_list)
