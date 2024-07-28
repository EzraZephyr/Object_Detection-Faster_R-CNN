import os
import shutil
import requests
import tarfile
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

def data_download(data_dir):

    os.makedirs(data_dir, exist_ok=True)
    # 创建数据存放的路径 方便调试 如果目录已存在的话不会报错

    voc_url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    # 下载连接

    voc_tar_path = os.path.join(data_dir, 'VOCtrainval_11-May-2012.tar')
    extracted_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2012')
    # 指定下载的内容存放路径和解压内容存放路径

    if not os.path.exists(voc_tar_path):
        # 如果这个路径没有该文件的话

        print("Downloading VOC2012 Dataset...")

        response = requests.get(voc_url, stream=True)
        # 进行请求 并使用流式下载

        with open(voc_tar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        # 以二进制模型打开文件 并分块读取下载内容 每块大小为1024字节并写入

        print("Completed downloading VOC2012 Dataset.")

    if not os.path.exists(extracted_dir):
        print("Decompressing VOC2012 Dataset...")
        with tarfile.open(voc_tar_path, 'r') as tar_ref:
            tar_ref.extractall(data_dir)
        # 解压所有文件到指定目录

        print("Completed decompressing VOC2012 Dataset.")

    annotations_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2012', 'Annotations')
    # 构建标注文件所在的路径

    all_xml_files = [os.path.join(annotations_dir, f) for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    # listdir列出定义路径中所有的子文件 并通过endswith筛选出全部以.xml结尾的文件
    # 与定义的路径进行拼接构建出所有.xml文件的完整路径

    animal_xml_files = [f for f in all_xml_files if animal_annotation(f)]
    # 遍历储存的全部路径 并且放入animal_annotation函数中进行判断是否为动物

    print(f"Find {len(animal_xml_files)} photos containing animals")

    return animal_xml_files

def animal_annotation(xml_file):

    animal_classes = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
    # 定义VOC2012的全部动物类别

    tree = ET.parse(xml_file)
    root = tree.getroot()
    # 解析xml文件 并通过getroot获取根目录 方便访问处理整篇文档

    for obj in root.findall('object'):
        obj_name = obj.find('name').text.lower()
        if obj_name in animal_classes:
            return True
    return False
    # 遍历该xml文件里全部的object节点 并查找标签为name的子节点
    # 并提取子节点的文本内容转换小写 如果该节点在定义的animal_classes中
    # 则返回True

def copy_images(animal_xml_files, data_dir, train_ratio=0.8):

    train_xml_files, test_xml_files = train_test_split(animal_xml_files, train_size=train_ratio, random_state=42)
    # 按比例分割训练集和测试集

    dirs = {
        'train': {
            'images': os.path.join(data_dir, 'animal_images_train'),
            'annotations': os.path.join(data_dir, 'animal_annotations_train')
        },
        'test': {
            'images': os.path.join(data_dir, 'animal_images_test'),
            'annotations': os.path.join(data_dir, 'animal_annotations_test')
        }
    }
    # 通过字典定义清晰的文件格式

    for dir_type in dirs.values():
        os.makedirs(dir_type['images'], exist_ok=True)
        os.makedirs(dir_type['annotations'], exist_ok=True)
    # 创建训练和测试文件夹

    image_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2012', 'JPEGImages')
    # 定义图片文件夹的路径

    def copy_files(xml_files, dir_type):
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            # 获取xml的根目录

            filename = root.find('filename').text
            image_path = os.path.join(image_dir, filename)
            # 提取里面的图像文件名 并合并路径返回

            shutil.copy(image_path, dirs[dir_type]['images'])
            shutil.copy(xml_file, dirs[dir_type]['annotations'])
            # 将图片文件和标注文件分别拷贝到创建好的文件夹中

    copy_files(train_xml_files, 'train')
    copy_files(test_xml_files, 'test')
    # 分别提取上面定义好的路径进行拷贝

    print("Completed copying images and annotations")

    return dirs['train']['images'], dirs['train']['annotations'], dirs['test']['images'], dirs['test']['annotations']
    # 返回定义的路径



def data_process(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    # 获取图像信息

    objects = []
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        # 获取标注的类别名称

        bbox = obj.find('bndbox')
        bbox = [int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                int(bbox.find('xmax').text), int(bbox.find('ymax').text)]
        # 获取边界框的坐标

        objects.append((obj_name, bbox))

    return filename, width, height, objects

def data_loader(data_dir):
    animal_xml_files = data_download(data_dir)
    animal_images_train, animal_annotations_train, animal_images_test, animal_annotations_test = copy_images(animal_xml_files, data_dir)

    annotation_files = [os.path.join(animal_annotations_train, f) for f in os.listdir(animal_annotations_train)]
    # 通过listdir从animal_annotation_dir中读取全部的文件 并与文件夹名拼成路径列表

    if annotation_files:
        filename, width, height, objects = data_process(annotation_files[0])
        print(filename, width, height)
        print(objects)

    else:
        print("No annotation files found.")

if __name__ == '__main__':
    data_dir = '../data/VOC2012'
    data_loader(data_dir)
