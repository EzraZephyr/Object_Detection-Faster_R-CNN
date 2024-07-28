import os
import torch
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class VOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform

        self.image_files = list(sorted(os.listdir(image_dir)))
        self.annotation_files = list(sorted(os.listdir(annotation_dir)))
        # 获取两个文件夹下的全部文件

        self.animal_classes = ['background', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.animal_classes)}
        # 定义class_to_idx可以使字符串转换为对应的索引 避免使用str


    def __len__(self):
        return len(self.image_files)
        # 返回数据集大小

    def __getitem__(self, idx):

        img_path = os.path.join(self.image_dir,self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        # 拼接图像文件的路径 并打开图像转换为RGB模式

        annotation_path = os.path.join(self.annotation_dir,self.annotation_files[idx])
        boxes,labels = self.parse_annotation(annotation_path)
        # 拼接标注文件的路径并放入parse_annotation函数获得边界框和标签

        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor([self.class_to_idx[label] for label in labels], dtype=torch.int64)
        # 将边界框和标签转换为张量形式并用字典进行储存

        if self.transform:
            img = self.transform(img)
            # 将图像转化为Transform中的格式

        return img, target

    def parse_annotation(self, annotation_path):

        tree = ET.parse(annotation_path)
        root = tree.getroot()
        # 解析xml文件并获取根节点

        boxes, labels = [],[]
        for obj in root.findall('object'):
            label = obj.find('name').text.lower()
            if label in self.animal_classes:
                labels.append(label)
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
            # 判断每个被选择的边框的标签是否为动物
            # 如果不为动物的话直接忽视 不参与训练

        return boxes,labels

def data_process(image_dir,annotation_dir):

    transform = transforms.Compose([
        transforms.ToTensor(),
        # 定义transform为转换张量
        # 这里不进行调整大小是因为这样的操作
        # 可能会导致标注的边框信息变得不准确

    ])

    dataset = VOCDataset(
        image_dir = image_dir,
        annotation_dir = annotation_dir,
        transform=transform
    )

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    # 对collate_fn的详细解释：
    # 再DataLoader调用__getitem__加载数据之后 会将这些数据传递给collate_fn中
    # 为了确保批次中的数据格式一致 方便后续的处理 需要将数据转化为元组形式
    # *x会将[(img1, target1), (img2, target2)]进行解包 传递给zip参数
    # 后经过zip 会将相同位置的元素打包在一起((img1, img2), (target1, target2))
    # 注意 这个时候返回的是一个迭代器 需要再经过tuple转换为元组


    return data_loader