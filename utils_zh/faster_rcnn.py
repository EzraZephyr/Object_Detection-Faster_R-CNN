from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50

def ResNet50_loader():

    backbone = resnet50(pretrained=True)
    # 下载resnet50模型上所有的预训练权重

    backbone = nn.Sequential(*(list(backbone.children())[:-2]))
    # 获取ResNet50模型的所有子模块并剪去最后的全连接层和平均池化层
    # 并重新组合成一个新的容器

    backbone.out_channels = 2048
    # ResNet的最后一个卷积层输出是2048 所以要定义一下

    return backbone

def Model():


    rpn_anchor_generator = AnchorGenerator(
        sizes=((16,32,64,128,256),),
        # 定义基础锚框大小

        aspect_ratios=((0.5, 1.0, 1.5),)
        # 定义基础长宽比
    )

    roi_pooling = MultiScaleRoIAlign(
        featmap_names=['0'],
        # 使用单特征图 简化模型

        output_size=7,
        # 平均池化为7*7

        sampling_ratio=2,
        # 使用2*2的采样点进行微调

    )

    return rpn_anchor_generator, roi_pooling

def load_model():

    backbone = ResNet50_loader()
    rpn_anchor_generator, roi_pooling = Model()
    model = FasterRCNN(
        backbone=backbone,
        rpn_anchor_generator=rpn_anchor_generator,
        roi_pooler=roi_pooling,
        num_classes=7,
        # 设置类别数为7 包含6个动物类别和一个背景类别
        # 背景类别用于标记图像中不包含目标对象的区域。
    )

    return model


