from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50

def ResNet50_loader():
    backbone = resnet50(pretrained=True)
    # Download the pre-trained weights for the ResNet50 model

    backbone = nn.Sequential(*(list(backbone.children())[:-2]))
    # Get all the child modules of ResNet50 and remove the final fully connected layer and average pooling layer
    # Then, reassemble them into a new container

    backbone.out_channels = 2048
    # The last convolutional layer of ResNet outputs 2048 channels, so we need to define this

    return backbone

def Model():
    rpn_anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256),),
        # Define the base anchor sizes

        aspect_ratios=((0.5, 1.0, 1.5),)
        # Define the base aspect ratios
    )

    roi_pooling = MultiScaleRoIAlign(
        featmap_names=['0'],
        # Use a single feature map to simplify the model

        output_size=7,
        # Perform average pooling to 7x7

        sampling_ratio=2,
        # Use a 2x2 sampling ratio for fine-tuning
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
        # Set the number of classes to 7, including 6 animal categories and one background category
        # The background category is used to mark areas in the image that do not contain the target object
    )

    return model
