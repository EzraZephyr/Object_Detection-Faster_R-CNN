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
        # Get all files in the image and annotation directories

        self.animal_classes = ['background', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.animal_classes)}
        # Define class_to_idx to map class names to indices for easier usage


    def __len__(self):
        return len(self.image_files)
        # Return the size of the dataset

    def __getitem__(self, idx):

        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        # Concatenate the path of the image file and open it in RGB mode

        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])
        boxes, labels = self.parse_annotation(annotation_path)
        # Concatenate the path of the annotation file and get boxes and labels

        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor([self.class_to_idx[label] for label in labels], dtype=torch.int64)
        # Convert boxes and labels to tensors and store them in a dictionary

        if self.transform:
            img = self.transform(img)
            # Apply the transform to the image if it is defined

        return img, target

    def parse_annotation(self, annotation_path):

        tree = ET.parse(annotation_path)
        root = tree.getroot()
        # Parse the XML file and get the root node

        boxes, labels = [], []
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
            # Check if the label is an animal class and ignore if not

        return boxes, labels

def data_process(image_dir, annotation_dir):

    transform = transforms.Compose([
        transforms.ToTensor(),
        # Define the transform to convert images to tensors
        # Note: No resizing to avoid altering bounding box accuracy
    ])

    dataset = VOCDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        transform=transform
    )

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    # Explanation of collate_fn:
    # After DataLoader calls __getitem__, the data is passed to collate_fn
    # To ensure consistency in batch data format, convert data to tuple format
    # *x unpacks [(img1, target1), (img2, target2)] and passes it to zip
    # zip then packs elements in the same position together ((img1, img2), (target1, target2))
    # Note: This returns an iterator which is converted to a tuple

    return data_loader
