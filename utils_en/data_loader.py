import os
import shutil
import requests
import tarfile
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

def data_download(data_dir):

    os.makedirs(data_dir, exist_ok=True)
    # Create the data directory if it doesn't exist for debugging convenience

    voc_url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    # Download link

    voc_tar_path = os.path.join(data_dir, 'VOCtrainval_11-May-2012.tar')
    extracted_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2012')
    # Path to save the downloaded content and the extracted content

    if not os.path.exists(voc_tar_path):
        # If the file does not exist at the specified path

        print("Downloading VOC2012 Dataset...")

        response = requests.get(voc_url, stream=True)
        # Make the request and download the file in chunks

        with open(voc_tar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        # Open the file in binary mode and write the downloaded content in chunks of 1024 bytes

        print("Completed downloading VOC2012 Dataset.")

    if not os.path.exists(extracted_dir):
        print("Decompressing VOC2012 Dataset...")
        with tarfile.open(voc_tar_path, 'r') as tar_ref:
            tar_ref.extractall(data_dir)
        # Extract all files to the specified directory

        print("Completed decompressing VOC2012 Dataset.")

    annotations_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2012', 'Annotations')
    # Build the path to the annotations directory

    all_xml_files = [os.path.join(annotations_dir, f) for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    # List all files in the directory and filter for those ending in .xml, constructing the full path for each

    animal_xml_files = [f for f in all_xml_files if animal_annotation(f)]
    # Check each file to see if it contains animal annotations

    print(f"Find {len(animal_xml_files)} photos containing animals")

    return animal_xml_files

def animal_annotation(xml_file):

    animal_classes = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
    # Define all animal categories in VOC2012

    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Parse the xml file and get the root for easy access to the entire document

    for obj in root.findall('object'):
        obj_name = obj.find('name').text.lower()
        if obj_name in animal_classes:
            return True
    return False
    # Traverse all 'object' nodes in the xml, find the 'name' child node,
    # convert its text content to lowercase, and check if it is in the animal_classes list
    # If yes, return True

def copy_images(animal_xml_files, data_dir, train_ratio=0.8):

    train_xml_files, test_xml_files = train_test_split(animal_xml_files, train_size=train_ratio, random_state=42)
    # Split the files into training and test sets according to the specified ratio

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
    # Define clear folder structure using a dictionary

    for dir_type in dirs.values():
        os.makedirs(dir_type['images'], exist_ok=True)
        os.makedirs(dir_type['annotations'], exist_ok=True)
    # Create the folders for training and testing data

    image_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2012', 'JPEGImages')
    # Define the path to the images directory

    def copy_files(xml_files, dir_type):
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            # Get the root of the xml file

            filename = root.find('filename').text
            image_path = os.path.join(image_dir, filename)
            # Extract the image file name and construct its path

            shutil.copy(image_path, dirs[dir_type]['images'])
            shutil.copy(xml_file, dirs[dir_type]['annotations'])
            # Copy the image file and the annotation file to the respective folders

    copy_files(train_xml_files, 'train')
    copy_files(test_xml_files, 'test')
    # Copy the files to their respective folders

    print("Completed copying images and annotations")

    return dirs['train']['images'], dirs['train']['annotations'], dirs['test']['images'], dirs['test']['annotations']
    # Return the paths to the training and testing images and annotations

def data_process(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    # Extract image information

    objects = []
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        # Get the annotated category name

        bbox = obj.find('bndbox')
        bbox = [int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                int(bbox.find('xmax').text), int(bbox.find('ymax').text)]
        # Get the bounding box coordinates

        objects.append((obj_name, bbox))

    return filename, width, height, objects

def data_loader(data_dir):
    animal_xml_files = data_download(data_dir)
    animal_images_train, animal_annotations_train, animal_images_test, animal_annotations_test = copy_images(animal_xml_files, data_dir)

    annotation_files = [os.path.join(animal_annotations_train, f) for f in os.listdir(animal_annotations_train)]
    # Read all files from animal_annotations_train using listdir and construct their paths

    if annotation_files:
        filename, width, height, objects = data_process(annotation_files[0])
        print(filename, width, height)
        print(objects)

    else:
        print("No annotation files found.")

if __name__ == '__main__':

    data_dir = '../data/VOC2012'
    data_loader(data_dir)
