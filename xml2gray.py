import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append({'name': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
    
    return objects

def visualize_annotation(image_path, objects):
    # Read the image
    image = cv2.imread(image_path)

    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Add rectangles for each object
    for obj in objects:
        rect = patches.Rectangle((obj['xmin'], obj['ymin']),
                                 obj['xmax'] - obj['xmin'],
                                 obj['ymax'] - obj['ymin'],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    xml_file = "E:\\crack_detect\\dataset\\object_detetion_data\\all_images\\beam\\1000\Annotations\\crack1.xml"
    image_path = "E:\\crack_detect\\dataset\\object_detetion_data\\all_images\\beam\\1000\\JPEGImages\\crack1.jpg"
    
    objects = parse_xml(xml_file)
    visualize_annotation(image_path, objects)
