import xml.dom.minidom as md
from xml.etree import ElementTree as ET


def get_xml_objects(file_path):
    """
    Get the objects from the xml file
    """
    root = ET.parse(file_path).getroot()

    folder = root.find('folder').text
    filename = root.find('filename').text
    path = root.find('path').text
    width = root.find('size/width').text
    height = root.find('size/height').text

    boxes = list()
    labels = list()

    for object_i in root.findall('object'):

        current_box = list()
        name = object_i.find('name').text
        xmin = int(float(object_i.find('bndbox/xmin').text))
        ymin = int(float(object_i.find('bndbox/ymin').text))
        xmax = int(float(object_i.find('bndbox/xmax').text))
        ymax = int(float(object_i.find('bndbox/ymax').text))

        labels.append(name)
        boxes.append([xmin, ymin, xmax, ymax])

    return folder, filename, path, width, height, labels, boxes


def get_obj(label, boxes):
    """
    Create a XML object from the given label and boxes
    """
    return    rf'''<object>
        <name>{label}</name>
        <pose>Unspecified</pose>
        <truncated>1</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{boxes[0]}</xmin>
            <ymin>{boxes[1]}</ymin>
            <xmax>{boxes[2]}</xmax>
            <ymax>{boxes[3]}</ymax>
        </bndbox>
    </object>
'''


def create_xml(folder, filename, path, width, height, labels, boxes):
    """
    Create an XML for Object detection from the given details
    """
    path = path.replace('\\', '/')
    filename = filename.split('\\')[-1]
    xml_data =rf'''<annotation>
    <folder>{folder}</folder>
    <filename>{filename}</filename>
    <path>{path}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>{labels[0]}</name>
        <pose>Unspecified</pose>
        <truncated>1</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{boxes[0][0]}</xmin>
            <ymin>{boxes[0][1]}</ymin>
            <xmax>{boxes[0][2]}</xmax>
            <ymax>{boxes[0][3]}</ymax>
        </bndbox>
    </object>
'''

    for i, label in enumerate(labels):
        if i == 0:
            continue
        xml_data += get_obj(label, boxes[i])



    xml_data += '''</annotation>'''


    tree = ET.XML(xml_data)

    with open(path.replace('JPG', 'xml'), "wb") as f:
        f.write(ET.tostring(tree))



