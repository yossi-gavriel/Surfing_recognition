import cv2
from detecto import core
import os
# calculate intersection over union between two rectangles
from xml_util import create_xml


def calc_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_object_inside_wave(object_name, labels, boxes):
    """
    Check if the boxes of the wave contains the boxes of the surfer
    """
    object_idxs = list()
    wave_idxs = list()
    for i, label in enumerate(labels):

        if label == object_name:
            object_idxs.append(i)
        elif label == 'wave':
            wave_idxs.append(i)

    if len(object_idxs) == 0 or len(wave_idxs) == 0:
        return -1, -1

    max_iou = {'object_idx': -1, 'wave_idx': -1, 'iou': 0}
    for object_idx in object_idxs:
        for wave_idx in wave_idxs:
            iou = calc_intersection_over_union(boxes[object_idx], boxes[wave_idx])
            if iou > max_iou['iou']:
                max_iou = {'object_idx': object_idx, 'wave_idx': wave_idx, 'iou': iou}

    return max_iou['object_idx'], max_iou['wave_idx']


def detect_objects_and_export_as_xml(video_path, object_detection_model_path, labels, score_threshold, output_folder):

    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_count = 0
    while True:

        ret, frame = video.read()

        detection_model = core.Model.load(object_detection_model_path, labels)
        predictions = detection_model.predict(frame)
        labels = predictions[0]
        boxs = predictions[1]
        score = predictions[2]

        labels = ['surfer' if label == 'surfing_person' else label for i, label in enumerate(labels)]

        surfer_idx, wave_idx = get_object_inside_wave('surfer', labels, boxs)
        surfboard_idx, _ = get_object_inside_wave('surfboard', labels, boxs)

        labels = [labels[wave_idx], labels[surfer_idx], labels[surfboard_idx]]
        boxs = [boxs[wave_idx], boxs[surfer_idx], boxs[surfboard_idx]]

        filename = f'{frames_count}.JPG'
        path = os.path.join(output_folder, filename)

        create_xml(output_folder, filename, path, frame_width, frame_height, labels, boxs)

        cv2.imwrite(path, frame)

        frames_count += 1


if __name__ == '__main__':

    video_path = r"C:\Users\User\Downloads\videoplayback.mp4"
    labels = ['surfing_person', 'wave', 'surfboard']
    score_threshold = 0.5
    object_detection_model_path = r"C:\Users\User\Desktop\SCOOL\try\Surfing-analysis\trained_models\dynamic_camera_object_detection.pth"
    output_folder = r"C:\Users\User\Desktop\SCOOL\Surfing_recognition\new_data_for_train"
    detect_objects_and_export_as_xml(video_path, object_detection_model_path, labels, score_threshold, output_folder)
