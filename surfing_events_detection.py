#!/usr/bin/env python
# coding: utf-8
import os
import cv2
import sys
import torch
import pickle
import gc
import time
import numpy as np
import pandas as pd
from torchvision import transforms
from collections import defaultdict
from detecto import core
from tensorflow.keras.models import load_model
from detecto.utils import normalize_transform
from keras.utils import load_img, img_to_array


cwd = os.getcwd().replace('\\', '/')
root_directory = cwd.replace('/notebooks/surfing_analysis', '')

sys.path.insert(0, root_directory)

from config import surfing_analysis_config, surfer_direction_config

camera_setup = surfing_analysis_config['camera_setup']
to_export_video = surfing_analysis_config['to_export_video']

surfer_direction_img_size = surfer_direction_config['img_size']
surfer_direction_left_or_right_threshold = surfing_analysis_config['surfer_direction_left_or_right_threshold']
white_level_threshold = surfing_analysis_config['white_level_threshold']
is_fall_frames_window = surfing_analysis_config['is_fall_frames_window']
surfer_above_wave_threshold = surfing_analysis_config['surfer_above_wave_threshold']
exercise_frames_window = surfing_analysis_config['exercise_frames_window']
pipe_frames_window = surfing_analysis_config['pipe_frames_window']
direction_threshold = surfing_analysis_config['direction_threshold']
direction_window = surfing_analysis_config['direction_window']
starting_direction_frame_count = surfing_analysis_config['starting_direction_frame_count']
range_dist_threshold_direction_analysis = surfing_analysis_config['range_dist_threshold_direction_analysis']
range_length_threshold_direction_analysis = surfing_analysis_config['range_length_threshold_direction_analysis']
smooth_tr_direction_analysis = surfing_analysis_config['smooth_tr_direction_analysis']
window_threshold_direction_analysis = surfing_analysis_config['window_threshold_direction_analysis']
right_percent_threshold = surfing_analysis_config['right_percent_threshold']
left_percent_threshold = surfing_analysis_config['left_percent_threshold']
check_is_fall_frame = surfing_analysis_config['check_is_fall_frame']
sleep_time = surfing_analysis_config['sleep_time']
gc_files_threshold = surfing_analysis_config['gc_files_threshold']


labels = ['wave', 'surfer']


# Create results dirs

object_detection_model_path = os.path.join(root_directory, f'trained_models/{camera_setup}_object_detection.pth')
object_direction_model_path = os.path.join(root_directory, 'trained_models/direction_model.h5')

videos_source_path = os.path.join(root_directory, f'datasets/videos/{camera_setup}')
videos_res_path = os.path.join(root_directory, f'results/surfing_analysis/{camera_setup}/videos')
predicted_ranges_dir_path = os.path.join(root_directory, f'results/surfing_analysis/{camera_setup}/predicted_ranges')
predicted_ranges_path = os.path.join(predicted_ranges_dir_path, 'predicted_ranges.pkl')
surfing_analysis_predicted_info = os.path.join(root_directory, f'results/surfing_analysis/{camera_setup}/surfing_analysis_features')

    
for dir_path in [videos_res_path, predicted_ranges_dir_path, surfing_analysis_predicted_info]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

surfer_dict = {'dedicated_pools': 'surfer', 'dynamic_camera': 'surfing_person'}

# calculate intersection over union between two rectangles
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


def get_first_labels(labels, labels_names):
    """
    Get the first labels from the labels dict
    """
    label_ids_map = {label:0 for label in labels_names}
    labels_ids = list()
    for label in labels:
        labels_ids.append(label_ids_map[label])
        label_ids_map[label] += 1
    return labels_ids, label_ids_map


def is_wave_contains_surfer(labels, boxes):
    """
    Check if the boxes of the wave contains the boxes of the surfer 
    """
    surfer_idx = wave_idx = -1
    for i, label in enumerate(labels):
        
        if label == surfer_dict[camera_setup] and surfer_idx == -1:
            surfer_idx = i
            continue
        elif label == 'wave' and wave_idx == -1:
            wave_idx = i
            continue
    if surfer_idx == -1 or wave_idx == -1:
        return False
    
    if boxes[surfer_idx][0] > boxes[wave_idx][0] and boxes[surfer_idx][1] > boxes[wave_idx][1] and boxes[wave_idx][2] > boxes[surfer_idx][2] and boxes[wave_idx][3] > boxes[surfer_idx][3]:
            return True
    return False


def get_surfer_insied_wave(labels, boxes):
    """
    Check if the boxes of the wave contains the boxes of the surfer 
    """
    surfer_idxs = list()
    wave_idxs = list()
    for i, label in enumerate(labels):
        
        if label == surfer_dict[camera_setup]:
            surfer_idxs.append(i)
        elif label == 'wave':
            wave_idxs.append(i)
            
    if len(surfer_idxs) == 0 or len(wave_idxs) == 0:
        return False
    
    max_iou = {'surfer_idx':-1, 'wave_idx':-1, 'iou':-1}
    for surfer_idx in surfer_idxs:
        for wave_idx in wave_idxs:
            iou = calc_intersection_over_union(boxes[surfer_idx], boxes[wave_idx])
            if iou > max_iou['iou']:
                max_iou = {'surfer_idx':surfer_idx, 'wave_idx':wave_idx, 'iou':iou}

    return max_iou


def put_analysis(frame, pipe=0, pipe2=0, up=0, down=0, left=0, right=0, normal_factor=0, exercise=0, excercise_size=0, surfer_box_shape_count=0, is_fall=False, frames='', direction = '', new_left=0, new_right=0, to_export_video=True, smooth_left=0, smooth_right=0, smooth_up=0, white_level=0):
    """
    Put the analysis details on the given frame 
    """
    surfing_analysis = f'up:{smooth_up}, left:{smooth_left}, right:{smooth_right}'
    surfing_analysis2 = f'pipe:{pipe}, pipe2: {pipe2}, exercise: {exercise}'
    surfing_analysis3 = f'body_direction:{direction}, frames:{frames}, is_fall:{is_fall}'
    surfing_analysis4 = f'wave_size:{normal_factor}, white_level:{white_level}'
    
    if to_export_video:
        
        color = (255, 255, 255)
        cv2.putText(frame, surfing_analysis, (int(len(frame)/2), int(len(frame)/8)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3) 

        cv2.putText(frame, surfing_analysis2, (int(len(frame)/2), int(len(frame)/7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3) 

        cv2.putText(frame, surfing_analysis3, (int(len(frame)/2), int(len(frame)/6)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3) 

        cv2.putText(frame, surfing_analysis4, (int(len(frame)/2), int(len(frame)/5)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3) 


def put_analysis(frame, pipe=0, pipe2=0, up=0, down=0, left=0, right=0, normal_factor=0, exercise=0, excercise_size=0, surfer_box_shape_count=0, is_fall=False, frames='', direction = '', new_left=0, new_right=0, to_export_video=True, smooth_left=0, smooth_right=0, smooth_up=0, white_level=0, surfing_direction='', is_static_camera=False, height_width=0):
    """
    Put the analysis details on the given frame 
    """
    
    if not is_static_camera:
        cutback_left = smooth_left if surfing_direction == "right" else 0
        cutback_right = smooth_right if surfing_direction == "left" else 0
    else:
        cutback_left = smooth_left
        cutback_right = smooth_right
    surfing_analysis = f'Snap:{smooth_up}, Air: {exercise}'
    surfing_analysis2 = f'Cutback left:{cutback_left}, Cutback right:{cutback_right}'
    surfing_analysis3 = f'pipe:{pipe}, pipe2: {pipe2}'
    surfing_analysis4 = f'body_direction:{direction}, frames:{frames}, surfing_direction:{surfing_direction}'
    surfing_analysis5 = f'wave_size:{normal_factor}, white_level:{white_level}'
    surfing_analysis7 = f'surfer_box_shape_count:{surfer_box_shape_count}, height_width:{height_width}'

    
    if to_export_video:
        
        color = (255, 255, 255)
        cv2.putText(frame, surfing_analysis, (int(len(frame)/2), int(len(frame)/20)+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) 

        cv2.putText(frame, surfing_analysis2, (int(len(frame)/2), int(len(frame)/20)+70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) 

        cv2.putText(frame, surfing_analysis3, (int(len(frame)/2), int(len(frame)/20)+90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) 

        cv2.putText(frame, surfing_analysis4, (int(len(frame)/2), int(len(frame)/20)+110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(frame, surfing_analysis5, (int(len(frame)/2), int(len(frame)/20)+130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if is_fall != '':
            
            surfing_analysis6 = f'Surfer fall:{is_fall}'
            
            cv2.putText(frame, surfing_analysis6, (int(len(frame)/2), int(len(frame)/20)+150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, surfing_analysis7, (int(len(frame)/2), int(len(frame)/20)+170),

                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def calculate_wave_wight_level(box, frame):
    """
    Calculate the wave white level
    """
    x_min, y_min, x_max, y_max = box
    ans = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

    ans = np.mean(ans, axis=2)

    ans = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

    white_level = round(np.sum(ans > 150)/(ans.size), 2)
    
    return white_level


def predict_surfer_direction(box, frame, direction_model):
    """
    Predict the surfer body direction
    """
    x_min, y_min, x_max, y_max = box
    ans = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
    ans_rgb = cv2.cvtColor(ans, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(ans_rgb, surfer_direction_img_size, interpolation = cv2.INTER_AREA)
    direction = predict_direction(resized, direction_model)
    return direction
    

def draw_rect(label_info, frame, box, cv2, to_export_video=True):
    """
    Draw a rectangle with labels and ids on the given frame for each detected object 
    """
    # Since the predictions are for scaled down frames,
    # we need to increase the box dimensions
    # box *= scale_down_factor  # TODO Issue #16
    
    if to_export_video:
        
        # Create the box around each object detected
        # Parameters: frame, (start_x, start_y), (end_x, end_y), (r, g, b), thickness
        if 'Surfer' in label_info:
            
            color = (0, 0, 255)
        elif 'Wave' in label_info:
            
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
            
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)

        # Write the label and score for the boxes
        # Parameters: frame, text, (start_x, start_y), font, font scale, (r, g, b), thickness
        cv2.putText(frame, label_info, (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)


def test(labels_names):
    get_first_labels(['wave', 'surfing_person', 'surfboard'], labels_names)


# # helper functions for the surfing events detection

# # the logic function that gets the surfing video and perform analysis

# # helper function for tracking and selecting the detected objects  

# In[6]:


def get_maximum_matches(range_lst_1, range_lst_2, threshold=0):
    """
    Get the pairs with maximum intersection_over_union match between to list ranges
    """
    pairs_iou_list = list()
    for i, range_1 in range_lst_1:
        for j, range_2 in range_lst_2:
            calculate_iou = calc_intersection_over_union(range_1, range_2)
            pairs_iou_list.append((i, j, calculate_iou))
    
    pairs_iou_list.sort(key=lambda tup: tup[2], reverse=True)  # sorts in place
    ises = list()
    jses = list()
    matched_pairs = list()
    for i, j, iou in pairs_iou_list:
        if i not in ises and j not in jses and iou>= threshold:
            ises.append(i)
            jses.append(j)
            matched_pairs.append((i,j))
    return matched_pairs, ises


def get_predictions_map(boxes, labels, labels_names):
    """
    Create a list of idx and boxes for each label
    """    
    predictions_map = defaultdict(list)
    for i, elem in enumerate(boxes):
        label = labels[i]
        if label in labels_names:
            predictions_map[label].append((i, [elem[0], elem[1], elem[2], elem[3]]))
            
    return predictions_map


def predict_and_compare_with_labels(prev_res, current_res, next_ids, labels_names, min_score):
    """
    Select the predictions that match previous predictions 
    Select the predictions that doesn't overlap to the predictions that matched to prev predictions
    produce id's for the selected predictions:
        if prediction matched to previous give the same id
        produce new id otherwise 
    """
    current_ids = [0 for i in current_res]
    
    percent_per_label = {label:0.01 for label in labels_names}
    
    predicted_labels, predicted_boxes, predicted_scores = current_res
    prev_labels, prev_boxes, prev_scores, prev_ids = prev_res     
        
    # create a detections map with the prev predictions labels as a key 
    prev_map = get_predictions_map(prev_boxes, prev_labels, labels_names)

    # create a detections map with the current predictions labels as a key 
    current_map = get_predictions_map(predicted_boxes, predicted_labels, labels_names)
    
    matched_pairs_per_label = dict()
    pred_idx_to_remove = list()
    
    new_labels = list()
    new_boxes = list()
    new_scores = list()
    new_matched_pairs = dict()
    for label in labels_names: # calculate current predictions and prev matches pairs
        
        matched_pairs_per_label[label] = get_maximum_matches(current_map[label], prev_map[label], threshold=0)
        matched_pairs, idx_predicted_with_match = matched_pairs_per_label[label]
        
        new_matched_pairs[label] = list()
        new_matched_pairs[label].append(list())
        new_matched_pairs[label].append(list())
        
        # construct new predictions lists
        for cur_idx, prev_idx in matched_pairs:
            new_labels.append(label)
            new_boxes.append(predicted_boxes[cur_idx])
            new_scores.append(predicted_scores[cur_idx])
            current_ids.append(prev_ids[prev_idx])
        
        # add the current predictions that not overlap with the prediction that matched with prev and produce new id
        for idx, boxes in current_map[label]:
            if idx not in idx_predicted_with_match:
                is_match = False
                for cur_idx, prev_idx in matched_pairs:
                    if calc_intersection_over_union(boxes, predicted_boxes[cur_idx]) > min_score:
                        is_match = True
                if not is_match:
                    new_labels.append(label)
                    new_boxes.append(boxes)
                    new_scores.append(predicted_scores[idx])
                    current_ids.append(next_ids[label])
                    next_ids[label] += 1

    predicted_labels = new_labels
    predicted_boxes = new_boxes
    predicted_scores = new_scores
    
    return current_ids, next_ids, (predicted_labels, predicted_boxes, predicted_scores)


def get_direction_analysis_smooth(axis, current_frame_idx, xy_list, smooth_info, window_threshold=30, smooth_tr=10, range_length_threshold=15, range_dist_threshold=50, zero_threshold=7, body_directions_list=None, is_static_camera=False):
    """
    Calculate the surfing maneuvers direction 
    """
    current_val = xy_list[-1]
    prev_val = xy_list[-2]
    
    minus_label = 'left' if axis == 'x' else 'up'
    plus_label = 'right' if axis == 'x' else 'down'
    
    if prev_val == -1 or current_val == -1:
        return smooth_info
    
    directions_list = smooth_info[f'directions_list_{axis}_axis']
    smooth_directions_list = smooth_info[f'smooth_directions_list_{axis}_axis']
    total_smooth_left_ranges = smooth_info[f'total_smooth_{minus_label}_ranges'] 
    total_smooth_right_ranges = smooth_info[f'total_smooth_{plus_label}_ranges']
    current_smooth_left_start_idx = smooth_info[f'current_smooth_{minus_label}_start_idx']
    current_smooth_right_start_idx = smooth_info[f'current_smooth_{plus_label}_start_idx']
    current_smooth_zero_start_idx = smooth_info[f'current_smooth_zero_start_idx_{axis}_axis']
    current_smooth_zero_start_idx_in_directions_list = smooth_info[f'current_smooth_zero_start_idx_{axis}_axis_in_directions_list']

    surfer_list_length = len(directions_list)
    
    if current_val - prev_val > 0: # extend the directions list
        directions_list.append(1)
    else:
        directions_list.append(-1)

    if surfer_list_length > window_threshold: # verify that we will not get index error
        
        if sum(directions_list[-window_threshold:]) > smooth_tr:  # extend the smooth directions list
            smooth_directions_list.append(1)
        elif sum(directions_list[-window_threshold:]) < -smooth_tr:
            smooth_directions_list.append(-1)
        else:
            smooth_directions_list.append(0)
    else:
        smooth_info[f'directions_list_{axis}_axis'] = directions_list
        smooth_info[f'smooth_directions_list_{axis}_axis'] = smooth_directions_list
        return smooth_info
    
    if len(smooth_directions_list) < 2:
        smooth_info[f'directions_list_{axis}_axis'] = directions_list
        smooth_info[f'smooth_directions_list_{axis}_axis'] = smooth_directions_list
        return smooth_info

    elif smooth_directions_list[-2] == 0:
        
        if smooth_directions_list[-1] == 0: # extending the zero's list
            smooth_info[f'directions_list_{axis}_axis'] = directions_list
            smooth_info[f'smooth_directions_list_{axis}_axis'] = smooth_directions_list

            return smooth_info
        
        elif current_frame_idx - current_smooth_zero_start_idx > zero_threshold: # start new count and check if the last direction list is big anough to add the new range 
            
            if smooth_directions_list[current_smooth_zero_start_idx_in_directions_list -1] == 1:
                
                if current_smooth_zero_start_idx - current_smooth_right_start_idx > range_length_threshold: # check frames condition
                    
                    if abs(xy_list[current_smooth_zero_start_idx] - xy_list[current_smooth_right_start_idx]) > range_dist_threshold: # check distance condition
                        
                        if body_directions_list is not None and not is_static_camera:
                            right_percent = body_directions_list[current_smooth_right_start_idx:current_smooth_zero_start_idx].count('right')/(current_smooth_zero_start_idx - current_smooth_right_start_idx)
                            direction_condition = True if right_percent > right_percent_threshold else False
                        else:
                            direction_condition = True
                            
                        if direction_condition:
                            total_smooth_right_ranges.append((current_smooth_right_start_idx, current_smooth_zero_start_idx -1))
                            smooth_info[f'total_{plus_label}_count'] += 1
                        else:
                            pass # there is no body position for that side
                    else:

                        pass # not enogh distance to calculate as turning
                else:

                    pass # not enogh frames to calculate as turning
                    
                current_smooth_right_start_idx = current_frame_idx

            elif smooth_directions_list[current_smooth_zero_start_idx_in_directions_list -1] == -1:

                if current_smooth_zero_start_idx - current_smooth_left_start_idx > range_length_threshold:

                    if abs(xy_list[current_smooth_zero_start_idx] - xy_list[current_smooth_left_start_idx]) > range_dist_threshold:
                        
                        if body_directions_list is not None and not is_static_camera:
                            left_percent = body_directions_list[current_smooth_left_start_idx:current_smooth_zero_start_idx].count('left')/(current_smooth_zero_start_idx - current_smooth_left_start_idx)
                            direction_condition = True if left_percent > left_percent_threshold else False
                        else:
                            direction_condition = True
                            
                        if direction_condition:

                            total_smooth_left_ranges.append((current_smooth_left_start_idx, current_smooth_zero_start_idx -1))
                            smooth_info[f'total_{minus_label}_count'] += 1
                        else:
                            pass # there is no body position for that side
                    else:

                        pass # not enogh distance to calculate as turning
                else: 
                    pass

                current_smooth_left_start_idx = current_frame_idx

    elif smooth_directions_list[-1] == 0:
        
        current_smooth_zero_start_idx = current_frame_idx
        current_smooth_zero_start_idx_in_directions_list = len(smooth_directions_list) -1

    elif smooth_directions_list[-1] == -1 and current_smooth_left_start_idx == -1:
        current_smooth_left_start_idx = current_frame_idx - window_threshold
        
    elif smooth_directions_list[-1] == 1 and current_smooth_right_start_idx == -1:
        current_smooth_right_start_idx = current_frame_idx - window_threshold
    
    else:
        pass

    smooth_info[f'directions_list_{axis}_axis'] = directions_list
    smooth_info[f'smooth_directions_list_{axis}_axis'] = smooth_directions_list
    smooth_info[f'total_smooth_{minus_label}_ranges'] =  total_smooth_left_ranges
    smooth_info[f'total_smooth_{plus_label}_ranges']  = total_smooth_right_ranges
    smooth_info[f'current_smooth_{minus_label}_start_idx'] =  current_smooth_left_start_idx
    smooth_info[f'current_smooth_{plus_label}_start_idx']  = current_smooth_right_start_idx
    smooth_info[f'current_smooth_zero_start_idx_{axis}_axis']  = current_smooth_zero_start_idx
    smooth_info[f'current_smooth_zero_start_idx_{axis}_axis_in_directions_list']  = current_smooth_zero_start_idx_in_directions_list

    return smooth_info


def predict_direction(image_ar, direction_model):
    """
    predict the surfer pose direction left or right
    """
    # convert image to numpy array
    images = img_to_array(image_ar)
    # expand dimension of image
    images = np.expand_dims(images, axis=0)
    # making prediction with model
    prediction = direction_model.predict(images)

    #del direction_model
    return 'left' if prediction < surfer_direction_left_or_right_threshold else 'right'


def is_fall_func(current_frame, video_length, surfer_y_max, wave_y_min, white_level, is_fall):
    """
    Check if the surfer fall/not
    """
    if is_fall != False:
        is_fall = True
        if surfer_y_max != -1: # surfer detected
            if white_level > white_level_threshold:
                is_fall = False
            elif surfer_y_max - wave_y_min < 20: # surfer above the wave 
                is_fall = False
    return is_fall



def is_excercise_func(wave_min_y, surfer_max_y, excercise_size, current_frame, last_ex_frame, ex_threshold, excercises_lengths, excercises_count, current_ex_start_frame):
    """
    Check if the surfer performed excercise 
    Calculated by checking if buttom surfer box is above the top of the wave box
    (means if the surfer is cross the top of the wave)
    """
    if surfer_max_y != -1:

        # if buttom surfer box is above the top of the wave box
        if surfer_max_y < wave_min_y:
            
            if last_ex_frame != -1: # extend the existing list
                
                if current_frame - last_ex_frame <= exercise_frames_window:
                    excercise_size += current_frame - last_ex_frame # include all the times that the sufer wasn't seen
                    
                else:
                    excercise_size = 1
                    current_ex_start_frame = current_frame
                    
            else:  # create new list
                excercise_size = 1
                current_ex_start_frame = current_frame
            
            last_ex_frame = current_frame

        else:
            if current_frame - last_ex_frame > exercise_frames_window: # we define if it an exercise in the moment that it can't be extended
                
                if excercise_size > ex_threshold:
                    excercises_count += 1
                    excercises_lengths.append((current_ex_start_frame, last_ex_frame))
                    
                excercise_size = 0 
                last_ex_frame = -1
                    
    return excercise_size, last_ex_frame, excercises_lengths, excercises_count, current_ex_start_frame


def is_pipe_func(surfer_seen, pipe_size, current_frame, last_pipe_frame, pipe_threshold, pipe_threshold2, pipes_lengths, pipes_lengths2, pipes_count, pipes_count2, current_pipe_start_frame):
    """
    Check if the surfer entered into a pipe (disappeared for the video during x frames)
    """
    if not surfer_seen:

    # if buttom surfer box is above the top of the wave box

        if last_pipe_frame != -1: # extend the existing list

            if current_frame - last_pipe_frame <= pipe_frames_window:
                pipe_size += current_frame - last_pipe_frame # include all the times that the sufer wasn't seen

            else:
                pipe_size = 1
                current_pipe_start_frame = current_frame

        else:  # create new list
            pipe_size = 1
            current_pipe_start_frame = current_frame

        last_pipe_frame = current_frame
    else:
        
        if current_frame - last_pipe_frame > pipe_frames_window: # we define if it as pipe in the moment that it can't be extended

            if pipe_size > pipe_threshold:
                pipes_count += 1
                pipes_lengths.append((current_pipe_start_frame, last_pipe_frame))
            elif pipe_size > pipe_threshold2:
                pipes_count2 += 1
                pipes_lengths2.append((current_pipe_start_frame, last_pipe_frame))

            pipe_size = 0 
            last_pipe_frame = -1
              
    return pipe_size, last_pipe_frame, pipes_lengths, pipes_lengths2, pipes_count, pipes_count2, current_pipe_start_frame

def get_direction_analysis(direction, current_frame, direction1_obj, direction2_obj, turning_threshold, other_direction_count, other_direction_lengths):
    """
    Check if the surfer turning for the given direction during x frames 
    """
    if direction1_obj['start_frame'] == -1: # start new one

        direction1_obj = {'start_frame': current_frame, 'last_frame': current_frame}

    elif current_frame - direction1_obj['last_frame'] < direction_threshold: # update the end

        direction1_obj['last_frame'] = current_frame

    else: # start new one
        direction1_obj = {'start_frame': current_frame, 'last_frame': current_frame}


    # update only when we can't extand the list from previuos list
    if direction2_obj['start_frame'] != -1 and current_frame - direction2_obj['last_frame'] > direction_threshold:

        if direction2_obj['last_frame'] - direction2_obj['start_frame'] > direction_window: # count it
            other_direction_count += 1
            other_direction_lengths.append((direction2_obj['start_frame'], direction2_obj['last_frame']))

        direction2_obj = {'start_frame': -1, 'last_frame': -1}
            
    return direction1_obj, direction2_obj, other_direction_count, other_direction_lengths


def is_turning_func(direction, current_frame, turning_threshold, left_lengths, right_lengths, left_count, right_count, left_obj, right_obj):
    """
    Check if the surfer turning for sone direction during x frames 
    """
    if direction == 'left':
        left_obj, right_obj, right_count, right_lengths =  get_direction_analysis(direction, current_frame, left_obj, right_obj, turning_threshold, right_count, right_lengths)
    else:
        right_obj, left_obj, left_count, left_lengths =  get_direction_analysis(direction, current_frame, right_obj, left_obj, turning_threshold, left_count, left_lengths)
         
    return left_obj, left_count, right_lengths, right_obj, right_count, left_lengths


# In[9]:

def surfer_perform_trick(surfer_xy, excercise_size, current_frame, last_ex_frame, ex_threshold, excercises_lengths, excercises_count, current_ex_start_frame):
    """
    Check if the surfer performed excercise
    Calculated by checking if buttom surfer box is above the top of the wave box
    (means if the surfer is cross the top of the wave)
    """
    width = surfer_xy[-1][0]
    height = surfer_xy[-1][1]
    if width != -1:

        # if buttom surfer box is above the top of the wave box
        if height - width < 30:

            if last_ex_frame != -1:  # extend the existing list

                if current_frame - last_ex_frame <= exercise_frames_window:
                    excercise_size += current_frame - last_ex_frame # include all the times that the sufer wasn't seen

                else:
                    excercise_size = 1
                    current_ex_start_frame = current_frame

            else:  # create new list
                excercise_size = 1
                current_ex_start_frame = current_frame

            last_ex_frame = current_frame

        else:
            if current_frame - last_ex_frame > exercise_frames_window: # we define if it an exercise in the moment that it can't be extended

                if excercise_size > ex_threshold:
                    excercises_count += 1
                    excercises_lengths.append((current_ex_start_frame, last_ex_frame))

                excercise_size = 0
                last_ex_frame = -1

    return excercise_size, last_ex_frame, excercises_lengths, excercises_count, current_ex_start_frame, height - width




def detect_video(input_file, output_file, labels_names, fps=100, score_filter=0.6, count_pipe_threshold = 18, count_disapear_threshold=11, count_turning_threshold=10, ex_threshold=3, predictions_dict='', to_export_video=True, is_static_camera=False, detection_model=None, direction_model=None):
    """Takes in a video and produces an output video with object detection
    run on it (i.e. displays boxes around detected objects in real-time).
    Output videos should have the .avi file extension. Note: some apps,
    such as macOS's QuickTime Player, have difficulty viewing these
    output videos. It's recommended that you download and use
    `VLC <https://www.videolan.org/vlc/index.html>`_ if this occurs.


    :param model: The trained model with which to run object detection.
    :type model: detecto.core.Model
    :param input_file: The path to the input video.
    :type input_file: str
    :param output_file: The name of the output file. Should have a .avi
        file extension.
    :type output_file: str
    :param fps: (Optional) Frames per second of the output video.
        Defaults to 30.
    :type fps: int
    :param score_filter: (Optional) Minimum score required to show a
        prediction. Defaults to 0.6.
    :type score_filter: float

    **Example**::

        >>> from detecto.core import Model
        >>> from detecto.visualize import detect_video

        >>> model = Model.load('model_weights.pth', ['tick', 'gate'])
        >>> detect_video(model, 'input_vid.mp4', 'output_vid.avi', score_filter=0.7)
    """

    # Read in the video

    video = cv2.VideoCapture(input_file)

    if int((cv2.__version__).split('.')[0]) < 3:
        video_length = int(video.get(cv2.cv.CAP_PROP_FRAME_COUNT))
    else:
        video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video frame dimensions
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale down frames when passing into model for faster speeds
    scaled_size = 800
    scale_down_factor = min(frame_height, frame_width) / scaled_size

    # The VideoWriter with which we'll write our video with the boxes and labels
    # Parameters: filename, fourcc, fps, frame_size
    if to_export_video:
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))

    # Transform to apply on individual frames of the video
    transform_frame = transforms.Compose([  # TODO Issue #16
        transforms.ToPILImage(),
        transforms.Resize(scaled_size),
        transforms.ToTensor(),
        normalize_transform(),
    ])

    
    xy_per_label = defaultdict(list)
    
    first_time = True
    wave_size_norm = list()
    
    surfer_count = surfer_count2 = 0 
    count_disapear = 0
    first_time_surfer = False
    pipes_count = pipes_count2 = pipe_size = pipe_size2 = 0 
    excercise_size = excercises_count = 0
    last_ex_frame = -1
    last_box_shape_frame = -1
    current_frame = last_ex_count = 0
    white_levels = list()
    white_level = 0
    is_fall = ''
    
    first_seen_surfer_idx = -1
    pipes_count= pipes_count2= up= down= left= right= wave_size= excercises_count= excercise_size= surfer_box_shape_count = surfer_box_shape_size = 0
    current_direction = ''
    excercises_lengths = list()
    surfer_box_shape_lengths = list()
    pipes_lengths = list()
    pipes_lengths2 = list()
    
    last_pipe_frame = -1
    
    left_count = 0
    right_lengths = list()
    right_count = 0
    left_lengths = list()
    left_obj = {'start_frame': -1, 'last_frame': -1}
    right_obj = {'start_frame': -1, 'last_frame': -1}
    current_ex_start_frame = current_pipe_start_frame = current_surfer_box_start_frame = 0
    
    directions_list=list()
    smooth_directions_list=list()
    smooth_left = smooth_right = 0
    
    directions_ups_list=list()
    smooth_directions_ups_list=list()
    smooth_left = smooth_right = smooth_up = smooth_down = 0   
                
    smooth_info = dict()
    smooth_info['directions_list_x_axis'] = list()
    smooth_info['smooth_directions_list_x_axis'] = list()
    smooth_info['total_smooth_left_ranges'] =  list()
    smooth_info['total_smooth_right_ranges']  = list()
    smooth_info['current_smooth_left_start_idx'] =  -1
    smooth_info['current_smooth_right_start_idx']  = -1
    smooth_info['current_smooth_zero_start_idx_x_axis']  = -1
    smooth_info[f'total_right_count'] = 0
    smooth_info[f'total_left_count'] = 0
    smooth_info[f'current_smooth_zero_start_idx_x_axis_in_directions_list']  = -1

    smooth_info['directions_list_y_axis'] = list()
    smooth_info['smooth_directions_list_y_axis'] = list()
    smooth_info['total_smooth_up_ranges'] = list()
    smooth_info['total_smooth_down_ranges'] = list()
    smooth_info['current_smooth_up_start_idx'] = -1
    smooth_info['current_smooth_down_start_idx'] = -1
    smooth_info['current_smooth_zero_start_idx_y_axis'] = -1
    smooth_info[f'total_down_count'] = 0
    smooth_info[f'total_up_count'] = 0
    smooth_info[f'current_smooth_zero_start_idx_y_axis_in_directions_list']  = -1
    direction = ''
    body_directions_list = list()
    
    surfing_direction = ''

    surfer_xy = list()
    height_width = 0
    while True:

        current_frame += 1
        ret, frame = video.read()

        # Stop the loop when we're done with the video
        if not ret:
            break
        
        # The transformed frame is what we'll feed into our model
        # transformed_frame = transform_frame(frame)
        transformed_frame = frame  
        if not predictions_dict:
            predictions = detection_model.predict(transformed_frame)
        else:
            predictions = predictions_dict[current_frame - 1]
        print(predictions[0])
                
        for i, predicted_label in enumerate(predictions[0]): # correction for dedicated pools 
            if predicted_label == surfer_dict[camera_setup]:
                predictions[0][i] = surfer_dict[camera_setup]
                surfer_boxes = predictions[1][i]
        
        # extract the surfer boxes
        max_iou_wave_surfer = get_surfer_insied_wave(predictions[0], predictions[1])
        if type(max_iou_wave_surfer) != type(False):
            surfer_boxes = predictions[1][max_iou_wave_surfer['surfer_idx']]
        
        # initialize the ids for tracking 
        if first_time:
            current_ids, next_ids = get_first_labels(predictions[0], labels_names)
            
            if not is_wave_contains_surfer(predictions[0], predictions[1]):
                for pure_label in ['wave', surfer_dict[camera_setup]]:
                    xy_per_label[pure_label].append((-1,-1)) 
                continue
            id_wave = -1

        else:
            # traking and selecting the best object detections
            current_ids, next_ids, predictions = predict_and_compare_with_labels(prev_res, predictions, next_ids, labels_names, score_filter)
            
        # Add the top prediction of each class to the frame
        wave_size = round(np.mean(wave_size_norm),2) if len(wave_size_norm) < 7 else round(np.mean(wave_size_norm[len(wave_size_norm)-6:]),2)
        
        i = 0
        surfboard = wave = is_wave = surfing_person = False
        for label, box, score in zip(*predictions): # go over the predictions, draw the boxes and labels info
            
            if type(score) not in [float]:
                score = score.item()

            if score < score_filter:
                i+=1
                continue
            score = round(score, 2)
            label_info = ''
            if label == 'wave':
                    
                if first_time or (not is_wave and id_wave != -1): #calc_intersection_over_union(prev_res[1][id_wave], box)> 0.6):
                    
                    
                    white_level = calculate_wave_wight_level(box, frame)
                    label_info = 'Wave' if 'wave' in label else label
                    label_info = f'{label_info}'
                    white_levels.append(white_level)
                    id_wave = i
                    is_wave = True
                    xy_per_label[label].append((box[1], box[3]))
                    wave = True
                    wave_hight = abs(box[3] - box[1])
                    
                    
            elif label == surfer_dict[camera_setup] and not surfing_person:
                if True: #type(max_iou_wave_surfer) != type(False):
                    
                    label_info = f'Surfer' if surfer_dict[camera_setup] in label else label
                    pure_label = surfer_dict[camera_setup]
                    xy_per_label[pure_label].append((box[0] + (box[2]- box[0])/2, box[3], box[1]))
                    surfer_xy.append((box[2] - box[0], box[3]- box[1]))
                    surfing_person = True
                    surfing_person_hight = abs(box[3] - box[1])

                    if first_seen_surfer_idx == -1:
                        first_seen_surfer_idx = current_frame
                    
                    direction = predict_surfer_direction(box, frame, direction_model)
                
            elif label == 'surfboard' and not surfboard:
                pure_label = 'surfboard'
                label_info = 'Surfboard'
                xy_per_label[pure_label].append((box[0] + (box[2]- box[0])/2, box[1] + (box[3] - box[1])/2))  
                surfboard = True

            else:
                i+=1
                continue
            
            label_info = f'{label_info}_{current_ids[i]}_{score}'
            draw_rect(label_info, frame, box, cv2, to_export_video=to_export_video)
            i+=1
            
        direction = direction if surfing_person else ''
        body_directions_list.append(direction)
        if current_frame == starting_direction_frame_count:
            
            right_percent = body_directions_list.count('right')/len(body_directions_list)
            surfing_direction = 'right' if right_percent > surfer_direction_left_or_right_threshold else 'left'
        
        # Draw the analysis information on the given frame 
        put_analysis(frame, pipes_count, pipes_count2, up, down, left, right, wave_size, excercises_count, excercise_size, surfer_box_shape_count, is_fall, f'{current_frame}/{video_length}', direction, left_count, right_count, to_export_video, smooth_left, smooth_right, smooth_up, white_level, surfing_direction, is_static_camera, height_width)
        
        
        if surfing_person and wave: # calculate wave size
            
            wave_size = wave_hight/surfing_person_hight
            wave_size = wave_size.data if torch.is_tensor(wave_size) else wave_size
            wave_size_norm.append(float(wave_size))
        
        if surfing_person != True: # complete the surfer predicted locations list
            pure_label = surfer_dict[camera_setup]
            xy_per_label[pure_label].append((-1,-1, -1))
            surfer_xy.append((-1, -1))

        else:
            surfer_count += 1
        if wave != True: # complete the wave predicted locations list
            pure_label = 'wave'
            xy_per_label[pure_label].append((-1,-1)) 
            white_levels.append(np.mean(white_levels))
        if surfboard != True: # complete the surfboard predicted locations list
            pure_label = 'surfboard'
            xy_per_label[pure_label].append((-1,-1)) 
        
        ################################################# dynamic analysis####################################################
        if surfer_count >= 2 or first_time_surfer:
            
            lst = [i[1] for i in xy_per_label[surfer_dict[camera_setup]]] # y_max of the surfer
            lst2 = [i[0] for i in xy_per_label[surfer_dict[camera_setup]]] # x_center of the surfer
            surfer_lst_len = len(xy_per_label[surfer_dict[camera_setup]])
            
            if not surfing_person:
                count_disapear += 1

            else:
                
                smooth_info = get_direction_analysis_smooth('x', current_frame, lst2, smooth_info, window_threshold=window_threshold_direction_analysis, smooth_tr=smooth_tr_direction_analysis, range_length_threshold=range_length_threshold_direction_analysis, range_dist_threshold=range_dist_threshold_direction_analysis, body_directions_list=body_directions_list, is_static_camera=is_static_camera)
                smooth_info = get_direction_analysis_smooth('y', current_frame, lst, smooth_info, window_threshold=window_threshold_direction_analysis, smooth_tr=smooth_tr_direction_analysis, range_length_threshold=range_length_threshold_direction_analysis, range_dist_threshold=range_dist_threshold_direction_analysis, is_static_camera=is_static_camera)
                smooth_down = smooth_info['total_down_count']
                smooth_up = smooth_info['total_up_count']
                smooth_left = smooth_info['total_left_count']
                smooth_right = smooth_info['total_right_count']

                surfer_box_shape_size, last_box_shape_frame, surfer_box_shape_lengths, surfer_box_shape_count, current_surfer_box_start_frame, height_width = surfer_perform_trick(surfer_xy, surfer_box_shape_size, current_frame, last_box_shape_frame, ex_threshold=5, excercises_lengths=surfer_box_shape_lengths, excercises_count=surfer_box_shape_count, current_ex_start_frame=current_surfer_box_start_frame)
                
                count_disapear = 0
                
                last_seen_surfer = surfer_lst_len -1
                last_seen_surfer2 = surfer_lst_len -1
            excercise_size, last_ex_frame, excercises_lengths, excercises_count, current_ex_start_frame = is_excercise_func(xy_per_label['wave'][len(xy_per_label['wave']) -1][0], lst[surfer_lst_len -1], excercise_size, current_frame, last_ex_frame, ex_threshold, excercises_lengths, excercises_count, current_ex_start_frame)

            pipe_size, last_pipe_frame, pipes_lengths, pipes_lengths2, pipes_count, pipes_count2, current_pipe_start_frame = is_pipe_func(surfing_person, pipe_size, current_frame, last_pipe_frame, count_pipe_threshold, count_disapear_threshold, pipes_lengths, pipes_lengths2, pipes_count, pipes_count2, current_pipe_start_frame)

        
        ################################################# end ########################################################
        
        prev_res = (predictions[0], predictions[1], predictions[2], current_ids)
        first_time = False
        
        if current_frame > video_length-is_fall_frames_window:
            is_fall = is_fall_func(current_frame, video_length, xy_per_label[surfer_dict[camera_setup]][len(xy_per_label[surfer_dict[camera_setup]]) -1][1], xy_per_label['wave'][len(xy_per_label['wave']) -1][0], white_level, is_fall )
        
        # Write this frame to our video file
        if to_export_video:
            out.write(frame)

        # If the 'q' key is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    
    directions_lengths = dict()
    directions_lengths['left'] = smooth_info['total_smooth_left_ranges']
    directions_lengths['right'] = smooth_info['total_smooth_right_ranges']
    directions_lengths['up'] = smooth_info['total_smooth_up_ranges']
    directions_lengths['down'] = smooth_info['total_smooth_down_ranges']


    # When finished, release the video capture and writer objects
    if to_export_video:
        out.release()
        video.release()

        # Close all the frames
        cv2.destroyAllWindows()
    return xy_per_label, round(np.mean(wave_size_norm),4), white_levels, is_fall, current_frame, excercises_lengths, pipes_lengths, pipes_lengths2, directions_lengths, surfing_direction, surfer_box_shape_lengths


# load detection and directions models and known object predictions
def get_prediction_dict(load_predictions_file):
    with open(load_predictions_file, 'rb') as f:
        predictions_dict = pickle.load(f)
    return predictions_dict


def run_surfing_analysis(source_dir, target_dir,  predictions_file='', to_export_video=True, is_static_camera=False): 
    '''
    Perform surfing analysis on all the videos of the source dir.
    The surfing analysis includes: 
        1. Surfing to the right/left/up
        2. Surfer fall/not
        3. Surfer enter into a pipe 
        4. Surfer perform exercise
        5. Wave breaking level
        6. Wave size (normelized)
    The surfing analysis is printing on each frame and updated dynamicly
    Save the videos with the analysis
    Save the predicted analysis in a file 
    ''' 
    if predictions_file:
        predictions_dict = get_prediction_dict(predictions_file)
        detection_model = None
        print('Predictions_dict loaded')
    else:
        predictions_dict = None

    detection_model = core.Model.load(object_detection_model_path, labels)
    direction_model = load_model(object_direction_model_path)

    movdir = source_dir
    basedir = target_dir
    # Walk through all files in the directory that contains the files to copy
    i = 0 
    files_count = 0
    ranges = dict()
    analysis = dict()
    
    for root, dirs, files in os.walk(movdir):

        for filename in files:
            
            if files_count % gc_files_threshold == 0:
                collected = gc.collect()
                time.sleep(sleep_time)
            files_count += 1
            file = f'{root}/{filename}'

            #if True:

            try:
                if not is_static_camera:
                    label = filename.split('_')[1]
                    if label not in ['n', 'y']:
                        continue
                    exercise = filename.split('_')[0]
                    if exercise not in ['n', 'y', 'y2', 'y3', 'y4', 'y5']:
                        continue

                    if predictions_dict:

                        single_video_predictions_dict = predictions_dict[file]
                    else:
                        single_video_predictions_dict = None

                    score = float(f'{filename.split("_")[-1].split(".")[0]}.{filename.split("_")[-1].split(".")[1]}')

                    
                    export_file = f'{target_dir}/{filename}_tracking.avi'
                    print(export_file)
                    i = 0

                    if to_export_video:
                        while(True):
                            if os.path.exists(export_file):
                                export_file = f'{target_dir}/{filename}_tracking{i}.avi'
                                i += 1
                            else:
                                break
                else:
                    label = exercise = 'f'
                    score = 0
                    i+=1
                    export_file = f'{target_dir}/{filename}_tracking{i}.avi'
                    single_video_predictions_dict = None


                xy_per_label, normal_factor, white_levels, is_fall, frames_count, excercises_lengths, pipe_lengths, pipe_lengths2, directions_ranges, surfing_direction, surfer_box_shape_lengths = detect_video(file, export_file, fps=50, score_filter=0.6, labels_names=labels, predictions_dict=single_video_predictions_dict, to_export_video=to_export_video, is_static_camera=is_static_camera, detection_model=detection_model, direction_model=direction_model)

                if to_export_video:
                    os.rename(export_file, f'{target_dir}/{filename}_tracking{i}_{is_fall}.avi')


                ranges[file] = dict()
                ranges[file]['export_file'] = export_file
                ranges[file]['left'] = directions_ranges['left']
                ranges[file]['right'] = directions_ranges['right']
                ranges[file]['up'] = directions_ranges['up']
                ranges[file]['down'] = directions_ranges['down']
                ranges[file]['pipes'] = pipe_lengths
                ranges[file]['pipes2'] = pipe_lengths2
                ranges[file]['excercises'] = excercises_lengths
                ranges[file]['box_shape'] = surfer_box_shape_lengths

                ranges[file]['score'] = score
                ranges[file]['actual_is_fall'] = label
                ranges[file]['pred_is_fall'] = is_fall
                ranges[file]['normal_factor'] = normal_factor
                ranges[file]['frames_count'] = frames_count
                ranges[file]['surfing_direction'] = surfing_direction
                print('ranges: ', ranges[file])

            except Exception as e:
                print(e, file)

    return analysis, ranges

def run_all():
    
    is_static_camera = False if camera_setup == 'dynamic_camera' else True
    analysis, ranges = run_surfing_analysis(videos_source_path, videos_res_path, None, to_export_video, is_static_camera)
    with open(predicted_ranges_path, 'wb') as f:
        pickle.dump(ranges, f)


def run_surfing_events_detection():

    run_all()

    with open(predicted_ranges_path, 'rb') as f:
        ranges = pickle.load(f)

    analysis_features = dict()
    for video_path, video_events_ranges in ranges.items():

        analysis_features[video_path] = dict()
        for event_name, event_info in video_events_ranges.items():

            if type(event_info) == list:
                analysis_features[video_path][event_name] = len(event_info)
            else:
                analysis_features[video_path][event_name] = event_info

    analysis_features_df = pd.DataFrame.from_dict(analysis_features, orient='index')

    file_name = os.path.join(surfing_analysis_predicted_info, f'surfing_analysis_features.csv')

    analysis_features_df.to_csv(file_name, sep=',', encoding='utf-8')

    print('analysis_features:\n', analysis_features)

    #ranges_report = run_evaluation()
    #fall_report = run_fall_or_not_evaluation()


    #return f'\n{ranges_report}\n{fall_report}\n'


if __name__ == '__main__':

    run_surfing_events_detection()