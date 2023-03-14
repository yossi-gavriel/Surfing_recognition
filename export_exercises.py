import os
import pickle
import sys
from collections import defaultdict
import shutil
import cv2

from config import util_config


def merge_ranges_files(actual_ranges_file, predicted_ranges_file):


    all_ranges = dict()
    count = 0
    with open(actual_ranges_file, 'rb') as f:
        actual_ranges = pickle.load(f)

    with open(predicted_ranges_file, 'rb') as f:
        predicted_ranges = pickle.load(f)

    for actual_range_key, actual_range_val in actual_ranges.items():

        video_name = actual_range_key.split('/')[-1]
        all_ranges[video_name] = actual_range_val

    for predicted_range_key, predicted_range_val in predicted_ranges.items():

        if predicted_range_key not in all_ranges:
            video_name = predicted_range_key
            all_ranges[video_name] = predicted_range_val
        else:
            count += 1
    return all_ranges

def analyze_all_ranges(all_ranges):

    analysis = defaultdict(list)
    for key, vals in all_ranges.items():

        for val, val_list in vals.items():

            if val in util_config['exercises_list']:
                analysis[val].extend(val_list)

    print([(val, len(lst)) for val, lst in analysis.items()])

def init_dirs(dir_path, dirs_list):

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    for new_dir_name in dirs_list:
        out_dir = os.path.join(dir_path, new_dir_name)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)


def export_all_ranges(cwd, all_ranges, export_dir, fps=50):

    init_dirs(os.path.join(cwd, export_dir), util_config['exercises_list'])
    analysis = defaultdict(list)
    new_events_counts = {key:0 for key in util_config['exercises_list']}

    for video_name, video_vals in all_ranges.items():
        video_name = video_name.split('/')[-1]
        video_path = os.path.join(util_config['videos_location'], video_name)
        for key, key_list in video_vals.items():

            if key in util_config['exercises_list']:
                # read video
                video = cv2.VideoCapture(video_path)
                frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                current_frame = 0
                for ex_start, ex_end in key_list:

                    output_file = os.path.join(util_config['export_dir'], key, str(new_events_counts[key])+ video_name)

                    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))

                    while current_frame < ex_end:

                        current_frame += 1
                        ret, frame = video.read()
                        if current_frame > ex_start:
                            out.write(frame)

                    out.release()

                    new_events_counts[key] += 1
                    print('done: ', output_file)

                video.release()



    print([(val, len(lst)) for val, lst in analysis.items()])

def export():

    cwd = os.getcwd()
    sys.path.insert(0, cwd)
    with open(util_config['detected_events_file'], 'rb') as f:
        all_ranges = pickle.load(f)
    #all_ranges = merge_ranges_files(os.path.join(cwd, util_config['detected_events_file']), os.path.join(cwd, util_config['actual_events_file']))
    analyze_all_ranges(all_ranges)
    export_all_ranges(cwd, all_ranges, util_config['export_dir'])


if __name__ == '__main__':
    export()