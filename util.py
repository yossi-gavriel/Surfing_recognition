import os
import pickle
import sys
from collections import defaultdict

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

def main():

    cwd = os.getcwd()
    sys.path.insert(0, cwd)
    all_ranges = merge_ranges_files(os.path.join(cwd, util_config['detected_events_file']), os.path.join(cwd, util_config['actual_events_file']))
    analyze_all_ranges(all_ranges)
    print(len(all_ranges))


if __name__ == '__main__':
    main()