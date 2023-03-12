import os
import pickle
import sys

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
            print('here')
            count += 1
    return all_ranges


def main():

    cwd = os.getcwd()
    sys.path.insert(0, cwd)
    all_ranges = merge_ranges_files(os.path.join(cwd, util_config['detected_events_file']), os.path.join(cwd, util_config['actual_events_file']))
    print(len(all_ranges))


if __name__ == '__main__':
    main()