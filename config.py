
model_config = {
'IMG_SIZE' : 512,
'BATCH_SIZE' : 16,
'EPOCHS' : 1000,

'MAX_SEQ_LENGTH' : 30,
'NUM_FEATURES' : 2048,
'TRAIN_SPLIT': 0.85,
'feature_extractor_weights': 'imagenet',
'pooling': 'avg',
'train_path' : r'datasets\videos\dynamic_camera\train',
'test_path' : r'datasets\videos\dynamic_camera\train'
}


util_config = {

    'detected_events_file': 'results\surfing_analysis\{surfing_analysis_config["camera_setup"]}\predicted_ranges\predicted_ranges.pkl',
    'actual_events_file': 'datasets\events_ranges\dynamic_camera_actual_dict.pkl',
    'export_dir': r'datasets\videos\dedicated_pools',
    'exercises_list2': ['box_shape'],
    'exercises_list': ['left', 'right', 'up', 'pipes', 'pipes2', 'excercises', 'box_shape'],
    'videos_location': r'C:\Users\User\Desktop\SCOOL\SCOOL\thesis\data-sets\videos\dedicated_pools'

}

surfing_analysis_config = {
    'camera_setup' : 'dedicated_pools',
    'to_export_video': True,
    'surfer_direction_left_or_right_threshold': 0.5,
    'white_level_threshold': 0.75,
    'is_fall_frames_window': 50,
    'surfer_above_wave_threshold': 20,
    'exercise_frames_window': 10,
    'pipe_frames_window': 10,
    'direction_threshold': 5,
    'direction_window': 15,
    'starting_direction_frame_count':10,
    'range_dist_threshold_direction_analysis': 50,
    'range_length_threshold_direction_analysis': 15,
    'smooth_tr_direction_analysis': 10,
    'window_threshold_direction_analysis': 30,
    'right_percent_threshold': 0.3,
    'left_percent_threshold': 0.3,
    'sleep_time': 100,
    'gc_files_threshold': 30,
    'check_is_fall_frame': 15
}

surfer_direction_config = {

    'img_size': (64, 64)

}