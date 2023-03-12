
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

    'detected_events_file': 'datasets\events_ranges\predicted_ranges.pkl',
    'actual_events_file': 'datasets\events_ranges\dynamic_camera_actual_dict.pkl',
    'export_dir': r'datasets\videos\dynamic_camera',
    'exercises_list': ['left', 'right', 'up', 'pipes', 'pipes2', 'excercises'],
    'videos_location': r'C:\Users\User\Desktop\SCOOL\SCOOL\thesis\data-sets\videos\dynamic_camera'

}