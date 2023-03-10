from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
from config import model_config



train_directory = os.fsencode(model_config['train_path'])
test_directory = os.fsencode(model_config['test_path'])

def get_data(directory):

    files = []
    tags = []

    for file in os.listdir(directory):
        # file = str(file[1:])
        file = str(file).split("b")[1].replace("'", "")

        try:
            #tags.append(0 if file.split('_')[1].split('.')[0] ==  'n' else 1)
            tags.append(file.split('_')[1].split('.')[0])
            files.append(file)
        except Exception as e:
            print(e, file)

    data = pd.DataFrame(
        {'video_name': files,
         'tag': tags,
         })


    return data

data = get_data(train_directory)

data = data.groupby('tag').tail(130)
print(data['tag'].value_counts())

test_data_from_folder = get_data(test_directory)

msk = np.random.rand(len(data)) < model_config['TRAIN_SPLIT']

train_df  = data[msk]
train_df.reset_index(inplace=True)
del train_df['index']

test_df = data[~msk]
test_df.reset_index(inplace=True)
del test_df['index']

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

#train_df.sample(10)


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(model_config['IMG_SIZE'], model_config['IMG_SIZE'])):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights=model_config["feature_extractor_weights"],
        include_top=False,
        pooling=model_config["pooling"],
        input_shape=(model_config['IMG_SIZE'], model_config['IMG_SIZE'], 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((model_config['IMG_SIZE'], model_config['IMG_SIZE'], 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
)
print(label_processor.get_vocabulary())


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, model_config['MAX_SEQ_LENGTH']), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, model_config['MAX_SEQ_LENGTH'], model_config['NUM_FEATURES']), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(model_config['train_path'], path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, model_config['MAX_SEQ_LENGTH'],), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, model_config['MAX_SEQ_LENGTH'], model_config['NUM_FEATURES']), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(model_config['MAX_SEQ_LENGTH'], video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()
    print(labels)
    return (frame_features, frame_masks), labels


train_data, train_labels = prepare_all_videos(train_df, "train")
test_data, test_labels = prepare_all_videos(test_df, "test")

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")

# Utility for our sequence model.
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((model_config['MAX_SEQ_LENGTH'], model_config['NUM_FEATURES']))
    mask_input = keras.Input((model_config['MAX_SEQ_LENGTH'],), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(4, activation="relu", kernel_initializer='random_normal')(x)
    #top_layers = Dense(4096, activation='relu', kernel_initializer='random_normal')(top_layers)


    output = keras.layers.Dense(len(class_vocab) -1, activation="sigmoid")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    adagrad = tf.keras.optimizers.SGD(
    learning_rate=0.001, momentum=0.0, nesterov=False, name="SGD",clipvalue = 0.5)
    #rnn_model.compile(loss='binary_crossentropy', optimizer=adagrad, metrics=["accuracy"])

    rnn_model.compile(
        loss="binary_crossentropy", optimizer=adagrad, metrics=["accuracy"]
    )
    return rnn_model


# Utility for running experiments.
def run_experiment():
    filepath = r"C:\Users\User\Desktop\SCOOL\SCOOL\thesis\tmp\video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.15,
        epochs=model_config['EPOCHS'],
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


_, sequence_model = run_experiment()

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, model_config['MAX_SEQ_LENGTH'],), dtype="bool")
    frame_features = np.zeros(shape=(1, model_config['MAX_SEQ_LENGTH'], model_config['NUM_FEATURES']), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(model_config['MAX_SEQ_LENGTH'], video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join(model_config['test_path'], path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%, {np.argsort(probabilities)}")
        print('yes')
    return frames


test_videos = test_data_from_folder["video_name"].values.tolist()

for test_video in test_videos:
    print(f"Test video path: {test_video}")
    test_frames = sequence_prediction(test_video)
