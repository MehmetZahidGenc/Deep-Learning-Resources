import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2


class AR_predictor:

    def __init__(self, MAX_SEQ_LENGTH, NUM_FEATURES, IMG_SIZE, Seq_model_path, classes_of_model, channel_size):
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
        self.NUM_FEATURES = NUM_FEATURES
        self.IMG_SIZE = IMG_SIZE
        self.Seq_model_path = Seq_model_path
        self.classes_of_model = classes_of_model
        self.channel_size = channel_size
        self.seq_model = tf.keras.models.load_model(filepath=self.Seq_model_path)


    def build_feature_extractor(self):
        feature_extractor = keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(self.IMG_SIZE, self.IMG_SIZE, self.channel_size),
        )
        preprocess_input = keras.applications.inception_v3.preprocess_input

        inputs = keras.Input((self.IMG_SIZE, self.IMG_SIZE, self.channel_size))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)

        return keras.Model(inputs, outputs, name="feature_extractor")


    def crop_center_square(self, frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2)-(min_dim // 2)
        start_y = (y // 2)-(min_dim // 2)

        return frame[start_y: start_y+min_dim, start_x: start_x+min_dim]


    def prepare_frames(self, frame):
        frame = self.crop_center_square(frame)
        frame = cv2.resize(frame, (self.IMG_SIZE, self.IMG_SIZE))

        return frame

    def prepare_single_video(self, frames, feature_extractor):
        frames = frames[None, ...]
        frame_mask = np.zeros(shape=(1, self.MAX_SEQ_LENGTH,), dtype="bool")
        frame_features = np.zeros(shape=(1, self.MAX_SEQ_LENGTH, self.NUM_FEATURES), dtype="float32")

        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(self.MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        return frame_features, frame_mask


    def sequence_prediction(self, frames, feature_extractor):

        frame_features, frame_mask = self.prepare_single_video(frames, feature_extractor)

        probabilities = self.seq_model.predict([frame_features, frame_mask])[0]

        for i in np.argsort(probabilities)[::-1]:
            print(f"  {self.classes_of_model[i]}: {probabilities[i] * 100:5.2f}%")