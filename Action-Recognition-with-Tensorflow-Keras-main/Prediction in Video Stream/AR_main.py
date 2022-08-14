from ARpredictorClass import AR_predictor
import cv2
import numpy as np


MAX_SEQ_LENGTH = 30 # you should use whatever feature the model you are going to use is trained on
NUM_FEATURES = 2048 # you should use whatever feature the model you are going to use is trained on
IMG_SIZE = 320

classes = ['Basketball', 'BenchPres', 'Biking']

test_video_path = 'AcRecogDataset//test//Biking//v_Biking_g24_c05.avi'

predictor = AR_predictor(MAX_SEQ_LENGTH=MAX_SEQ_LENGTH, NUM_FEATURES=NUM_FEATURES, IMG_SIZE=IMG_SIZE, Seq_model_path='my_model.h5',
                         classes_of_model=classes, channel_size=3)

feature_extractor = predictor.build_feature_extractor()

frame_list = []

number_of_frame = 0

cam = cv2.VideoCapture(test_video_path)


while cam.isOpened:
    ret, frame_original = cam.read()

    if not ret:
        break

    frame = predictor.prepare_frames(frame_original)

    if number_of_frame < MAX_SEQ_LENGTH:
        frame_list.append(frame)
    else:
        frame_list.remove(frame_list[0])
        frame_list.append(frame)

    frame_array = np.array(frame_list)

    predictor.sequence_prediction(frames=frame_array, feature_extractor=feature_extractor)

    print('\n')

    cv2.imshow('frame', frame_original)

    if cv2.waitKey(16) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()