from ARpredictorClass import AR_predictor
import socket
import struct
import cv2
import time
import numpy as np
import pickle

HOST = ''
PORT = 2000

socket_object = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

socket_object.bind((HOST, PORT))
print('Socket bind complete')

socket_object.listen(10)
print('Socket now listening')

conn, address = socket_object.accept()

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))

MAX_SEQ_LENGTH = 30 # you should use whatever feature the model you are going to use is trained on
NUM_FEATURES = 2048 # you should use whatever feature the model you are going to use is trained on
IMG_SIZE = 320

classes = ['Basketball', 'BenchPres', 'Biking']

predictor = AR_predictor(MAX_SEQ_LENGTH=MAX_SEQ_LENGTH, NUM_FEATURES=NUM_FEATURES, IMG_SIZE=IMG_SIZE, Seq_model_path='my_model.h5',
                         classes_of_model=classes, channel_size=3)

feature_extractor = predictor.build_feature_extractor()

frame_list = []

number_of_frame = 0

start_time = time.time()


while True:
    while len(data) < payload_size:
        data += conn.recv(4096)

        # receive image row data form client socket
    packed_msg_size = data[:payload_size]

    data = data[payload_size:]

    msg_size = struct.unpack(">L", packed_msg_size)[0]

    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]

    data = data[msg_size:]

    # unpack image using pickle
    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")

    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    frame = predictor.prepare_frames(frame)

    if number_of_frame < MAX_SEQ_LENGTH:
        frame_list.append(frame)
    else:
        frame_list.remove(frame_list[0])
        frame_list.append(frame)

    number_of_frame += 1

    frame_array = np.array(frame_list)

    constant_time = time.time()

    if constant_time - start_time > 10:
        predictor.sequence_prediction(frames=frame_array, feature_extractor=feature_extractor)
        start_time = time.time()
    else:
        pass