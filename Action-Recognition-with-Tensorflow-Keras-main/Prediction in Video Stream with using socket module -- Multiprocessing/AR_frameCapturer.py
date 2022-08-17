import socket
import struct
import pickle
import imutils
from cv2 import *

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 2000))

test_video_path = 'videoplayback.mp4'

cam = cv2.VideoCapture(test_video_path) # or you can use camera

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    ret, frame_original = cam.read()

    frame = imutils.resize(frame_original, width=320)

    frame = cv2.flip(frame, 1)

    result, image = cv2.imencode('.jpg', frame, encode_param)

    data = pickle.dumps(image, 0)
    size = len(data)

    client_socket.sendall(struct.pack(">L", size)+data)
    cv2.imshow('Frame_main', frame_original)

    if cv2.waitKey(16) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()