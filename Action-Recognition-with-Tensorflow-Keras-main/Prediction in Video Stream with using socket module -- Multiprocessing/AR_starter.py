import runpy
from multiprocessing import Process
import time


def run_analyzer():
    runpy.run_path(path_name='AR_predictor_reciever.py')


def run_capturer():
    runpy.run_path(path_name='AR_frameCapturer.py')


if __name__ == '__main__':
    p1 = Process(target=run_analyzer)
    p1.start()

    time.sleep(10) # wait socket created and start to listen

    p2 = Process(target=run_capturer)
    p2.start()

    p1.join()
    p2.join()