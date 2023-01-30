# import the necessary packages
from threading import Thread
import sys
import cv2
import time

from queue import Queue


class RTPVideoStream:
    def __init__(self, ip  = "192.168.1.25", port = "5000",queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.frame_width = 1920
        self.frame_height = 1080
        self.stopped = False
        self.fps = 60
        self.gst_str_rtp = f"appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=6000 speed-preset=superfast ! rtph264pay ! udpsink host={ip} port={port}"
        
        # print(self.gst_str_rtp)
        # self.gst_str_rtp = f"appsrc ! videoconvert ! openh264enc ! rtph264pay config-interval=10 pt=96 ! udpsink host={ip} port={port}"

        self.out = cv2.VideoWriter(self.gst_str_rtp, 0, self.fps, (self.frame_width, self.frame_height), True)

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break
            
            if self.Q:
                # print("cool",cv2.resize(self.Q.get(),(self.frame_width,self.frame_height)).shape)
                self.out.write(cv2.resize(self.Q.get(),(self.frame_width,self.frame_height)))

        self.out.release()

    def write(self,frame):
        # return next frame in the queue
        if not self.Q.full():
            self.Q.put(frame)
        else:
            self.Q.get()

    def running(self):
        return self.more() or not self.stopped

    def more(self):
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.out.release()
        self.stopped = True
        self.thread.join()
