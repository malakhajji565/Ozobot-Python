from tkinter import *
import datetime
import imutils
import cv2
import os
from operator import itemgetter

from imutils.video import VideoStream
import argparse
import time
import math
import numpy as np
from playsound import playsound

class App:
    def __init__(self,master):
        self.master = master

        self.frame = None

        # creating the frames in the master
        # left side of the frame
        self.app = Frame(master, width=640, height=640, bg = 'lightblue')

        self.app.grid()

        # labels for the window
        self.heading =Label(self.app, text="Program the Ozobot!", font='arial 40 bold', fg='black',
                             bg='lightblue')
        self.heading.grid()

        self.translate = Button(self.app, text="Translate to Ozopy", width=20, height=2, command=lambda : self.detect_marker())
        self.translate.grid()

        self.recognition = Button(self.app, text="Recognise Block", width=20, height=2,
                             command=lambda: self.recognise())
        self.recognition.grid()


    def detect_marker(self):
        self.vs = VideoStream(src=0).start()
        # define names of each possible ArUco tag OpenCV supports
        ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            #	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            #	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            #	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            #	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }



        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_7X7_50"])
        arucoParams = cv2.aruco.DetectorParameters_create()

        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        time.sleep(2.0)

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 600 pixels
            self.frame = self.vs.read()
            self.frame = imutils.resize(self.frame, width=1000)

            # detect ArUco markers in the input frame
            (corners, ids, rejected) = cv2.aruco.detectMarkers(self.frame,
                                                               arucoDict, parameters=arucoParams)


            # verify *at least* one ArUco marker was detected
            if len(corners) > 0:
                # flatten the ArUco IDs list
                ids = ids.flatten()

                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):
                    # extract the marker corners (which are always returned
                    # in top-left, top-right, bottom-right, and bottom-left
                    # order)
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners

                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    # draw the bounding box of the ArUCo detection
                    cv2.line(self.frame, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(self.frame, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(self.frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(self.frame, bottomLeft, topLeft, (0, 255, 0), 2)

                    # compute and draw the center (x, y)-coordinates of the
                    # ArUco marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(self.frame, (cX, cY), 4, (0, 0, 255), -1)
                    # draw the ArUco marker ID on the frame
                    cv2.putText(self.frame, str(markerID),
                                (topLeft[0], topLeft[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

            # show the output frame

            cv2.imshow('frame', self.frame)

            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, grab last frame and order markers then break.
            if key == ord("q"):
                self.vs.stop()
                self.last_frame = self.vs.read()
                (corner, self.id, rejecteds) = cv2.aruco.detectMarkers(self.last_frame,

                                             arucoDict, parameters=arucoParams)
                listC = []
                listB =[]
                for (markerCorner, markerID) in zip(corner, self.id):
                    # extract the marker corners (which are always returned
                    # in top-left, top-right, bottom-right, and bottom-left
                    # order)
                    corner = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corner
                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    # draw the bounding box of the ArUCo detection
                    # compute and draw the center (x, y)-coordinates of the
                    # ArUco marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                    listB.append(cX)
                    listC.append(cY)
                res_list = list(zip(self.id, listB, listC))
                #sort by distance from start block and the (x,y) coordinates of its center, we get cX,cY of every marker and sort it
                #comparingly to centre of start block.
                sorted_list = sorted(res_list, key=lambda x: x[2], reverse=True)
                #print(res_list)
                print(sorted_list)
                my_id = list(map(itemgetter(0), sorted_list ))

                time.sleep(3.0)
                break
        # do a bit of cleanup

        cv2.destroyAllWindows()
        self.makefile(my_id)
        self.vs.stop()


    def recognise(self):
        self.detect_marker()
        for i in self.id:
            if i == 1:
                playsound('Go Forward.mp3')




    def makefile(self, id):
        import os.path

        file_name = '/Users/malak/Documents/GitHub/Ozobot-Project/Ozobot-Python/mycode.ozopy'
        f = open(file_name, "w")
        print(id)
        id = [7,9,12,8,11,9,14,7,8]
        #if id[1] !=1 :
            #print('Insert Start Block')
        #if id[-1] !=12:
            #print('Insert end block')
        i = 0
        while i < len(id):
            if id[i] == 1:
                continue
            if id[i] == 2:
                f.write('wait\n')
            if id[i] == 3:
                f.write('rotate(90,50)\n') #turn left
            if id[i] == 4:
                f.write('rotate(-90,50)\n')  # turn right

            if id[i] == 5:
                f.write('rotate(90,50)\n rotate(90,50)\n')  # u turn left

            if id[i] == 6:
                f.write('rotate(-90,50)\n rotate(-90,50)\n')  # u turn right

            if id[i] == 7 and id[i+1] == 8 or id[i] == 7 and id[i+1] == 8:
                f.write('move(20,50)\n') #move 2cm forward

            if id[i] == 7 and id[i+1] == 9 or id[i] == 7 and id[i+1] == 9:
                f.write('move(50,50)\n')  # move 5cm forward

            if id[i] == 7 and id[i + 1] == 10 or id[i] == 7 and id[i + 1] == 10:
                f.write('move(100,50)\n') #move 10 cm

            if id[i] == 11 and id[i + 1] == 8 or id[i] == 11 and id[i + 1] == 8:
                f.write('move(-20,50)\n')  # move 2cm backward

            if id[i] == 11 and id[i + 1] == 9 or id[i] == 11 and id[i + 1] == 9:
                f.write('move(-50,50)\n')  # move 5cm backwards

            if id[i] == 11 and id[i + 1] == 10 or id[i] == 11 and id[i + 1] == 10:
                f.write('move(-100,50)\n')  # move 10 cm backwards

            if id[i] == 12 and id[i+1] == 8 or id[i] == 12 and id[i+1] == 8: #loop with 2 repetitions

                end_index = id.index(14)
                loop_list = id[i+2:end_index]
                rest_list = id[end_index:]
                j = 0
                k =0
                break
            i = i + 1
        while j < len(loop_list):  #takes commands in loop list, from start of loop till end of loop index 14.

            if loop_list[j] == 11 and loop_list[j+1] == 8 or loop_list[j] == 11 and loop_list[j+1] == 8:
                f.write('i =0 \nwhile i < 2:\n\tmove(-20,50)\n\ti=i+1\n')

            if loop_list[j] == 11 and loop_list[j+1] == 9 or loop_list[j] == 11 and loop_list[j+1] == 9:
                f.write('i =0 \nwhile i < 2:\n\tmove(-50,50)\n\ti=i+1\n')

            if loop_list[j] == 11 and loop_list[j+1] == 10 or loop_list[j] == 11 and loop_list[j+1] == 10:
                f.write('i =0 \nwhile i < 2:\n\tmove(-100,50)\n\ti=i+1\n')

            if loop_list[j] == 7 and loop_list[j+1] == 8 or loop_list[j] == 7 and loop_list[j+1] == 8:
                f.write('i =0 \nwhile i < 2:\n\tmove(20,50)\n\ti=i+1\n')

            if loop_list[j] == 7 and loop_list[j+1] == 9 or loop_list[j] == 7 and loop_list[j+1] == 9:
                f.write('i =0 \nwhile i < 2:\n\tmove(50,50)\n\ti=i+1\n')

            if loop_list[j] == 7 and loop_list[j+1] == 10 or loop_list[j] == 7 and loop_list[j+1] == 10:
                f.write('i =0 \nwhile i < 2:\n\tmove(100,50)\n\ti=i+1\n')

            if loop_list[j] == 2:
                f.write('i =0 \nwhile i < 2:\n\twait\n\ti=i+1\n')
            if loop_list[j] == 3:
                f.write('i =0 \nwhile i < 2:\n\trotate(90,50)\n\ti=i+1\n')
            if loop_list[j] == 4:
                f.write('i =0 \nwhile i < 2:\n\trotate(-90,50)\n\ti=i+1\n')
            if loop_list[j] == 5:
                f.write('i =0 \nwhile i < 2:\n\trotate(90,50)\nrotate(90,50)\n\ti=i+1\n') #u turn left

            if loop_list[j] == 6:
                f.write('i =0 \nwhile i < 2:\n\trotate(90,50)\nrotate(90,50)\n\ti=i+1\n') # u turn right

            j = j +1

        while k <len(rest_list):
            if rest_list[k] == 7 and rest_list[k+1] == 8 or rest_list[k] == 7 and rest_list[k+1] == 8:
                f.write('move(20,50)\n') #move 2cm forward
            k = k +1










        return file_name







root = Tk()
b = App(root)
root.mainloop()

