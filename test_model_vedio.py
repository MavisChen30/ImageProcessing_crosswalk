# import the necessary packages
from nms import non_max_suppression
from object_detector import ObjectDetector
from hog import HOG
from conf import Conf
import numpy as np
import imutils
import argparse
import pickle
import cv2
# import pyttsx3  # 用於語音提示

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to theconfiguration file")
ap.add_argument("-v", "--video", required=True, help="path to the video file")
args = vars(ap.parse_args())
# load the configuration file
conf = Conf(args["conf"])
# load the classifier, then initialize the Histogram of OrientedGradients descriptor
# and the object detector
model = pickle.loads(open(conf["classifier_path"], "rb").read())
hog = HOG(orientations=conf["orientations"],
          pixelsPerCell=tuple(conf["pixels_per_cell"]),
          cellsPerBlock=tuple(conf["cells_per_block"]),
          normalize=conf["normalize"], block_norm="L1")
od = ObjectDetector(model, hog)

# # Initialize text-to-speech engine
# engine = pyttsx3.init()
# # Set properties for speech (optional)
# engine.setProperty('rate', 150)  # 語速
# engine.setProperty('volume', 1)  # 音量 (範圍從 0.0 到 1.0)

# load the image and convert it to grayscale
cap = cv2.VideoCapture(args["video"])
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

played_voice = False

while True:
    # grab the next frame
    ret, frame = cap.read()

    # if no frame is grabbed, we've reached the end of the video
    if not ret:
        break

    # resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width=320)
    frame = imutils.resize(frame, width=min(260, frame.shape[1]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect objects in the frame
    (boxes, probs) = od.detect(gray, conf["window_dim"],
                               winStep=conf["window_step"],
                               pyramidScale=conf["pyramid_scale"],
                               minProb=conf["min_probability"])
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])

    # draw the original bounding boxes (red)
    # for (startX, startY, endX, endY) in boxes:
    #     cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # draw the filtered bounding boxes after non-max suppression (green)
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, "There's a crosswalk ahead", (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Only play the voice alert if it hasn't been played yet
        # if not played_voice:
        #     engine.say("There's a crosswalk ahead")
        #     engine.runAndWait()
        #     played_voice = True  # Mark that the voice has been played

    # display the frame
    cv2.imshow("Frame", frame)

    # break the loop if the user presses the 'q' key
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):  # 'q' key to quit
        break
    elif key == 32:  # Space key (ASCII 32) to stop
        print("Program stopped by user.")
        break


# release the video capture and close all OpenCV windows
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()