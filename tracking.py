# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
from ultralytics import YOLO

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
ap.add_argument("-m", "--model", type=str, default="best.pt", help="path to yolo model")
args = vars(ap.parse_args())

# Load the YOLOv8 model
model = YOLO(args["model"])

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]
# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
	tracker = cv2.Tracker_create(args["tracker"].upper())
# otherwise, for OpenCV 3.3 OR NEWER, we need to explicitly call the
# appropriate object tracker constructor:
else:
	# initialize a dictionary that maps strings to their corresponding
	# OpenCV object tracker implementations
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"mil": cv2.TrackerMIL_create
	}
	# grab the appropriate object tracker using our dictionary of
	# OpenCV object tracker objects
	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
# initialize the FPS throughput estimator
fps = None
fps_started = False

# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
	# check to see if we have reached the end of the stream
	if frame is None:
		break
	# resize the frame (so we can process it faster) and grab the
	# frame dimensions
	frame = imutils.resize(frame, width=640)
	(H, W) = frame.shape[:2]

	# Initialize success to False at the beginning of each loop
	success = False

	# check to see if we are currently tracking an object
	if initBB is not None:
		# grab the new bounding box coordinates of the object
		(success, box) = tracker.update(frame)
		# check to see if the tracking was a success
		if success:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
		else:
			initBB = None  # Reset if tracking fails
			fps_started = False
	else:
		# Run YOLOv8 detection
		results = model(frame)
		best_conf = 0
		best_box = None
		for result in results:
			boxes = result.boxes.cpu().numpy()
			for box in boxes:
				conf = box.conf[0]
				if conf > best_conf:
					best_conf = conf
					best_box = box.xyxy[0].astype(int)
			
		if best_box is not None:
			x1, y1, x2, y2 = best_box
			initBB = (x1, y1, x2 - x1, y2 - y1)
			tracker.init(frame, initBB)
			if not fps_started:
				fps = FPS().start()
				fps_started = True
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

	if initBB is not None:
		# update the FPS counter
		if fps_started:
			fps.update()
			fps.stop()
		# initialize the set of information we'll be displaying on
		# the frame
		info = [
			("Tracker", args["tracker"]),
			("Success", "Yes" if success else "No"),
			("FPS", "{:.2f}".format(fps.fps()) if fps_started else "N/A"),
		]
		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()
# otherwise, release the file pointer
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()
