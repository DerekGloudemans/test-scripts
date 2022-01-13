import numpy as np
import cv2
import matplotlib.pyplot as plt
import timestamp_utilities as tsu

%## global settings
video_sequence = ""
coarse_rps = 1
fine_rps = 8

checksum_path="/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_pixel_checksum_6.pkl"
geom_path="/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_geometry_4K.pkl"
checksums = tsu.get_precomputed_checksums(checksum_path)
geom = tsu.get_timestamp_geometry(geom_path)
       
%## Load video file
cap = cv2.VideoCapture(video_sequence)
ret,frame = cap.read()

%## using first frame, select center and radius of coarse and fine wheel
if frame is not None:
  
all_analog_ts = []
all_timestamp_ts = []
while ret:
  
  # extract timestamp both ways and append to lists


  # display frame
  cv2.imshow("frame {}".format(i), frame)
  key = cv2.waitKey(1)
  if key == ord("q"):
    cv2.destroyAllWindows()
    cap.release()
    break

  # load next frame
  ret, frame = cv2.read()

  # end main loop
  
  

def extract_analog_time(frame,center1,center2,radius1,radius2):
  pass

def extract_timestamp_time(frame):
  ts = tsu.parse_frame_timestamp(frame_pixels = frame, timestamp_geometry = geom, precomputed_checksums = checksums)[0]
  return ts
