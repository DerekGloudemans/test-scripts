import numpy as np
import cv2
import matplotlib.pyplot as plt
import timestamp_utilities as tsu
global px,py
import time

#%% global settings
video_sequence = "/home/worklab/Data/cv/video/clock_test/6128test.mp4"
coarse_rps = 2
fine_rps = 6.015
circle_trim = 25
discard_frames = 60

checksum_path="/home/worklab/Documents/derek/test-scripts/ts/timestamp_pixel_checksum_6_h23.pkl"
geom_path="/home/worklab/Documents/derek/test-scripts/ts/ts_geom_h23.pkl"
checksums = tsu.get_precomputed_checksums(checksum_path)
geom = tsu.get_timestamp_geometry(geom_path)
 

def extract_analog_time(coarse_direction,coarse_rev,coarse_prev,coarse_rps,coarse_start,fine_direction,fine_rev,fine_prev,fine_rps,fine_start):
    # coarse time
    coarse_sec_per_deg = 1/coarse_rps/360.0
    coarse_sec_per_rev = 1/coarse_rps
    coarse_sec = ((coarse_start-coarse_direction[2])%360)/360 * coarse_sec_per_deg
    
    if coarse_direction[2] < coarse_start and coarse_prev > coarse_start: # since spinning clockwise
        coarse_rev += 1
    
    coarse_prev = coarse_direction[2]
    coarse_time = coarse_sec + coarse_rev*coarse_sec_per_rev
    
    # fine time
    fine_sec_per_deg = 1/fine_rps/360.0
    fine_sec_per_rev = 1/fine_rps
    fine_sec = ((fine_start-fine_direction[2])%360)/360 * fine_sec_per_deg
    
    if fine_direction[2] < fine_start and fine_prev > fine_start: # since spinning clockwise
        fine_rev += 1
    
    fine_prev = fine_direction[2]
    fine_time = fine_sec + fine_rev*fine_sec_per_rev
    
    
    return coarse_rev,coarse_prev,coarse_time,fine_rev,fine_prev,fine_time
        
def on_mouse(event, x, y, flags,params):
    global px,py
    if event == cv2.EVENT_LBUTTONDOWN:
      px = x
      py = y  

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((int(cx), int(cy)), int(radius))    

def dist(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)



def find_directions(contours):
    # # assume 2 contours
    # if len(contours) != 2:
    #     return None
    
    output = []
    for cont in contours:
        # find minimum enclosing triangle
        tri = cv2.minEnclosingTriangle(cont)[1][:,0,:]
        
        # find closest point on contour to each corner
        refined_triangle = []
        for corner in tri:
            minp = None
            mindist = np.inf
            for point in cont[:,0,:]:
                d = dist(point,corner)
                if d < mindist:
                    mindist = d
                    minp = point
            refined_triangle.append(minp)
        
        # of 3 triangle edges, find two that point in most similar direction
        d01 = dist(refined_triangle[0],refined_triangle[1])
        d12 = dist(refined_triangle[1],refined_triangle[2])
        d20 = dist(refined_triangle[2],refined_triangle[0])
        
        if d01 < d12 and d01 < d20:
            back = (refined_triangle[0][0] + refined_triangle[1][0])/2.0,(refined_triangle[0][1]+refined_triangle[1][1])/2.0
            front = refined_triangle[2]
        elif d12 < d20 and d12 < d01:
            back = (refined_triangle[2][0] + refined_triangle[1][0])/2.0,(refined_triangle[2][1]+refined_triangle[1][1])/2.0
            front = refined_triangle[0]
        else:
            back = (refined_triangle[2][0] + refined_triangle[0][0])/2.0,(refined_triangle[2][1]+refined_triangle[0][1])/2.0
            front = refined_triangle[1]
    
        angle = np.arctan2(-(front[1]-back[1]),front[0]-back[0])*180/np.pi %360
    
        # record angle, and edges for plotting
        output.append([front,back,angle])
    
    return output
    
#%% Load video file
cap = cv2.VideoCapture(video_sequence)
ret,frame = cap.read()

#%% define circles
cv2.namedWindow("window")
cv2.setMouseCallback("window", on_mouse, 0)
px,py = None,None 
points = []
circles = []

while True:    
    cv2.imshow("window",frame)
    key = cv2.waitKey(1)

    if px is not None and py is not None:
        points.append([px,py])
        px,py = None,None
    
        
    if len(points) == 3:
        circles.append(define_circle(*points))
        points = []
        cv2.circle(frame,circles[-1][0],circles[-1][1],[0,255,0],2)
    
    if key == ord("q"):
        break
    
#%% Create mask
mask = np.zeros(frame.shape).astype(np.uint8)
for circle in circles:
    cv2.circle(mask,circle[0],circle[1]-circle_trim,[255,255,255],-1)
    
# cv2.imshow("Mask",mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


all_coarse_ts = []
all_fine_ts = []
all_timestamp_ts = []

start_ts = None
start_analog = None

white = (np.ones(frame.shape)*255).astype(np.uint8)

coarse_rev = 0
fine_rev = 0

fine_prev = 0
coarse_prev = 0

fine_start = None
coarse_start = None

i = 0
while i < discard_frames:
    ret,frame = cap.read()
    i += 1
 
start = time.time()
i = 0
while ret:
    
    # parse timestamp
    timestamp = tsu.parse_frame_timestamp(geom,checksums,frame_pixels = frame)[0]
    all_timestamp_ts.append(timestamp)

    
    # mask and threshold image
    mask_white = cv2.bitwise_and(white,mask-255) * 255
    mask_frame = cv2.bitwise_and(frame,mask)
    masked = cv2.addWeighted(mask_white,1,mask_frame,1,0)
    masked = cv2.blur(masked, (3,3), cv2.BORDER_DEFAULT) 
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(masked, 170, 255, cv2.THRESH_BINARY)


    contours,_ = cv2.findContours(thresh, 1, 2)
    directions = find_directions(contours)
    
    # select only contours directions corresponding to coarse and fine clock
    for di in directions:
        if dist(di[0],circles[0][0]) < circles[0][1] and dist(di[1],circles[0][0]) < circles[0][1]:
            coarse_direction = di
            cv2.line(frame,(int(di[0][0]),int(di[0][1])),(int(di[1][0]),int(di[1][1])),(0,0,255),2)
        elif dist(di[0],circles[1][0]) < circles[1][1] and dist(di[1],circles[1][0]) < circles[1][1]:
            fine_direction = di
            cv2.line(frame,(int(di[0][0]),int(di[0][1])),(int(di[1][0]),int(di[1][1])),(255,0,0),2)

    if coarse_start is None:
        coarse_start = coarse_direction[2]
        fine_start = fine_direction[2]
        
    coarse_pos = ((coarse_start-coarse_direction[2])%360)/360
    if coarse_pos < coarse_prev:
        coarse_rev += 1
    coarse_prev = coarse_pos
    coarse_pos += coarse_rev
    coarse_time = coarse_pos/coarse_rps
    
    fine_pos = ((fine_start-fine_direction[2])%360)/360
    if fine_pos < fine_prev:
        fine_rev += 1
    fine_prev = fine_pos
    fine_pos += fine_rev
    fine_time = fine_pos/fine_rps
    
    
    all_coarse_ts.append(coarse_time)
    all_fine_ts.append(fine_time)
    
    # display
    if False:
        
        cv2.rectangle(frame,(20,35),(300,280),(255,255,255),-1)
        text_size = 1.0
        cv2.putText(frame, "Coarse Analog", (20,60), cv2.FONT_HERSHEY_PLAIN,text_size*2, [0,0,255], 2)
        cv2.putText(frame, "Position: {:.1f}deg".format(coarse_direction[2]),(20,75), cv2.FONT_HERSHEY_PLAIN,text_size, [1,1,1], 1)
        cv2.putText(frame, "{:.3f}rev since start".format(coarse_pos),(20,90), cv2.FONT_HERSHEY_PLAIN,text_size, [1,1,1], 1)
        cv2.putText(frame, "Est. time: {:.3f}s since start".format(coarse_time),(20,105), cv2.FONT_HERSHEY_PLAIN,text_size, [1,1,1], 1)
    
        cv2.putText(frame, "Fine Analog", (20,140), cv2.FONT_HERSHEY_PLAIN,text_size*2, [255,0,0], 2)
        cv2.putText(frame, "Position: {:.1f}deg".format(fine_direction[2]),(20,155), cv2.FONT_HERSHEY_PLAIN,text_size, [1,1,1], 1)
        cv2.putText(frame, "{:.3f}rev since start".format(fine_rev + ((fine_start-fine_direction[2])%360)/360),(20,170), cv2.FONT_HERSHEY_PLAIN,text_size, [1,1,1], 1)
        cv2.putText(frame, "Est. time: {:.3f}s since start".format(fine_time),(20,185), cv2.FONT_HERSHEY_PLAIN,text_size, [1,1,1], 1)
        
        cv2.putText(frame, "Timestamp", (20,220), cv2.FONT_HERSHEY_PLAIN,text_size*2, [1,1,1], 2)
        cv2.putText(frame, "Start time: {}".format(all_timestamp_ts[0]),(20,235), cv2.FONT_HERSHEY_PLAIN,text_size, [1,1,1], 1)
        cv2.putText(frame, "Reported time: {}".format(all_timestamp_ts[-1]),(20,250), cv2.FONT_HERSHEY_PLAIN,text_size, [1,1,1], 1)
        cv2.putText(frame, "Est. time: {:.3f}s since start".format(all_timestamp_ts[-1] - all_timestamp_ts[0]),(20,265), cv2.FONT_HERSHEY_PLAIN,text_size, [1,1,1], 1)
        
        # display frame
        cv2.imshow("window", frame)
        key = cv2.waitKey(10)
        cv2.imwrite("_out_ims/{}.png".format(str(i).zfill(5)),cv2.resize(frame,(1920,1080)))
        if key == ord("p"):
            cv2.waitKey(0)
        if key == ord("q"):
          cv2.destroyAllWindows()
          cap.release()
          break
      
    # load next frame
    ret, frame = cap.read()
    i += 1
    print("\r{} fps       ".format(i/(time.time() - start)),end = '\r', flush = True)
    # end main loop

# plot results
all_timestamp_ts = [item - all_timestamp_ts[0] for item in all_timestamp_ts]

coarse_delta = [all_coarse_ts[i] - all_coarse_ts[i-1] for i in range(1,len(all_coarse_ts))]
fine_delta = [all_fine_ts[i] - all_fine_ts[i-1] for i in range(1,len(all_fine_ts))]
timestamp_delta = [all_timestamp_ts[i] - all_timestamp_ts[i-1] for i in range(1,len(all_timestamp_ts))]
fps30_delta = [1/30.0 for i in range(len(fine_delta))]
all_30fps = [i/30.0 for i in range(len(all_fine_ts))]

plt.figure(figsize = (20,10))
plt.plot(coarse_delta,color = (1,0,0))
plt.plot(fine_delta,color = (0,0,1))
plt.plot(timestamp_delta,color = (0,0,0),marker = ".")
plt.plot(fps30_delta,color = (0.5,1,0.5),linestyle = "--")
plt.legend(["Coarse Time Deltas",
            "Fine Time Deltas",
            "Timestamp Deltas",
            "30 FPS Deltas"])
plt.xlabel("Frame number",fontsize = 20)
plt.ylabel("Frame deltas (s)",fontsize = 20)
plt.show()


rel_coarse = [all_coarse_ts[i] - all_fine_ts[i] for i in range(len(all_fine_ts))]
rel_fine   = [all_fine_ts[i] - all_fine_ts[i] for i in range(len(all_fine_ts))]
rel_ts     = [all_timestamp_ts[i] - all_fine_ts[i] for i in range(len(all_fine_ts))]
rel_30fps  = [all_30fps[i] - all_fine_ts[i] for i in range(len(all_fine_ts))]
plt.figure(figsize = (20,10))
plt.plot(rel_coarse,color = (1,0,0))
plt.plot(rel_fine,color = (0,0,1))
plt.plot(rel_ts,color = (0,0,0))
plt.plot(rel_30fps,color = (0.5,1,0.5),linestyle = "--")
plt.legend(["Coarse Relative Time",
            "Fine Baseline",
            "Timestamp Relative Time",
            "30 FPS Relative Time"])
plt.xlabel("Frame number",fontsize = 20)
plt.ylabel("Relative Time (s)",fontsize = 20)
plt.show()



# coarse_delta = [all_coarse_ts[i] - all_coarse_ts[i-1] for i in range(1,len(all_coarse_ts))]
# fine_delta = [all_fine_ts[i] - all_fine_ts[i-1] for i in range(1,len(all_fine_ts))]
# timestamp_delta = [all_timestamp_ts[i] - all_timestamp_ts[i-1] for i in range(1,len(all_timestamp_ts))]
# fps30_delta = [1/30.0 for i in range(len(fine_delta))]
# all_30fps = [i/30.0 for i in range(len(all_fine_ts))]

# x = all_fine_ts[:-1]
# plt.figure(figsize = (20,10))
# plt.plot(x,coarse_delta,color = (1,0,0))
# plt.plot(x,fine_delta,color = (0,0,1))
# plt.plot(x,timestamp_delta,color = (0,0,0))
# plt.plot(x,fps30_delta,color = (0.5,1,0.5),linestyle = "--")
# plt.legend(["Coarse Time Deltas",
#             "Fine Time Deltas",
#             "Timestamp Deltas",
#             "30 FPS Deltas"])
# plt.xlabel("Frame number",fontsize = 20)
# plt.ylabel("Frame deltas (s)",fontsize = 20)
# plt.show()
