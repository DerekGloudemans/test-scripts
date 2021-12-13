import cv2
import numpy as np
import time

cv2.namedWindow("frame")
im = np.zeros([2160,3840,1]) * 0 
im[0:10,0:10,0] = 255
cv2.imshow("frame",im)
cv2.waitKey(0)

times = []
for i in range(20000):
    t= time.time()
    im = np.zeros([2160,3840,1]) * 0 
    #im[:,:,0] = np.random.randint(0,255)
    # im[:,:,1] += np.random.randint(0,255)
    # im[:,:,2] += np.random.randint(0,255)
    
    #im = cv2.rectangle(im,(0,200*(i%4+1)),(999,200*(i%4+1) + 5),(255,255,255),-1)
    
    if i%2 == 0:
        im = im + 255
    cv2.imshow("frame",im)
    cv2.waitKey(1)
    t2 = time.time()
    
    diff = t2-t
    times.append(diff)
    
    
mean = sum(times)/len(times)
print("Average refresh time: {}".format(mean))
    