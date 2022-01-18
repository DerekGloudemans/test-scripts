import os
import ast
import csv
import cv2
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Big
if False:
    video_file = "/home/worklab/Data/dataset_beta/sequence_4/record_50_p1c2_00000.mp4.mp4"
    w = 13
    h = 18
    x0 = 14
    y0 = 3
    n = 13
    h13 = int(h/3)
    h23 = int(2*h/3)
    h12 = int(h/2)
    w12 = int(w/2)
    tenths = 11
    get_dig = [(0, 14, tenths), (1, 17, tenths), (2, 20, tenths), (3,23,tenths), (4, 25, tenths),
               (5, 27 ,tenths), (6, 29, tenths), (7, 32, tenths), (8, 35, tenths), (9, 38, tenths)]

# clock test super big
elif True:
    video_file = "/home/worklab/Data/cv/video/clock_test/6128test.mp4"
    w = 18
    h = 23
    x0 = 2
    y0 = 5
    n = 13
    h13 = int(h/3)
    h23 = int(2*h/3)
    h12 = int(h/2)
    w12 = int(w/2)
    tenths = 11
    get_dig = [(0, 33, tenths), (1, 36, tenths), (2, 39, tenths), (3,42,tenths), (4, 14, tenths),
               (5, 48 ,tenths), (6, 20, tenths), (7, 23, tenths), (8, 26, tenths), (9, 30, tenths)]
    
# Small
else:
    video_file = "/home/worklab/Data/dataset_beta/sequence_4/record_50_p3c2_00000.mp4.mp4"
    w = 8
    h = 12
    x0 = 10
    y0 = 2
    n = 13
    h13 = int(h/3)
    h23 = int(2*h/3)
    h12 = int(h/2)
    w12 = int(w/2)
    tenths = 11
    # (digit, frame_i, character_i)
    get_dig = [(0, 11, tenths), (1, 14, tenths), (2, 17, tenths), (3, 20, tenths), (4,23,tenths), 
                (5, 25 ,tenths), (6, 29, tenths), (7, 32, tenths), (8, 35, tenths), (9, 38, tenths)]

cap = cv2.VideoCapture(video_file)
assert cap.isOpened(), "Cannot open file \"{}\"".format(video_file)
ret, frame = cap.read()


# save geometry 
with open('ts_geom_h{}.pkl'.format(h), 'wb') as f:
    pickle.dump({'w': w, 'h': h, 'x0': x0, 'y0': y0, 'n': n, 
                  'h13': h13, 'h23': h23, 'h12': h12, 'w12': w12}, f)
    
    
# extract digits
digits = {}
try:
    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened(), "Cannot open file \"{}\"".format(video_file)
    
    i = 0
    dig, fri, chi = zip(*get_dig)
    
    while i <= max(fri):
        ret, frame = cap.read()
        # check if we need to extract from this frame
        if i in fri:
            # convert timestamp to black/white
            tsimg = frame[0:y0+h, 0:x0+(n*w), :]
            tsgray = cv2.cvtColor(tsimg, cv2.COLOR_BGR2GRAY)
            ret, tsmask = cv2.threshold(tsgray, 127, 255, cv2.THRESH_BINARY)
            
            # extract digits for any references of this frame
            for j in range(len(fri)):
                
                # pixels = tsmask[y0:y0+h, x0+j*w:x0+(j+1)*w]
                # cv2.imshow("dig",pixels)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                if fri[j] == i:
                    pixels = tsmask[y0:y0+h, x0+chi[j]*w:x0+(chi[j]+1)*w]
                    
                    
                    
                    digits[dig[j]] = pixels
        i += 1
        continue
except BaseException as e:
    print(e)
finally:
    cap.release()

# plot digits
fig, axs = plt.subplots(1, len(digits), figsize=(len(digits), 2))
for i, (dig, pixels) in enumerate(sorted(digits.items())):
    axs[i].imshow(pixels, 'gray')
    axs[i].set_title(str(dig), fontsize=16)
plt.tight_layout()
plt.show()


# get checksums
dig_checksum4 = {}
dig_checksum6 = {}
for dig, pixels in sorted(digits.items()):
#     print('\n', dig)
    cs = [[int(pixels[:h12, :w12].sum()/255), int(pixels[:h12, w12:].sum()/255)],
          [int(pixels[h12:, :w12].sum()/255), int(pixels[h12:, w12:].sum()/255)]
         ]
    cs = np.array(cs)
#     print(cs)
    dig_checksum4[dig] = cs
    cs = [[int(pixels[:h13, :w12].sum()/255), int(pixels[:h13, w12:].sum()/255)], 
          [int(pixels[h13:h23, :w12].sum()/255), int(pixels[h13:h23, w12:].sum()/255)],
          [int(pixels[h23:, :w12].sum()/255), int(pixels[h23:, w12:].sum()/255)]
         ]
    cs = np.array(cs)
#     print(cs)
    dig_checksum6[dig] = cs
    
# save checksum
with open('./timestamp_pixel_checksum_6_h{}.pkl'.format(h), 'wb') as f:
    pickle.dump(dig_checksum6, f)
fig, axs = plt.subplots(3, len(dig_checksum6), figsize=(len(dig_checksum6), 6))
vmin4 = min([np.array(cs).min() for cs in dig_checksum4.values()])
vmax4 = max([np.array(cs).max() for cs in dig_checksum4.values()])
vmin6 = min([np.array(cs).min() for cs in dig_checksum6.values()])
vmax6 = max([np.array(cs).max() for cs in dig_checksum6.values()])
for i, (dig, cs) in enumerate(sorted(dig_checksum6.items())):
    im6 = axs[1][i].imshow(cs, vmin=vmin6, vmax=vmax6)
    axs[1][i].set_axis_off()
    im4 = axs[0][i].imshow(dig_checksum4[dig], vmin=vmin4, vmax=vmax4)
    axs[0][i].set_axis_off()
    axs[0][i].set_title(str(dig), fontsize=16)
    axs[2][i].imshow(digits[dig], 'gray')
    axs[2][i].set_axis_off()

plt.savefig('timestamp_digits_4K_h{}.pdf'.format(h))
plt.show()


# test

t0 = time.time()
cap = cv2.VideoCapture(video_file)
assert cap.isOpened(), "Cannot open file \"{}\"".format(video_file)

cam = video_file.split('record_')[1].split('_')[0]
print("CAMERA {}".format(cam))
cam_ts = []
i = 0
while True:
    ret, frame = cap.read()
    if frame is None:
        print("END OF VIDEO. BREAKING LOOP.")
        break
    tsimg = frame[0:y0+h, 0:x0+(n*w), :]
    tsgray = cv2.cvtColor(tsimg, cv2.COLOR_BGR2GRAY)
    ret, tsmask = cv2.threshold(tsgray, 127, 255, cv2.THRESH_BINARY)

    ts_dig = []
    for j in range(n):
        if j == 10:
            ts_dig.append('.')
            continue
        pixels = tsmask[y0:y0+h, x0+j*w:x0+(j+1)*w]
        
        cs = [[int(pixels[:h13, :w12].sum()/255), int(pixels[:h13, w12:].sum()/255)], 
              [int(pixels[h13:h23, :w12].sum()/255), int(pixels[h13:h23, w12:].sum()/255)],
              [int(pixels[h23:, :w12].sum()/255), int(pixels[h23:, w12:].sum()/255)]
             ]
        cs = np.array(cs)
        cs_diff = [(dig, abs(cs - cs_ref).sum()) for dig, cs_ref in dig_checksum6.items()]
        pred_dig, pred_err = min(cs_diff, key=lambda x: x[1])
        if pred_err > 0:
            pass#print(cs)
        else:
            ts_dig.append(pred_dig)
    cam_ts.append(ast.literal_eval(''.join(map(str, ts_dig))))
    i += 1
    print (cam_ts[-1])
    continue
timestamps[cam] = cam_ts
cap.release()
print(i / (time.time() - t0), "fps")
        