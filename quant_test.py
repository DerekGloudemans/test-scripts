import torch
import torchvision
import time
import os,sys
import numpy as np

#model = torchvision.models.resnet50()

detector_path = os.path.join("./retinanet")
sys.path.insert(0,detector_path)
# filter and CNNs
from retinanet.model import resnet50 


b = 250
size = [4,3,1920,1080]
dev = "cuda:0"

device = torch.device(dev)
model = resnet50(8)
model = model.to(device)
model.eval()
model.training = False


# Full frame detection benchmark

total_time = 0
for i in range(b):
    inp = torch.rand(size).to(device)
    
    start = time.time()
    output = model(inp)
    torch.cuda.synchronize()
    total_time += time.time() - start

print("FP32 1080p inference:  {}s, {} bps with batch size {}".format(total_time,b/total_time,size[0]))    
del model


model = resnet50(8)
device = torch.device(dev)
model = model.to(device).half()
model.eval()
model.training = False

total_time2 = 0
for i in range(b):
    inp = torch.rand(size).to(device).half()
    
    start = time.time()
    output = model(inp)
    torch.cuda.synchronize()
    total_time2 += time.time() - start

speedup = np.round((total_time/total_time2  -1 )*100,2)
print("FP16 1080p inference: {}s, ({}% speedup), {} bps with batch size {}".format(total_time2,speedup,b/total_time2,size[0]))    



# Crop detection benchmark
size = [30,3,112,112]
b = 2500

device = torch.device(dev)
model = resnet50(8)
model = model.to(device)
model.eval()
model.training = False

total_time = 0
for i in range(b):
    inp = torch.rand(size).to(device)
    
    start = time.time()
    output = model(inp)
    torch.cuda.synchronize()
    total_time += time.time() - start

print("FP32 crop inference: {}s, {} bps with batch size {}".format(total_time,b/total_time,size[0]))    
del model


model = resnet50(8)
device = torch.device(dev)
model = model.to(device).half()
model.eval()
model.training = False

total_time2 = 0
for i in range(b):
    inp = torch.rand(size).to(device).half()
    
    start = time.time()
    output = model(inp)
    torch.cuda.synchronize()
    total_time2 += time.time() - start

speedup = np.round((total_time/total_time2  -1 )*100,2)
print("FP16 crop inference: {}s, ({}% speedup), {} bps with batch size {}".format(total_time2,speedup,b/total_time2,size[0]))    