import torch
import torchvision
import time
import os,sys
import numpy as np


detector_path = os.path.join("./retinanet")
sys.path.insert(0,detector_path)
# filter and CNNs
from retinanet.model import resnet50 

# use torchvision model instead
#resnet50 = torchvision.models.detection.retinanet_resnet50_fpn

torch.cuda.set_device(0)
torch.cuda.empty_cache()
dev = "cuda:0"

b = 250
for batch_size in [4]:
    size = [batch_size,3,1920,1080]
    
    device = torch.device(dev)
    model = resnet50(8)
    model = model.to(device)
    model.eval()
    model.training = False
    
    
    # Full frame detection benchmark
    
    total_time = 0
    for i in range(b):
        inp = torch.rand(size).to(device)
        
        with torch.no_grad():
            start = time.time()
            output = model(inp)
            torch.cuda.synchronize()
            total_time += time.time() - start
    
    print("FP32 1080p inference:  {}s, {} bps with batch size {}".format(total_time,b/total_time,size[0]))    
    del model



    model = resnet50(8)
    device = torch.device(dev)
    model = model.to(device).half()
    
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.float()
            
    model.eval()
    model.training = False
    
    total_time2 = 0
    for i in range(b):
        inp = torch.rand(size).to(device).half()
        
        with torch.no_grad():
            start = time.time()
            output = model(inp)
            torch.cuda.synchronize()
            total_time2 += time.time() - start
            
    speedup = np.round((total_time/total_time2  -1 )*100,2)
    print("FP16 1080p inference: {}s, ({}% speedup), {} bps with batch size {}".format(total_time2,speedup,b/total_time2,size[0]))    



# Crop detection benchmark
b = 2500
for batch_size in [30]:
    size = [batch_size,3,112,112]
    
    device = torch.device(dev)
    model = resnet50(8)
    model = model.to(device)
    model.eval()
    model.training = False
    
    total_time = 0
    for i in range(b):
        inp = torch.rand(size).to(device)
        
        with torch.no_grad():
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
        
        with torch.no_grad():
            start = time.time()
            output = model(inp)
            torch.cuda.synchronize()
            total_time2 += time.time() - start
    
    speedup = np.round((total_time/total_time2  -1 )*100,2)
    print("FP16 crop inference: {}s, ({}% speedup), {} bps with batch size {}".format(total_time2,speedup,b/total_time2,size[0]))    
