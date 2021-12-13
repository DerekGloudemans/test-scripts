import torch
import numpy
import time

# allocation 
start = time.time()
test = torch.rand([10000,1000,100])
torch.cuda.synchronize()
end = time.time()
print("Allocation took {} sec. Data is at address {}. Contiguous: {}".format(end - start,test.storage().data_ptr(),test.is_contiguous()))

# view of a contiguous tensor
start = time.time()
test.view(-1)
torch.cuda.synchronize()
end = time.time()
print("view() took {} sec. Data is at address {}. Contiguous: {}".format(end - start,test.storage().data_ptr(),test.is_contiguous()))


# diagonal() on a contiguous tensor
start = time.time()
test.diagonal()
torch.cuda.synchronize()
end = time.time()
print("diagonal() took {} sec. Data is at address {}. Contiguous: {}".format(end - start,test.storage().data_ptr(),test.is_contiguous()))


# Diagonal and a few tensor view ops on a non-contiguous tensor
test = test[::2,::2,::2]    # indexing is a Tensor View op resulting in a non-contiguous output
print(test.is_contiguous()) # False
start = time.time()
test = test.unsqueeze(-1).expand([test.shape[0],test.shape[1],test.shape[2],100]).diagonal()
torch.cuda.synchronize()
end = time.time()
print("non-contiguous tensor ops() took {} sec. Data is at address {}. Contiguous: {}".format(end - start,test.storage().data_ptr(),test.is_contiguous()))

# reshape, which requires a tensor copy operation to new memory
start = time.time()
test = test.reshape(-1) + 1.0
torch.cuda.synchronize()
end = time.time()
print("reshape() took {} sec. Data is at address {}. Contiguous: {}".format(end - start,test.storage().data_ptr(),test.is_contiguous()))
