import os
from pynvml import *

try:
    nvmlInit()
except:
    print('Failed to initialize NVML.')
    print('Exiting...')
    os._exit(1)

print("Driver Version:", nvmlSystemGetDriverVersion(), '\n')

deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    gpu_device = nvmlDeviceGetHandleByIndex(i)
    print("Device", i, ":", nvmlDeviceGetName(gpu_device))
    totalMemory = float(nvmlDeviceGetMemoryInfo(gpu_device).total)/1e6
    usedMemory = float(nvmlDeviceGetMemoryInfo(gpu_device).used)/1e6
    print('total memory [MBi]:\t%.2f\nused memory [MBi]:\t%.2f' %(totalMemory, usedMemory))
    print("used memory [perc]:\t%.3f\n" %(usedMemory/totalMemory))

