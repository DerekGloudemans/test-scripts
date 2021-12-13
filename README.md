# test-scripts

## Quantization Test
Do we see a speedup quantizing from FP32 to FP16 on various GPUs for the Retinanet Resnet50FPN detector? We look at 3 different GPUs, with 2 different implementations, at two different image sizes.

| GPU | Detector | Batch Size | Trials | FP32 perf (bps) | FP16 perf (bps) | Speedup |
| --- | -------- | ---------- | ------ | ---------------- | ---------------- | ------- |
| Quadro RTX 6000 | custom      | [4,3,1920,1080] | 250  | 3.138 | 6.613 | +110.7% |
| Quadro RTX 6000 | custom      | [30,3,112,112]  | 2500 | 43.00 | 66.28 | +42.68%  |
| Quadro RTX 6000 | torchvision | [4,3,1920,1080] | 250  | N/A  | N/A | N/A |
| Quadro RTX 6000 | torchvision | [30,3,112,112]  | 2500 | N/A  | N/A | N/A |
| RTX A5000       | custom      | [4,3,1920,1080] | 250  | 3.82 | 7.56 | +97.79% |
| RTX A5000       | custom      | [30,3,112,112]  | 2500 | 52.32 | 61.22  | +17.03% |
| RTX A5000       | torchvision | [4,3,1920,1080] | 250  | 6.58 | 6.16  | -6.34% |
| RTX A5000       | torchvision | [30,3,112,112]  | 25 | 1.36 | 0.91 | -33.29% |
| RTX A6000       | custom      | [4,3,1920,1080] | 250  | 5.11 | 9.27 | +81.14% |
| RTX A6000       | custom      | [30,3,112,112]  | 2500 | 60.17 | 65.92 | +9.57% |
| RTX A6000       | torchvision | [4,3,1920,1080] | 250  | 8.48  | 7.12 | -16.04% |
| RTX A6000       | torchvision | [30,3,112,112]  | 25 | 1.71 | 1.05 | -38.55% |
| A30             | custom      | [4,3,1920,1080] | 250  | | | |
| A30             | custom      | [30,3,112,112]  | 2500 | | | |
| A30             | torchvision | [4,3,1920,1080] | 250  |  | | |
| A30             | torchvision | [30,3,112,112]  | 2500 |  | | |
