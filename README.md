# test-scripts

## Quantization Test
Do we see a speedup quantizing from FP32 to FP16 on various GPUs for the Retinanet Resnet50FPN detector? We look at 3 different GPUs, with 2 different implementations, at two different image sizes.

| GPU | Detector | Batch Size | Trials | FP32 perf (bps) | FP16 perf (bps) | Speedup |
| --- | -------- | ---------- | ------ | ---------------- | ---------------- | ------- |
| Quadro RTX 6000 | custom      | [4,3,1920,1080] | 250  | 3.138 | 6.613 | +110.7% |
| Quadro RTX 6000 | custom      | [30,3,112,112]  | 2500 | 43.00 | 40.76 | +42.68%  |
| Quadro RTX 6000 | torchvision | [4,3,1920,1080] | 250  | N/A  | N/A | N/A |
| Quadro RTX 6000 | torchvision | [30,3,112,112]  | 2500 | N/A  | N/A | N/A |
