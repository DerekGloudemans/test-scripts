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

At different batch sizes, on Quadro RTX6000:
- FP32 1080p inference:  20.41656255722046s, 12.244960399153273 bps with batch size 1
- FP16 1080p inference: 9.65434193611145s, (111.48% speedup), 25.895084476435514 bps with batch size 1
- FP32 1080p inference:  99.54781913757324s, 2.5113558706344397 bps with batch size 5
- FP16 1080p inference: 46.7585551738739s, (112.9% speedup), 5.3466151610194785 bps with batch size 5
- FP32 1080p inference:  191.56854677200317s, 1.3050159027282255 bps with batch size 10
- FP16 1080p inference: 92.82949876785278s, (106.37% speedup), 2.69310944600916 bps with batch size 10
- FP32 1080p inference:  403.1211507320404s, 0.6201609604110753 bps with batch size 20
- FP16 1080p inference: 182.76192116737366s, (120.57% speedup), 1.3678998251011467 bps with batch size 20
- FP32 crop inference: 36.69801211357117s, 68.12358097934911 bps with batch size 1
- FP16 crop inference: 34.85886359214783s, (5.28% speedup), 71.71777110264547 bps with batch size 1
- FP32 crop inference: 38.03472137451172s, 65.72941537769039 bps with batch size 10
- FP16 crop inference: 35.488523960113525s, (7.17% speedup), 70.44530797645501 bps with batch size 10
- FP32 crop inference: 56.58725094795227s, 44.1795626774562 bps with batch size 30
- FP16 crop inference: 38.05466675758362s, (48.7% speedup), 65.69496498091905 bps with batch size 30
- FP32 crop inference: 83.59995317459106s, 29.9043229698822 bps with batch size 60
- FP16 crop inference: 47.80979800224304s, (74.86% speedup), 52.29053676157992 bps with batch size 60
