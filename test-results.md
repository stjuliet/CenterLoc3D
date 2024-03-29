# test-results

## AP

### 2021.7.20
- all modules, Scene A-C

| backbone            |       Car AP (threshold=0.5\\0.7)  2D/3D        |     FPS                             |
|---------------------|-------------------------------------------------|-------------------------------------|
| ResNet-50           | val: 94.76% / 88.10% \\ 94.28% / 69.48% <br/>test: 81.36% / 70.02% \\ 80.09% / 38.67% |23.4 <br/>21.8|

### 2022.2.20
- all modules, Scene A-E

| backbone            |       Car AP (threshold=0.5\\0.7)  2D/3D        |     FPS                             |
|---------------------|-------------------------------------------------|-------------------------------------|
| ResNet-50           | val: 95.77% / 91.34% \\ 95.52% / 79.36% <br/>test: 78.82% / 68.19% \\ 77.44% / 51.30% |41.0860 <br/>41.1783|

### 2022.2.25
- iou module(×), Scene A-E

| backbone            |       Car AP (threshold=0.5\\0.7)  2D/3D        |     FPS                             |
|---------------------|-------------------------------------------------|-------------------------------------|
| ResNet-50           | val: 96.00% / 90.92% \\ 95.69% / 67.89% <br/>test: 79.18% / 68.38% \\ 77.97% / 45.32% |42.1808 <br/>41.3121|

### 2022.2.26
- iou, proj module(×), Scene A-E

| backbone            |       Car AP (threshold=0.5\\0.7)  2D/3D        |     FPS                             |
|---------------------|-------------------------------------------------|-------------------------------------|
| ResNet-50           | val: 96.90% / 90.54% \\ 96.79% / 57.07% <br/>test: 85.45% / 70.22% \\ 81.58% / 42.84% |43.0502 <br/>43.2298|

### 2022.2.27
- iou, proj, weighted-fusion module(×), Scene A-E

| backbone            |       Car AP (threshold=0.5\\0.7)  2D/3D        |     FPS                             |
|---------------------|-------------------------------------------------|-------------------------------------|
| ResNet-50           | val: 95.13% / 83.46% \\ 94.85% / 52.52% <br/>test: 68.76% / 58.83% \\ 65.52% / 36.20% |48.8892 <br/>46.7315|

### 2022.12.30
- noaug, Scene A-E

| backbone            | Car AP (threshold=0.5\\0.7)  2D/3D                                                    | FPS           |
|---------------------|---------------------------------------------------------------------------------------|---------------|
| ResNet-50           | val: 96.26% / 91.10% \\ 95.85% / 71.95% <br/>test: 80.71% / 70.04% \\ 78.73% / 46.26% | 40.2505 <br/>41.149 |