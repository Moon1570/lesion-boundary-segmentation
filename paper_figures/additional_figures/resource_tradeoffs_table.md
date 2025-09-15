# Model Resource Trade-offs

| Model                  |   Dice | Parameters   | GPU Memory   | Inference Time   |   Efficiency Score |   Resource Score |
|:-----------------------|-------:|:-------------|:-------------|:-----------------|-------------------:|-----------------:|
| DuaSkinSeg             | 0.8785 | 31.2M        | 6.8 GB       | 0.210 s          |                7.1 |             0.48 |
| Lightweight DuaSkinSeg | 0.8772 | 8.4M         | 4.2 GB       | 0.150 s          |                9.2 |             0.64 |
| Attention U-Net        | 0.8722 | 57.8M        | 7.2 GB       | 0.310 s          |                6.2 |             0.43 |
| Custom U-Net           | 0.863  | 4.3M         | 3.5 GB       | 0.120 s          |                8.8 |             0.78 |
| MONAI U-Net            | 0.845  | 2.6M         | 2.8 GB       | 0.090 s          |                8.5 |             0.99 |