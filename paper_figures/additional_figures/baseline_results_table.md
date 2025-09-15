# Baseline Model Performance Results

| Model                  |   Dice |    IoU |   Boundary IoU |   Sensitivity |   Specificity | Parameters   |   GPU Memory (GB) |
|:-----------------------|-------:|-------:|---------------:|--------------:|--------------:|:-------------|------------------:|
| DuaSkinSeg             | 0.8785 | 0.7854 |         0.1512 |        0.8901 |        0.9711 | 31.2M        |               6.8 |
| Lightweight DuaSkinSeg | 0.8772 | 0.7839 |         0.1508 |        0.8896 |        0.9708 | 8.4M         |               4.2 |
| Attention U-Net        | 0.8722 | 0.7793 |         0.1495 |        0.8884 |        0.9698 | 57.8M        |               7.2 |
| Custom U-Net           | 0.863  | 0.7583 |         0.1445 |        0.8845 |        0.9682 | 4.3M         |               3.5 |
| MONAI U-Net            | 0.845  | 0.7321 |         0.1398 |        0.8789 |        0.9671 | 2.6M         |               2.8 |