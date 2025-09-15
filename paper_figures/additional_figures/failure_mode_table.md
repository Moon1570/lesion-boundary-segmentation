# Model Failure Mode Analysis

Legend: ++ Very Good (<15%), + Good (<25%), ! Fair (<35%), X Poor (≥35%)

| Failure Mode          | DuaSkinSeg   | Lightweight DuaSkinSeg   | Attention U-Net   | Custom U-Net   | MONAI U-Net   |
|:----------------------|:-------------|:-------------------------|:------------------|:---------------|:--------------|
| Small Lesions         | 10% ++       | 10% ++                   | 15% +             | 20% +          | 25% !         |
| Irregular Boundaries  | 10% ++       | 15% +                    | 20% +             | 30% !          | 35% X         |
| Low Contrast          | 15% +        | 20% +                    | 10% ++            | 25% !          | 15% +         |
| Hair Occlusions       | 15% +        | 20% +                    | 25% !             | 30% !          | 40% X         |
| Artifacts             | 10% ++       | 15% +                    | 20% +             | 15% +          | 10% ++        |
| Similar Color to Skin | 10% ++       | 10% ++                   | 15% +             | 20% +          | 30% !         |