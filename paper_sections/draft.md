Reproducible U-Net Baselines and Lightweight Ensembles for Skin Lesion Segmentation on ISIC-2018 under low GPU resources
*Note: Sub-titles are not captured in Xplore and should not be used

 
line 1: 1st Given Name Surname 
line 2: dept. name of organization 
(of Affiliation)
line 3: name of organization 
(of Affiliation)
line 4: City, Country
line 5: email address or ORCID
line 1: 4th Given Name Surname
line 2: dept. name of organization
(of Affiliation)
line 3: name of organization 
(of Affiliation)
line 4: City, Country
line 5: email address  or ORCID 
line 1: 2nd Given Name Surname
line 2: dept. name of organization 
(of Affiliation)
line 3: name of organization 
(of Affiliation)
line 4: City, Country
line 5: email address or ORCID
line 1: 5th Given Name Surname
line 2: dept. name of organization 
(of Affiliation)
line 3: name of organization 
(of Affiliation)
line 4: City, Country
line 5: email address  or ORCID 
line 1: 3rd Given Name Surname
line 2: dept. name of organization 
(of Affiliation)
line 3: name of organization 
(of Affiliation)
line 4: City, Country
line 5: email address or ORCID
line 1: 6th Given Name Surname
line 2: dept. name of organization 
(of Affiliation)
line 3: name of organization 
(of Affiliation)
line 4: City, Country
line 5: email address or ORCID 
 
 
 
Abstract—Skin lesion boundary segmentation remains a central problem in melanoma analysis, yet many reported benchmarks suffer from inconsistent experimental splits, limited reproducibility, and high computational demands. In this work, we present a comprehensive evaluation of segmentation architectures on the ISIC-2018 lesion segmentation challenge, establishing deterministic data splits and fixed seeds to ensure exact repeatability. Our evaluation shows that DuaSkinSeg achieves the highest performance (87.85% Dice), while our Lightweight DuaSkinSeg variant maintains competitive accuracy (87.72% Dice) with 73% fewer parameters (8.4M vs 31.2M). For more constrained environments, Custom U-Net (86.30% Dice, 4.3M parameters) and MONAI U-Net (84.50% Dice, 2.6M parameters) remain viable options. Our fine-tuned UNetMamba model shows meaningful improvements (81.61% Dice, +0.61% over baseline) through progressive unfreezing, as documented in our training logs, though it still lags behind transformer-based approaches. Our Enhanced Ensemble (87.53% Dice) demonstrates particular strength in boundary detection (0.1502 Boundary IoU), while maintaining deployment feasibility on consumer-grade hardware (≤8GB GPU). We provide detailed computational tradeoffs in parameters, FLOPs, GPU memory usage and inference time, alongside analysis of common failure modes including small lesions and irregular boundaries. To facilitate reproducibility, we release our code, data splits, and trained checkpoints, offering practical guidance for researchers with limited computational resources.
Keywords—Skin lesion segmentation, ISIC-2018 challenge, U-Net variants, Medical image segmentation, Reproducible benchmarks, Resource-constrained deep learning
Introduction
Skin cancer, particularly melanoma, is a leading cause of morbidity and mortality worldwide, and early detection through automated lesion analysis is crucial. Dermoscopic image segmentation of skin lesions enables precise measurement and feature extraction, aiding diagnosis. The ISIC dermoscopy dataset series has become a cornerstone for benchmarking these algorithms, providing extensive annotated images that have driven significant advances in dermatological imaging AI. In this context, U-Net and its variants are ubiquitous due to their ability to capture fine-grained details. The classic U-Net architecture uses a contracting path to capture context and a symmetric expanding path for accurate localization. Many works build upon this core, for example, adding attention gates or leveraging frameworks like MONAI for ease of development.
However, recent literature reveals two persistent issues. First, reproducibility is often limited: studies may omit precise train/val/test splits or random seeds, making exact replication impossible. Second, state-of-the-art models often demand substantial compute resources. For instance, hybrid architectures using Mamba state-space blocks (e.g., DermoMamba) report high Dice scores (~0.88-0.90 on ISIC data) but typically require significant GPU memory. Meanwhile, Transformer-based models like DuaSkinSeg achieve excellent performance (87.85% Dice) but at the cost of large parameter counts (31.2M). These trends can exclude researchers with limited hardware.
In this work, we address these issues by providing a fully deterministic, open benchmark of lightweight segmentation models and their ensembles on the ISIC-2018 Task 1 challenge. Our comprehensive evaluation shows that while the standard DuaSkinSeg model achieves the highest performance (87.85% Dice), our Lightweight DuaSkinSeg variant offers a compelling alternative with minimal performance reduction (87.72% Dice) despite using 73% fewer parameters (8.4M vs 31.2M) and 38% less GPU memory. For even more constrained environments, our Custom U-Net (86.30% Dice, 4.3M parameters) and MONAI U-Net (84.50% Dice, 2.6M parameters) remain viable options. Fine-tuned UNetMamba models demonstrate meaningful improvements (81.61% Dice, +0.61% over baseline) but still lag behind transformer-based approaches.
Our contributions are: (1) Reproducible Baselines: We fix data splits and random seeds to enable exact repeatability; (2) Comprehensive Benchmarking: We evaluate multiple architectures including U-Net variants, Transformer-based models, and State Space Models with detailed reporting of segmentation accuracy and resource requirements; (3) Lightweight Model Optimization: We demonstrate that parameter-efficient models can achieve near state-of-the-art performance while maintaining low memory footprints; (4) Resource Metrics: We quantify trade-offs by reporting model size, FLOPs, GPU memory usage, and inference speed; (5) Failure Analysis: We analyze segmentation performance across challenging cases including small lesions, irregular boundaries, and hair occlusions; and (6) Open Source: We release all code, data splits, and trained checkpoints for exact reproducibility.
The rest of the paper is organized as follows. Section 2 reviews related work on segmentation architectures and benchmarks. Section 3 describes our data, preprocessing, architectures, loss functions, optimization strategy, and experimental setup. Section 4 presents experimental results. Section 5 discusses key insights, practical implications, and limitations. Finally, Section 6 concludes.
Related Work
Skin Lesion Segmentation Approaches
Early approaches to skin lesion segmentation relied on traditional image processing techniques including thresholding [X], region growing [Y], and active contours [Z]. While these methods work well for high-contrast lesions, they struggle with heterogeneous pigmentation and irregular borders common in melanomas. The introduction of deep learning approaches, particularly CNN-based segmentation networks, has substantially improved performance on challenging cases.
U-Net Variants in Medical Image Segmentation
The original U-Net architecture [Ronneberger et al., 2015] consists of a contracting path to capture context and a symmetric expanding path for precise localization. MONAI U-Net [X] adapts this architecture with domain-specific optimizations for medical imaging, including residual connections and advanced normalization techniques. Attention U-Net [Y] incorporates attention gates that highlight salient features and suppress irrelevant regions, showing particular benefits for boundary detection in heterogeneous lesions.
State Space Models and Transformer Approaches
Recently, transformer-based architectures have shown promising results in medical image segmentation. Models like TransUNet [X] and SwinUNet [Y] leverage self-attention mechanisms to capture long-range dependencies. DuaSkinSeg [Z] combines a lightweight Vision Transformer with a CNN encoder for efficient feature extraction. Simultaneously, State Space Models (SSMs) have emerged as alternatives to transformers, with architectures like DermoMamba [A] and UNetMamba [B] reporting competitive performance with potentially reduced computational requirements.
Ensemble Approaches in Medical Image Segmentation
Ensemble methods have consistently demonstrated improved performance in medical image segmentation [X, Y]. These approaches typically combine predictions from multiple models through averaging [Z], weighted fusion [A], or meta-learning approaches [B]. While ensembles generally improve robustness to variations in lesion appearance, they traditionally come with increased computational costs during both training and inference, limiting their practical deployment.
Reproducibility Challenges in Medical Image Segmentation
Despite advances in skin lesion segmentation, reproducibility remains a significant challenge. Many studies report results without specifying exact data splits [X], random seeds [Y], or complete implementation details [Z]. This hampers direct comparison and impedes scientific progress. Recent initiatives like MedPerf [A] and MONAI Label [B] have attempted to standardize evaluation, but widespread adoption remains limited. Additionally, the trend toward increasingly complex models with substantial computational requirements (e.g., [C] requiring 16GB+ GPU memory) creates barriers to reproducibility for researchers with limited resources.
Efficient Deep Learning for Medical Imaging
The demand for efficient deep learning models has led to several parameter-efficient architectures for medical image segmentation. EfficientUNet [X] and MobileNetV2-UNet [Y] demonstrate competitive performance with reduced parameter counts. Orthogonally, model compression techniques including pruning [Z], quantization [A], and knowledge distillation [B] offer pathways to reduce computational requirements of existing architectures. Beyond accuracy, recent work has begun to systematically report efficiency metrics including FLOPS, memory usage, and inference time [C, D], highlighting the importance of these considerations for real-world deployment.
Methodology
Dataset and Preprocessing
Our study utilizes the ISIC 2018 Challenge Task 1 dataset, which contains 2,594 dermoscopic images with corresponding ground truth segmentation masks. We implemented a robust preprocessing pipeline to ensure consistent input for all models, addressing common challenges in dermoscopic imaging. Figure 1 shows some samples from the dataset.
 
Data Preprocessing Protocol: All images undergo a standardized pipeline including resolution normalization to 384×384 pixels with aspect ratio preservation, artifact removal (rulers, color calibration charts), color normalization, and optional hair removal through inpainting techniques. During validation, we identified and corrected issues with 140 problematic images that had corrupted masks or significant artifacts, ensuring consistent quality across the dataset.
Reproducible Data Splits: To ensure exact reproducibility, we created deterministic data splits using fixed random seeds (seed=42), resulting in:
	Training set: 2,293 images (80%)
	Validation set: 301 images (10%)
	Test set: 1,000 images (ISIC 2018 challenge test set)
These splits are preserved and shared in our code repository to enable precise replication of our results. We implemented a consistent augmentation pipeline including random flips, rotations (±30°), brightness/contrast adjustments, and elastic deformations, applied only during training with fixed seeds to ensure reproducibility.

Model Architectures
We implemented and evaluated several model architectures with varying complexity and parameter efficiency.
U-Net Variants
U-Net: Our baseline follows the original U-Net architecture with modern optimizations including batch normalization and residual connections. It uses a five-level encoder-decoder structure with skip connections and features [32, 64, 128, 256, 512]. This model contains 4.3M parameters and serves as our foundational architecture.
MONAI U-Net: Leveraging the medical imaging-optimized framework, this variant incorporates domain-specific enhancements including residual units, advanced normalization, and optimized memory usage. With only 2.6M parameters, it represents our most lightweight option.
Attention U-Net: This enhanced architecture incorporates attention mechanisms to highlight salient features and suppress irrelevant regions. It integrates CBAM (Convolutional Block Attention Module) in the skip connections to improve boundary detection. With 57.8M parameters, it represents our largest U-Net variant.
Advanced Architectures
DuaSkinSeg: This dual-encoder architecture combines a MobileNetV2 CNN path with a Vision Transformer encoder path, followed by feature fusion and a specialized decoder with skip connections. The standard version contains 31.2M parameters.
Lightweight DuaSkinSeg: We developed a parameter-efficient variant of DuaSkinSeg that reduces the model size by 73% (to 8.4M parameters) through architectural optimizations including reduced transformer embedding dimensions (from 384 to 192), fewer attention heads (from 12 to 6), and lighter fusion modules.
UNetMamba: This architecture integrates Mamba State Space Model blocks into the U-Net structure, replacing traditional convolutional blocks in the bottleneck and decoder pathways with 2D-adapted Mamba units that process image data along both height and width dimensions sequentially.
Ensemble Method
We implemented a lightweight ensemble approach that combines predictions from multiple models:
Enhanced Ensemble: Our final ensemble combines two Attention U-Net variants and two Custom U-Net variants, using weighted averaging based on validation performance. We incorporate test-time augmentation (TTA) with flips and rotations to further improve robustness.
Training and Optimization Strategy
Loss Functions
We evaluated multiple loss functions to optimize boundary segmentation:
	Dice Loss: Optimizes overlap between predicted and ground truth masks
	Combined Loss: Weighted combination of BCE and Dice (weights: 0.5, 0.5)
	Boundary-Aware Loss: Enhanced version with specific boundary term (weights: 0.4 BCE, 0.4 Dice, 0.2 Boundary)
	Advanced Combined Loss: Multi-term loss with specialized boundary enhancement
Lightweight Ensembles
To mitigate the weaknesses of individual models, we construct simple ensembles of U-Net variants. Ensembles are known to improve robustness and accuracy in segmentation. We experiment with ensembling the outputs of multiple independently trained models (e.g. two Attention U-Nets and one baseline U-Net). Two ensemble strategies are considered: (a) Logit averaging: we average the raw logits (pre-sigmoid) from each model and then threshold; (b) Majority voting: we convert each model’s sigmoid output to a binary mask and take the pixel-wise majority vote. Both methods are lightweight: they require no additional training and incur only minor extra inference time. By combining models that may err on different images, the ensemble often achieves higher overall Dice than any single model. This approach has been shown to yield performance gains in U-Net cascades and parallel U-Net systems.
Optimization & Reproducible training
All models were trained with Adam (initial lr = 1e-3) and a cosine-annealing scheduler; early stopping (patience = 10 epochs) was used to avoid overfitting. Fine-tuning used a progressive unfreeze: Phase 1: freeze the encoder and train the decoder for 10 epochs; Phase 2: unfreeze the encoder using a lower encoder lr (0.1×) with layer-wise learning rates for encoder/decoder. This protocol yielded a +0.61% Dice improvement for the UNetMamba run. For reproducibility we fixed random seeds (torch, numpy, random), set cudnn.deterministic = True, and kept exhaustive logging, checkpointing, and version-controlled configs.
Evaluation Framework
Segmentation metrics: We report standard overlap and boundary metrics: Dice (primary), IoU (Jaccard), Pixel Accuracy, Sensitivity (Recall), Specificity, Precision, Boundary IoU, and Hausdorff-95. Dice and IoU quantify overall overlap; Boundary IoU and HD-95 capture boundary quality and worst-case deviations.
Computational metrics: To characterize efficiency and deployability we measure: trainable parameters, model size (MB), peak GPU memory (GB) during inference, per-image inference time (s), and FLOPs. These allow direct comparison of accuracy vs resource cost and construction of Pareto frontiers.
Experimental Setup
All experiments ran on a single NVIDIA GTX 1070 (8 GB VRAM) using PyTorch 2.1.0 with CUDA 13.0; mixed precision (AMP) was enabled to reduce memory use. Models were trained for up to 100 epochs with early stopping (patience = 10) and batch sizes tuned per model (4–12 to fit memory). Training logs, validation curves, and checkpoints were recorded regularly (the UNetMamba fine-tuning run, for example, progressed 33 epochs and stopped by early stopping). To enable exact replication we release all code, data splits, configs, and pretrained checkpoints.
Result
Model Performance Comparison
We evaluated several architectures on our fixed ISIC-2018 dataset splits with consistent evaluation protocols. Table I & II summarizes the key performance metrics.
Table I: Model Performance Comparison
Model	Dice	IoU	Boundary IoU	HD-95 (px)	Pixel Accuracy
DuaSkinSeg	87.85	78.27	0.1420	19.31	0.9445
Lightweight DuaSkinSeg	87.72	78.11	0.1417	19.47	0.9441
Enhanced Ensemble	87.53	77.86	0.1502	18.73	0.9465
Custom U-Net	86.30	76.11	0.1392	20.15	0.9398
MONAI U-Net	84.50	73.72	0.1322	22.18	0.9365
UNetMamba (Fine-tuned)	81.61	70.09	0.1271	24.62	0.9278
Table II: Model Performance Comparison -2
Model	Sensitivity	Specificity	Precision
DuaSkinSeg	0.9012	0.9869	0.8740
Lightweight DuaSkinSeg	0.9019	0.9871	0.8747
Enhanced Ensemble	0.8804	0.9862	0.8608
Custom U-Net	0.8635	0.9847	0.8461
MONAI U-Net	0.8312	0.9821	0.8203
UNetMamba (Fine-tuned)	0.8388	0.9829	0.8265
As shown in Table 1 & 2, DuaSkinSeg achieved the highest Dice coefficient (87.85%) and IoU score (78.27%), demonstrating superior overall segmentation performance. Remarkably, our Lightweight DuaSkinSeg variant achieved comparable results (87.72% Dice, 78.11% IoU) despite having significantly fewer parameters. The Enhanced Ensemble, while slightly lower in Dice score (87.53%), excelled in boundary accuracy metrics, achieving the best Boundary IoU (0.1502) and Hausdorff distance (18.73 pixels) among all models.
Computational Efficiency
Efficiency score is measured by the equation below and then normalized to a 1-10 scale to produce the final efficiency scores shown in Figure 2.
Efficiency Score= α*1/(GPU Memory (GB) )+β*1/(Inference Time (ms) )+γ*Parameters/max⁡(Parameters) 
Fig. 2 shows the relationship between segmentation performance and computational resources, revealing important efficiency trade-offs.
As detailed in Table III, Lightweight DuaSkinSeg provides an exceptional balance between performance and efficiency.

 
Table III: Computational Efficiency Metrics
Model	Parameters (M)	GPU Memory (GB)	Inference (s)	FLOPs (G)	Efficiency Score
Lightweight DuaSkinSeg	8.4	4.2	0.15	12.6	9.2
Custom 
U-Net	4.3	3.5	0.12	6.8	8.8
MONAI 
U-Net	2.6	2.8	0.09	4.2	8.5
DuaSkinSeg	31.2	6.8	0.21	47.2	7.1
Enhanced Ensemble	Combined	8.0	0.85	124.8	6.8
Lightweight DuaSkinSeg requires 73% fewer parameters and 38% less GPU memory than standard DuaSkinSeg with only 0.13% Dice reduction. This translates to a superior efficiency score of 9.2/10.
Qualitative Results on Single Prediction
Analysis on a single image is shown in Figure 2. Figure 2 presents a qualitative and quantitative analysis of a representative segmentation result from the ISIC-2018 dataset. The model achieves strong agreement with the ground truth, with a Dice score of 0.9412, IoU of 0.8889, and pixel accuracy of 96.7%. While the predicted mask slightly overestimates the lesion area compared to the ground truth (43,257 vs. 38,491 pixels), the error map shows that most discrepancies are concentrated along lesion boundaries. The probability map further highlights boundary uncertainty, whereas the combined overlay demonstrates that the overall lesion shape and extent are well captured. 
Ensemble Performance
Our Enhanced Ensemble combines complementary model strengths to achieve robust performance, particularly for boundary detection. Table III shows the ensemble's performance compared to its components.
Fine-tuning Results
Our progressive unfreezing strategy for UNetMamba yielded a meaningful improvement of +0.61% Dice over the baseline pre-trained model (80.0% to 81.61%). Fig. 3 shows the training dynamics during fine-tuning.

Table III: Ensemble Composition and Improvement
Model	Dice%	IoU%	Boundary IoU	Component Weight
Attention U-Net (Best)	86.91	76.98	0.1418	0.35
Attention U-Net (Second)	86.72	76.64	0.1405	0.30
Custom U-Net (Best)	86.30	76.11	0.1392	0.20
Custom U-Net (Second)	86.30	76.11	0.1392	0.15
Enhanced Ensemble	87.53	77.86	0.1502	-
Improvement over average	+1.07	+1.6	+0.0102	-

Resource Trade-offs
We profile each architecture for resource usage to guide practical deployment. Figure 1 plots Dice accuracy versus parameter count, with bubble size indicating GPU memory. The larger DuaSkinSeg (31.2M parameters, 6.8 GB) outperforms slightly in Dice (87.85%) compared to Lightweight DuaSkinSeg (8.4M, 4.2 GB, 87.72%) and Custom U-Net (4.3M, 3.5 GB), but efficiency scores favor the smaller models, with Lightweight DuaSkinSeg scoring 9.2/10. FLOPs and inference speed further highlight differences: DuaSkinSeg requires 47.2 GFLOPs/image at 0.21 s, Custom U-Net 6.8 GFLOPs at 0.12 s, and MONAI U-Net achieves 0.09 s with only 4.2 GFLOPs. In contrast, Mamba-based hybrids from recent literature demand >8 GB memory, illustrating deployment constraints. Our Enhanced Ensemble balances performance and resources, reaching 87.53% Dice with 8.0 GB memory and 0.85 s inference. Overall, parameter-efficient models like Lightweight DuaSkinSeg and Custom U-Net achieve near state-of-the-art accuracy (within 0.13–1.55% Dice) with 73–86% fewer parameters, enabling practical use on consumer-grade GPUs.
