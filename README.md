# 4D-StOP

Official code for "4D-StOP: Panoptic Segmentation of 4D LiDAR using Spatio-temporal Object Proposal Generation and Aggregation".

In this work, we present a new paradigm called 4D-StOP to tackle the task of 4D Panoptic LiDAR Segmentation. 4D-StOP first generates spatio-temporal proposals using voting based center predictions, where each point in the 4D volume votes for a corresponding center. These tracklet proposals are further aggregated using learnt geometric features. The tracklet aggregation method effectively generates a video level 4D scene representation over the entire space-time volume. This is in contrast to the existing end-to-end trainable state-of-the-art approach which uses spatio-temporal embeddings that are represented by Gaussian probability functions. Our voting-based tracklet generation method followed by the geometric feature based aggregation technique generates a much better panoptic LiDAR segmentation quality as compared to modelling the entire 4D volume using Gaussian probabilities. 4D-StOP achieves a new state-of-the-art when applied to the SemanticKITTI test dataset with a score of 63.9 LSTQ, which is a large (+7%) improvement when compared to the current best performing end-to-end trainable method.

Code and models will be available soon.
