#!/bin/sh

#SBATCH --gres=gpu:1

python -u test_models.py
python stitch_tracklets.py --predictions=/globalwork/kreuzberg/4D-PLS/test/Log_2022-06-17_12-16-59_importance_None_str1_bigpug_4_current_chkp --n_test_frames=4 --dataset=/globalwork/kreuzberg/SemanticKITTI/dataset --data_cfg=/globalwork/kreuzberg/SemanticKITTI/dataset/semantic-kitti.yaml
python -u utils/evaluate_4dpanoptic.py --predictions=/globalwork/kreuzberg/4D-PLS/test/Log_2022-06-17_12-16-59_importance_None_str1_bigpug_4_current_chkp/stitch4 --dataset=/globalwork/kreuzberg/SemanticKITTI/dataset --data_cfg=/globalwork/kreuzberg/SemanticKITTI/dataset/semantic-kitti.yaml
