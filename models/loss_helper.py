import torch
import torch.nn as nn
from models.nn_distance import nn_distance, huber_loss

FAR_THRESHOLD = 1.8     #1.8, 1.5, 1.2, 0.9, 0.6
NEAR_THRESHOLD = 1.5    #1.5, 1.2, 0.9, 0.6, 0.3
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

def compute_objectness_loss(aggregated_vote_xyz, gt_center, objectness_scores):
    #print("FAR_THRESHOLD = " + str(FAR_THRESHOLD) + "NEAR_THRESHOLD = " + str(NEAR_THRESHOLD))
    """ Compute objectness loss for the proposals.
    Args:
        end_points: dict (read-only)
    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """
    # Associate proposal and GT objects by point-to-point distances
    # aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    # gt_center = end_points['center_label'][:, :, 0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center)  # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
    #print("Distances")
    #print(euclidean_dist1)
    objectness_label = torch.zeros((B, K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B, K)).cuda()
    objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1
    #print("Labels")
    #print(objectness_label)
    objectness_mask[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 > FAR_THRESHOLD] = 1
    #print("Masks")
    #print(objectness_mask)

    #print("Objectness Scores")
    #print(objectness_scores)

    # Compute objectness loss
    # objectness_scores = end_points['objectness_scores']
    #criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    criterion = nn.CrossEntropyLoss(reduction='none')
    #objectness_loss = criterion(objectness_scores.transpose(2, 1), objectness_label)
    objectness_loss = criterion(objectness_scores, objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)

    # Set assignment
    object_assignment = ind1  # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment