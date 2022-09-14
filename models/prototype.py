import torch
import torch.nn as nn
import numpy as np
import time
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
from os import getpid
from threadpoolctl import threadpool_limits
from functools import partial
import os

from models.backbone_module2 import KPFCNN
from models.voting_module import VotingModule
from models.loss_helper import *
from models.losses import *

from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

class PrototypeNet(nn.Module):

    def __init__(self, config, lbl_values, ign_lbls):
        super().__init__()

        self.C = len(lbl_values) - len(ign_lbls)

        # Backbone point feature learning
        self.backbone_net = KPFCNN(config, lbl_values, ign_lbls)

        # Hough voting
        self.vgen = VotingModule(vote_factor=1, seed_feature_dim=config.first_features_dim)


        self.pre_train = config.pre_train 


        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Define losses
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)


        self.point_semantic_classification_loss = torch.tensor(0)
        self.point_objectness_loss = torch.tensor(0)
        self.point_center_regression_loss = torch.tensor(0)
        self.proposal_objectness_loss = torch.tensor(0)
        self.proposal_semantic_loss = torch.tensor(0)
        self.proposal_agg_feature_loss = torch.tensor(0)

        self.instance_loss = torch.tensor(0)
        self.variance_loss = torch.tensor(0)

        self.max_instance_points = 0



    def forward(self, batch):
        point_positions = batch.points[0].clone().detach()

        t_start_backbone_forward_pass = time.time()
        point_semantic_classes, point_objectness_scores, point_features = self.backbone_net(batch)
        t_end_backbone_forward_pass = time.time()
        duration_backbone_forward_pass = t_end_backbone_forward_pass - t_start_backbone_forward_pass
        print('duration_backbone_forward_pass: ' + str(duration_backbone_forward_pass))

        # Reshape to have a minibatch size of 1
        point_features = torch.transpose(point_features, 0, 1)
        point_features = point_features.unsqueeze(0)
        point_positions = point_positions.unsqueeze(0)

        t_start_vgen_forward_pass = time.time()
        point_votes, point_votes_features = self.vgen(point_positions, point_features)
        t_end_vgen_forward_pass = time.time()
        duration_vgen_forward_pass = t_end_vgen_forward_pass - t_start_vgen_forward_pass
        print('duration_vgen_forward_pass: ' + str(duration_vgen_forward_pass))

        return point_semantic_classes, point_votes, point_objectness_scores


    def loss(self, point_semantic_classes, point_votes, point_objectness_scores, labels, ins_labels, centers_gt):
        
        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        point_semantic_classes = torch.transpose(point_semantic_classes, 0, 1)
        point_semantic_classes = point_semantic_classes.unsqueeze(0)
        target = target.unsqueeze(0)

        # PER-POINT SEMANTIC CLASSIFICATION LOSS (cross entropy loss)
        self.point_semantic_classification_loss = self.criterion(point_semantic_classes, target)

        # PER-POINT OBJECTNESS LOSS
        weights = (centers_gt[:, 0] > 0) * 99 + (centers_gt[:, 0] >= 0) * 1
        self.point_objectness_loss = weighted_mse_loss(point_objectness_scores.squeeze(), centers_gt[:, 0], weights)

        # Count maximal number of points that belongs to an instance
        #sem_ins_labels = torch.unique(ins_labels)
        #for _, semins in enumerate(sem_ins_labels):
        #    valid_ind = torch.where(ins_labels == semins)[0]
        #    if semins == 0:
        #        continue
        #    if valid_ind.shape[0] > self.max_instance_points:
        #        self.max_instance_points = valid_ind.shape[0]
        #        print("max_ins_points: ", self.max_instance_points)

        if not self.pre_train:
            # get ids of points that belongs to an object
            object_points_ids = torch.where(ins_labels != 0)[0]

            if object_points_ids.shape[0] > 0:
                # Get the GT object center points
                object_centers_gt = centers_gt[object_points_ids, 4:7]

                # Reshape object_centers_gt to have a minibatch size of 1
                object_centers_gt = object_centers_gt.unsqueeze(0)

                # PER-POINT CENTER REGRESSION LOSS (3D-MPA)
                # Only consider points that belongs to an object
                point_votes = point_votes[:, object_points_ids, :]
                point_votes_gt = object_centers_gt
                huber_loss = torch.nn.SmoothL1Loss()
                point_center_regression_loss = huber_loss(point_votes.double(), point_votes_gt)
                self.point_center_regression_loss = point_center_regression_loss
            
            else:
                self.point_center_regression_loss = torch.tensor(0)


        # COMBINED LOSS
        return self.point_semantic_classification_loss + self.point_objectness_loss + self.point_center_regression_loss


    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total
    
    
    def ins_pred_in_time_mpa(self, point_semantic_classes, point_votes, next_ins_id):
        """
        Calculate instance probabilities for each point with considering old predictions and using MPA
        Points can be assigned to multiple proposals
        :return: instance ids for all points, and new instances and next available ins_id
        """

        # IMPORTANT:
        # In our final two stage method, only the aggregation is performed here.
        # The proposal generation should be performed by the network.

        point_votes = point_votes.squeeze()

        point_semantic_classes = point_semantic_classes.cpu().detach().numpy()
        point_votes = point_votes.cpu().detach().numpy()

        ins_prediction = np.zeros(point_semantic_classes.shape)

        proposals = []

        ins_id = next_ins_id

        ins_idxs = np.where((point_semantic_classes < 9) & (point_semantic_classes != 0))[0]
        if len(ins_idxs) == 0:
            return

        ins_point_votes = point_votes[ins_idxs]

        number_proposals = 500
        print('number_proposals' + str(number_proposals))
        radius = 0.6
        #print('radius' + str(radius))
        epsilon = 0.8
        print('epsilon' + str(epsilon))
        minpts = 1
        use_majority_voting = True
        parallel = False

        # Random proposal center selection
        #proposals_ids = np.random.choice(ins_idxs, number_proposals)
        #print("Random")

        #FPS for proposal center selection
        proposals_ids = fps(ins_point_votes, number_proposals)
        proposals_ids = ins_idxs[proposals_ids]
        #print("FPS")

        i = 0

        # ToDo: Parellize Proposal Generation
        t_start_prop_generation = time.time()

        proposals_center = point_votes[proposals_ids]
        point_tree = cKDTree(ins_point_votes)
        shape = point_votes.shape[0]
        if not parallel:
            for i in range(number_proposals):
                proposal_center = proposals_center[i]
                               
                #with threadpool_limits(limits=1, user_api='blas'):
                #    associated_points_ids = point_tree.query_ball_point(proposal_center, radius)
                associated_points_ids = point_tree.query_ball_point(proposal_center, radius)

                '''
                distances = np.sum((ins_point_votes - proposal_center)**2, axis=1)
                distances = np.sqrt(distances)
                associated_points_ids = np.where(distances <= radius)[0]
                '''

                associated_points_ids = ins_idxs[associated_points_ids]

                proposals.append(associated_points_ids)

                #proposal = np.zeros(shape, np.int8)
                #proposal[associated_points_ids] = 1
                #proposals.append(proposal)
        else:
            '''
            def proposal_generation(proposal_center):
                #print(getpid())
                #with threadpool_limits(limits=1, user_api='blas'):
                    #associated_points_ids = point_tree.query_ball_point(proposal_center, radius)
                associated_points_ids = point_tree.query_ball_point(proposal_center, radius)
                associated_points_ids = ins_idxs[associated_points_ids]
                #proposal = np.zeros(shape, np.int8)
                #proposal[associated_points_ids] = 1
                #return proposal 
                return associated_points_ids
            '''
            #proposals = Parallel(n_jobs = 1)(delayed(proposal_generation)(proposals_center[i]) for i in range(number_proposals))
            proposals = Parallel(n_jobs = 4)(delayed(proposal_generation)(proposals_center[i], point_tree, radius, ins_idxs, shape) for i in range(number_proposals))
            #proposals = Parallel(n_jobs = 4, prefer="threads")(delayed(proposal_generation)(proposals_center[i], point_tree, radius, ins_idxs, shape) for i in range(number_proposals))
            #with multiprocessing.Pool(8) as pool:
                #proposals = pool.map(proposal_generation, range(0,number_proposals))
                #input_items = ((proposals_center[i], point_tree, radius, ins_idxs, shape) for i in range(number_proposals))
                #proposals = pool.starmap(proposal_generation, input_items)
                #prop_generation = partial(proposal_generation, point_tree=point_tree, radius=radius, ins_idxs=ins_idxs, shape=shape)
                #proposals = pool.map(prop_generation, proposals_center[i] for i in range(number_proposals))


        t_finish_prop_generation = time.time()
        duration_prop_generation = t_finish_prop_generation - t_start_prop_generation
        print('duration_prop_generation: ' + str(duration_prop_generation))
        
        # Proposal Clustering with DBScan
        clustering_inputs = proposals_center
        t_start_clustering = time.time()
        clustering_outputs = DBSCAN(eps=epsilon, min_samples=minpts).fit_predict(clustering_inputs)
        t_finish_clustering = time.time()
        duration_clustering = t_finish_clustering - t_start_clustering
        print('duration_clustering: ' + str(duration_clustering))


        new_ins_id = ins_id

        t_start_id_assoc = time.time()
        for index, label in np.ndenumerate(clustering_outputs):
            if label >= 0: # otherwise outlier
                #proposal_ids = np.where((proposals[index[0]] == 1))
                #proposal_ids = np.where((proposals[index[0]] == 1) & (ins_prediction == 0))
                proposal_ids = proposals[index[0]]
                proposal_ids = proposal_ids[0:5000]
                ins_prediction[proposal_ids] = ins_id + label
                if ins_id + label > new_ins_id:
                    new_ins_id = ins_id + label
        t_finish_id_assoc = time.time()
        duration_id_assoc = t_finish_id_assoc - t_start_id_assoc
        print('duration_id_assoc: ' + str(duration_id_assoc))

        # majority voting
        if use_majority_voting:
            t_start_mv = time.time()
            for i in range(ins_id, new_ins_id + 1):
                instance_ids = np.where(ins_prediction == i)
                if instance_ids[0].size == 0:
                    continue
                point_semantic_classes_current_instance = point_semantic_classes[instance_ids]
                bincount = np.bincount(point_semantic_classes_current_instance)
                most_frequent = bincount.argmax()
                point_semantic_classes[instance_ids] = most_frequent
            t_finish_mv = time.time()
            duration_mv = t_finish_mv - t_start_mv
            print('duration_mv :' + str(duration_mv))


        new_ins_id += 1

        #return ins_prediction, new_ins_id, np.asarray(proposals), np.asarray(proposals_center), point_semantic_classes
        return ins_prediction, new_ins_id, proposals, proposals_center, point_semantic_classes

    
    def ins_pred_in_time_mpa_bb(self, point_semantic_classes, point_votes, point_positions, next_ins_id):
        """
        Calculate instance probabilities for each point with considering old predictions and use the boudning boxes to aggregate proposals
        Points can be assigned to multiple proposals
        :return: instance ids for all points, and new instances and next available ins_id
        """

        # IMPORTANT:
        # In our final two stage method, only the aggregation is performed here.
        # The proposal generation should be performed by the network.

        point_votes = point_votes.squeeze()

        point_semantic_classes = point_semantic_classes.cpu().detach().numpy()
        point_votes = point_votes.cpu().detach().numpy()
        point_positions = point_positions.cpu().detach().numpy()

        ins_prediction = np.zeros(point_semantic_classes.shape)

        proposals = []
        proposal = np.zeros(ins_prediction.shape, np.int8)
        proposals_center = []
        
        final_objects_ids = []
        proposals_associated_ids = []


        ins_idxs = np.where((point_semantic_classes < 9) & (point_semantic_classes != 0))[0]
        if len(ins_idxs) == 0:
            return

        ins_point_votes = point_votes[ins_idxs]

        number_proposals = 400
        radius = 0.6
        iou_treshold = 0.1

        proposals_ids = np.random.choice(ins_idxs, number_proposals)

        i = 0

        while i < number_proposals:
            proposal_center = point_votes[proposals_ids[i]]
            point_tree = cKDTree(ins_point_votes)
            associated_points_ids = point_tree.query_ball_point(proposal_center, radius)
            associated_points_ids = ins_idxs[associated_points_ids]

            proposals_associated_ids.append(associated_points_ids)

            proposals_center.append(proposal_center)
            proposal_help = proposal.copy()
            proposal_help[associated_points_ids] = 1
            proposals.append(proposal_help)
        
            i += 1


        for i1, proposal_associated_ids in enumerate(proposals_associated_ids):
            if i1 == 0:
                final_objects_ids.append(proposal_associated_ids)
                continue
            bb1_points = point_positions[proposal_associated_ids]
            merged = False
            for i2, final_object_ids in enumerate(final_objects_ids):
                bb2_points = point_positions[final_object_ids]
                iou = get_iou(bb1_points, bb2_points)
                if iou >= iou_treshold:
                    final_objects_ids[i2] = np.concatenate((final_object_ids, proposal_associated_ids))
                    merged = True
                    break
            if not merged:
                final_objects_ids.append(proposal_associated_ids)


        prev_ins_id = next_ins_id
        for i, instance_ids in enumerate(final_objects_ids):
            ins_prediction[instance_ids] = next_ins_id
            next_ins_id += 1


        # majority voting
        for i in range(prev_ins_id, next_ins_id + 1):
            instance_ids = np.where(ins_prediction == i)
            if instance_ids[0].size == 0:
                continue
            point_semantic_classes_current_instance = point_semantic_classes[instance_ids]
            bincount = np.bincount(point_semantic_classes_current_instance)
            most_frequent = bincount.argmax()
            point_semantic_classes[instance_ids] = most_frequent


        return ins_prediction, next_ins_id, np.asarray(proposals), np.asarray(proposals_center), point_semantic_classes


def get_iou(bb1_points, bb2_points):
    bb1_x1 = np.min(bb1_points[:, 0])
    bb1_y1 = np.min(bb1_points[:, 1])
    bb1_z1 = np.min(bb1_points[:, 2])
    bb1_x2 = np.max(bb1_points[:, 0])
    bb1_y2 = np.max(bb1_points[:, 1])
    bb1_z2 = np.max(bb1_points[:, 2])

    bb2_x1 = np.min(bb2_points[:, 0])
    bb2_y1 = np.min(bb2_points[:, 1])
    bb2_z1 = np.min(bb2_points[:, 2])
    bb2_x2 = np.max(bb2_points[:, 0])
    bb2_y2 = np.max(bb2_points[:, 1])
    bb2_z2 = np.max(bb2_points[:, 2])

    x1 = max(bb1_x1, bb2_x1)
    y1 = max(bb1_y1, bb2_y1)
    z1 = max(bb1_z1, bb2_z1)
    x2 = min(bb1_x2, bb2_x2)
    y2 = min(bb1_y2, bb2_y2)
    z2 = min(bb1_z2, bb2_z2)

    if x2 < x1 or y2 < y1 or z2 < z1:
        return 0

    intersection_area = (x2 - x1) * (y2 - y1) * (z2 - z1)

    bb1_area = (bb1_x2 - bb1_x1) * (bb1_y2 - bb1_y1) * (bb1_z2 - bb1_z1)
    bb2_area = (bb2_x2 - bb2_x1) * (bb2_y2 - bb2_y1) * (bb2_z2 - bb2_z1)

    return intersection_area / (bb1_area + bb2_area - intersection_area + 1e-6)

def fps(pts, K):
    farthest_pts = np.zeros(K, dtype=int)
    farthest_pts[0] = np.random.randint(len(pts))
    distances = calc_distances(pts[farthest_pts[0]], pts)
    for i in range(1, K):
        farthest_pts[i] = np.argmax(distances)
        distances = np.minimum(distances, calc_distances(pts[farthest_pts[i]], pts))
    return farthest_pts

def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)

#'''
def proposal_generation(proposal_center, point_tree, radius, ins_idxs, shape):
    #print(getpid())
    with threadpool_limits(limits=1, user_api='blas'):
        associated_points_ids = point_tree.query_ball_point(proposal_center, radius)
    #associated_points_ids = point_tree.query_ball_point(proposal_center, radius)
    associated_points_ids = ins_idxs[associated_points_ids]
    #proposal = np.zeros(shape, np.int8)
    #proposal[associated_points_ids] = 1
    #return proposal
    return associated_points_ids
#'''