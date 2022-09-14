from sympy import false
import torch
import torch.nn as nn
import numpy as np
from os.path import exists, join

from models.backbone_module_binary import KPFCNN
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.loss_helper import *
from models.losses import *
from models.blocks import UnaryBlock
from models.focal_loss import FocalLoss2
from models.mask_module import MaskModule

from sklearn.cluster import DBSCAN

class MPAnetBinary(nn.Module):

    def __init__(
            self,
            config,
            lbl_values,
            ign_lbls,
            num_proposal=500,
            radius=0.6,
            nsample=5000,
            nsample_sub=5000,
            use_binary_mask_in_proposal_module=False,
            use_fps=True,                               # FPS or Random
            use_geo_features=True):                     # Geometric-Features or Embedding-Features

        super().__init__()

        print('NumberProposal: ' + str(num_proposal))
        print('NumberSample: ' + str(nsample))
        print('NumberSampleSub: ' + str(nsample_sub))

        self.num_proposal = num_proposal
        self.nsample = nsample
        self.use_binary_mask_in_proposal_module = use_binary_mask_in_proposal_module
        self.use_geo_features = use_geo_features

        self.C = len(lbl_values) - len(ign_lbls)

        # Backbone point feature learning
        self.backbone_net = KPFCNN(config, lbl_values, ign_lbls)

        # Hough voting
        self.vgen = VotingModule(vote_factor=1, seed_feature_dim=config.first_features_dim)

        # Vote aggregation and proposal generation
        self.pnet = ProposalModule(8, num_proposal, config.first_features_dim, radius, nsample, nsample_sub, use_fps, use_geo_features, use_binary_mask_in_proposal_module)

        # Semantic classes for background points (FP background points)
        self.sem_background_net = UnaryBlock(config.first_features_dim, 11, False, 0)

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
        self.fp_points_sem_loss = torch.tensor(0) 
        self.proposal_agg_feature_loss = torch.tensor(0)
        self.proposal_mask_loss = torch.tensor(0)

        self.instance_loss = torch.tensor(0)
        self.variance_loss = torch.tensor(0)


    def forward(self, batch):
        point_positions = batch.points[0].clone().detach()

        point_semantic_classes_binary, point_objectness_scores, point_features = self.backbone_net(batch)

        # Reshape to have a minibatch size of 1
        point_features = torch.transpose(point_features, 0, 1)
        point_features = point_features.unsqueeze(0)
        point_positions = point_positions.unsqueeze(0)

        normalization = True
        # Need to be consistent during training and testing
        if normalization:
            point_votes, point_votes_features = self.vgen(point_positions, point_features)
            point_votes_features_norm = torch.norm(point_votes_features, p=2, dim=1)
            point_votes_features = point_votes_features.div(point_votes_features_norm.unsqueeze(1))
        else:
            point_votes, _ = self.vgen(point_positions, point_features)
            point_votes_features = point_features.contiguous()


        # only consider vote points from objects as possible proposal center
        selected_point_semantic_classes = torch.argmax(point_semantic_classes_binary.data, dim=1)
        object_points_ids = torch.where(selected_point_semantic_classes == 1)[0]
        object_point_votes = point_votes[:, object_points_ids, :]
        object_point_votes_features = point_votes_features[:, :, object_points_ids]
        # handle the case that all points are predicted to be background
        if object_point_votes.shape[1] == 0:
            print("All points are predicted to be background.")
            object_point_votes = point_votes
            object_point_votes_features = point_votes_features
            object_points_ids = torch.where(selected_point_semantic_classes >= 0)[0]
            # In this case simply conisder all points as possible proposal points


        if not self.pre_train:
            #proposal_semantic_classes, proposal_aggregation_features, proposal_objectness_scores, proposal_positions, proposal_idx, proposal_features = self.pnet(object_point_votes, object_point_votes_features)
            proposal_binary_mask, proposal_semantic_classes, proposal_aggregation_features, proposal_objectness_scores, proposal_positions, proposal_idx, proposal_features = self.pnet(object_point_votes, object_point_votes_features)
        else:
            batch_size = 1
            number_proposals = 500 #1
            n_sample = 5000 #1
            proposal_semantic_classes = torch.zeros(batch_size, self.C, number_proposals)
            proposal_aggregation_features = torch.zeros(batch_size, 4, number_proposals)
            proposal_objectness_scores = torch.zeros(batch_size, 2, number_proposals)
            proposal_positions = torch.zeros(batch_size, number_proposals, 3)
            proposal_idx = torch.zeros(batch_size, number_proposals, n_sample, dtype=torch.int32)

        proposal_ids = object_points_ids[proposal_idx.long()[:]]

        fp_points_features = point_features
        fp_points_features = fp_points_features.squeeze()
        fp_points_features = torch.transpose(fp_points_features, 0, 1)
        fp_points_sem_classes = self.sem_background_net(fp_points_features)
        
        return proposal_binary_mask, fp_points_sem_classes, proposal_semantic_classes, proposal_aggregation_features, proposal_objectness_scores, proposal_positions, proposal_ids, point_semantic_classes_binary, point_votes, point_objectness_scores


    def loss(self, proposal_binary_mask, point_semantic_classes_binary, point_votes, point_objectness_scores, proposal_ids, proposal_objectness_scores, proposal_positions, proposal_semantic_classes, proposal_aggregation_features, fp_points_sem_classes, labels, ins_labels, centers_gt, points=None, times=None):
        """
        Runs the loss on outputs of the model
        :param point_semantic_classes: logits
        :param labels: labels
        :return: loss
        """

        self.point_semantic_classification_loss = torch.tensor(0)
        self.point_objectness_loss = torch.tensor(0)
        self.point_center_regression_loss = torch.tensor(0)
        self.proposal_objectness_loss = torch.tensor(0)
        self.proposal_semantic_loss = torch.tensor(0)
        self.fp_points_sem_loss = torch.tensor(0)
        self.proposal_agg_feature_loss = torch.tensor(0)
        self.proposal_mask_loss = torch.tensor(0)

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i
        
        target = target.unsqueeze(0)

        fp_target = - torch.ones_like(labels)
        for i in range(9,20):
            fp_target[labels == i] = i - 9
        
        fp_target = fp_target.unsqueeze(0)

        fp_points_sem_classes = torch.transpose(fp_points_sem_classes, 0, 1)
        fp_points_sem_classes = fp_points_sem_classes.unsqueeze(0)

        binary_target = - torch.ones_like(labels)
        binary_target[(labels >= 1) & (labels <= 8)] = 1
        binary_target[(labels >= 9) & (labels <= 19)] = 0

        binary_target = binary_target.unsqueeze(0)

        point_semantic_classes_binary = torch.transpose(point_semantic_classes_binary, 0, 1)
        point_semantic_classes_binary = point_semantic_classes_binary.unsqueeze(0)

        
        # get ids of points that belongs to an object
        object_points_ids = torch.where(ins_labels != 0)[0]

        if True:
        #if self.pre_train:
            # PER-POINT SEMANTIC CLASSIFICATION LOSS (cross entropy loss)
            self.point_semantic_classification_loss = self.criterion(point_semantic_classes_binary, binary_target)

            # PER-POINT OBJECTNESS LOSS
            # Not part of 3D-MPA or VoteNet, but used in 4D-PLS
            # Important to still use it because the per-point objectness scores are used to merge the frames together (4D-Volume Formation)
            weights = (centers_gt[:, 0] > 0) * 99 + (centers_gt[:, 0] >= 0) * 1
            self.point_objectness_loss = weighted_mse_loss(point_objectness_scores.squeeze(), centers_gt[:, 0], weights)

            
            if object_points_ids.shape[0] > 0:
                # Get the GT object center points
                object_centers_gt = centers_gt[object_points_ids, 4:7]
                # When we use the offset, we get rounding errors
                # point_offset_gt = centers_gt[:, 1:4]
                # object_centers_gt = points[object_points_ids] + point_offset_gt[object_points_ids]
                # Because of this, we directly use the centers

                # Reshape object_centers_gt and object_radius_gt to have a minibatch size of 1
                object_centers_gt = object_centers_gt.unsqueeze(0)

                # PER-POINT CENTER REGRESSION LOSS (3D-MPA)
                # Only consider points that belongs to an object
                point_votes = point_votes[:, object_points_ids, :]
                point_votes_gt = object_centers_gt
                # L1-Loss
                # l1_loss = torch.nn.L1Loss(reduction='mean')
                # point_center_regression_loss = l1_loss(point_votes, point_votes_gt)
                # Huber-Loss (How to set beta????) Standard: beta = 1
                huber_loss = torch.nn.SmoothL1Loss()
                point_center_regression_loss = huber_loss(point_votes.double(), point_votes_gt)
                self.point_center_regression_loss = point_center_regression_loss
            

        if not self.pre_train:
            if object_points_ids.shape[0] > 0:
                # Get the GT object center points
                object_centers_gt = centers_gt[object_points_ids, 4:7]
                # When we use the offset, we get rounding errors
                # point_offset_gt = centers_gt[:, 1:4]
                # object_centers_gt = points[object_points_ids] + point_offset_gt[object_points_ids]
                # Because of this, we directly use the centers

                # Get the GT object radius
                object_radius_gt = centers_gt[object_points_ids, 7:8]
                
                # Get the GT bb size
                object_bb_sizes_gt = centers_gt[object_points_ids, 8:11]

                # Reshape object_centers_gt and object_radius_gt to have a minibatch size of 1
                object_centers_gt = object_centers_gt.unsqueeze(0)
                object_radius_gt = object_radius_gt.unsqueeze(0)
                object_bb_sizes_gt = object_bb_sizes_gt.unsqueeze(0)

                # We want to know the center, the radius und the semantic classes of the objects later on
                #object_centers_pos_rad_sem_gt = torch.cat((object_centers_gt, object_radius_gt, target[:, object_points_ids].unsqueeze(2).double()), 2)
                object_centers_pos_rad_sem_gt = torch.cat((object_centers_gt, object_radius_gt, object_bb_sizes_gt, target[:, object_points_ids].unsqueeze(2).double()), 2)

                # Would now have many duplicates. Remove duplicates
                object_centers_pos_rad_sem_gt = torch.unique(object_centers_pos_rad_sem_gt, dim=1)

                # Remove objects with less than 5 points
                help_radius_gt = object_centers_pos_rad_sem_gt[:, :, 3]
                help_radius_gt = help_radius_gt.squeeze()
                help_ids = torch.where(help_radius_gt != 0)[0]
                object_centers_pos_rad_sem_gt = object_centers_pos_rad_sem_gt[:, help_ids, :]

                if help_ids.shape[0] > 0:
                        
                    # PROPOSAL OBJECTNESS LOSS (VoteNet)
                    proposal_objectness_loss, proposal_objectness_label, proposal_objectness_mask, proposal_objectness_assignment = compute_objectness_loss(proposal_positions, object_centers_pos_rad_sem_gt[:, :, 0:3], proposal_objectness_scores)
                    #self.proposal_objectness_loss = proposal_objectness_loss

                    # PROPOSAL SEMANTIC LOSS (like in 3D-MPA and VoteNet)
                    #objects_sem_gt = object_centers_pos_rad_sem_gt[:, :, 4].long()
                    objects_sem_gt = object_centers_pos_rad_sem_gt[:, :, 7].long()
                    proposal_sem_gt = torch.gather(objects_sem_gt, 1, proposal_objectness_assignment)  # select (B,K) from (B,K2)
                    #criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
                    ###can get rid of objectness-score with the line below (included in semantic-loss)
                    ###proposal_sem_gt = torch.where(proposal_objectness_label == 1, proposal_sem_gt, 8) #8 is background label
                    criterion_sem_cls = nn.CrossEntropyLoss()
                    proposal_semantic_loss = criterion_sem_cls(proposal_semantic_classes, proposal_sem_gt)  # (B,K)
                    #proposal_semantic_loss = torch.sum(proposal_semantic_loss * proposal_objectness_label) / (torch.sum(proposal_objectness_label) + 1e-6)
                    self.proposal_semantic_loss = proposal_semantic_loss

                    # FP-POINTS-SEMANTIC-LOSS
                    criterion_sem_fp = torch.nn.CrossEntropyLoss(ignore_index=-1)
                    self.fp_points_sem_loss = criterion_sem_fp(fp_points_sem_classes, fp_target)

                    # AGGREGATION FEATURE LOSS (3D-MPA)
                    #object_centers_pos_rad_gt = object_centers_pos_rad_sem_gt[:, :, 0:4]
                    object_centers_pos_rad_gt = object_centers_pos_rad_sem_gt[:, :, 0:7]
                    center_rad_gt_closest_object = object_centers_pos_rad_gt[:, proposal_objectness_assignment.squeeze(), :] # would not work like this if batch_size > 1
                    center_gt_closest_object = center_rad_gt_closest_object[:, :, 0:3]
                    radius_gt_closest_object = center_rad_gt_closest_object[:, :, 3]
                    bb_size_gt_closest_object = center_rad_gt_closest_object[:, :, 4:7]
                    proposal_aggregation_features = torch.transpose(proposal_aggregation_features, 1, 2)
                    refined_proposal_positions = proposal_positions + proposal_aggregation_features[:, :, 0:3]
                    huber_loss1 = torch.nn.SmoothL1Loss()
                    proposal_agg_feature_loss_1 = huber_loss1(refined_proposal_positions.double(), center_gt_closest_object)
                    print(proposal_agg_feature_loss_1)
                    proposal_radius = proposal_aggregation_features[:, :, 3]
                    huber_loss2 = torch.nn.SmoothL1Loss()
                    proposal_agg_feature_loss_2 = huber_loss2(proposal_radius.double(), radius_gt_closest_object)
                    print(proposal_agg_feature_loss_2)
                    proposal_bb_size = proposal_aggregation_features[:, :, 4:7]
                    huber_loss3 = torch.nn.SmoothL1Loss()
                    proposal_agg_feature_loss_3 = huber_loss3(proposal_bb_size.double(), bb_size_gt_closest_object)
                    print(proposal_agg_feature_loss_3)
                    #self.proposal_agg_feature_loss = proposal_agg_feature_loss_1 + proposal_agg_feature_loss_2
                    self.proposal_agg_feature_loss = proposal_agg_feature_loss_1 + proposal_agg_feature_loss_2 + proposal_agg_feature_loss_3
                    #self.proposal_agg_feature_loss = (proposal_agg_feature_loss_1 + proposal_agg_feature_loss_2 + proposal_agg_feature_loss_3) * 0.1
                    print(self.proposal_agg_feature_loss)

                    # MASK LOSS
                    if self.use_binary_mask_in_proposal_module:
                        proposal_points_sem_gt = torch.zeros(proposal_ids.shape, dtype=torch.int64).cuda()
                        proposal_points_sem_gt[:,:,:] = target[0, proposal_ids[:,:,:]]
                        proposal_sem_gt_help = proposal_sem_gt
                        proposal_sem_gt_help = proposal_sem_gt_help.unsqueeze(-1)
                        proposal_sem_gt_help = proposal_sem_gt_help.expand(-1,-1,proposal_ids.shape[2])
                        proposal_binary_mask_gt = torch.zeros(proposal_ids.shape, dtype=torch.int64).cuda()
                        proposal_binary_mask_gt[proposal_points_sem_gt == proposal_sem_gt_help] = 1
                        criterion_bin_mask_cls1 = nn.CrossEntropyLoss()
                        proposal_mask_loss1 = criterion_bin_mask_cls1(proposal_binary_mask, proposal_binary_mask_gt)
                        self.proposal_mask_loss = proposal_mask_loss1
                        print(self.proposal_mask_loss)

        # COMBINED LOSS
        loss = self.point_semantic_classification_loss + self.point_objectness_loss + self.point_center_regression_loss + self.proposal_objectness_loss + self.proposal_semantic_loss + self.fp_points_sem_loss + self.proposal_agg_feature_loss + self.proposal_mask_loss
        return loss
    
        
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
    
    def ins_pred_in_time_mpa(self, labels, proposals_binary_mask, fp_points_sem_classes, proposals_aggregation_features, proposals_semantic_classes, proposals_objectness_scores, proposals_positions, proposals_ids, point_positions, next_ins_id):
        """
        Calculate instance probabilities for each point with considering old predictions and using MPA
        Points can be assigned to multiple proposals
        :return: instance ids for all points, and new instances and next available ins_id
        """

        proposals_positions = proposals_positions.squeeze()
        proposals_ids = proposals_ids.squeeze()
        proposals_aggregation_features = proposals_aggregation_features.squeeze()
        proposals_aggregation_features = torch.transpose(proposals_aggregation_features, 0, 1)
        proposals_binary_mask = proposals_binary_mask.squeeze()
        proposals_binary_mask = torch.transpose(proposals_binary_mask, 0, 1)
        proposals_binary_mask = torch.transpose(proposals_binary_mask, 1, 2)

        proposals_positions = proposals_positions.cpu().detach().numpy()
        proposals_ids = proposals_ids.cpu().detach().numpy()
        proposals_aggregation_features = proposals_aggregation_features.cpu().detach().numpy()
        proposals_binary_mask = proposals_binary_mask.cpu().detach().numpy()

        ins_prediction = np.zeros(fp_points_sem_classes.shape, dtype=int)

        epsilon = 0.8
        print(epsilon)
        minpts = 1
        use_rp_r_bb = True            
        use_rp_bb = False            
        use_rp_r = False               
        use_rp = False                    
        use_p = False
        use_binary_mask = False
        use_majority_voting = True 

        point_semantic_classes_new = fp_points_sem_classes

        # Proposal Clustering with DBScan
        if use_rp_r_bb:
            print('use_rp_r_bb')
            refined_proposals_positions = proposals_aggregation_features[:, 0:3] + proposals_positions[:, 0:3]
            proposals_aggregation_features[:, 0:3] = refined_proposals_positions
            clustering_inputs = proposals_aggregation_features
        elif use_rp_r:
            print('use_rp_r') 
            refined_proposals_positions = proposals_aggregation_features[:, 0:3] + proposals_positions[:, 0:3]
            proposals_aggregation_features[:, 0:3] = refined_proposals_positions
            clustering_inputs = proposals_aggregation_features[:, 0:4]
        elif use_rp_bb:
            print('use_rp_bb') 
            refined_proposals_positions = proposals_aggregation_features[:, 0:3] + proposals_positions[:, 0:3]
            proposals_aggregation_features[:, 0:3] = refined_proposals_positions
            feature_ids = [0,1,2,4,5,6]
            clustering_inputs = proposals_aggregation_features[:, feature_ids]
        elif use_rp:
            print('use_rp')
            refined_proposals_positions = proposals_aggregation_features[:, 0:3] + proposals_positions[:, 0:3]
            clustering_inputs = refined_proposals_positions
        elif use_p:
            print('use_p')
            refined_proposals_positions = np.zeros((proposals_positions.shape[0], 3), dtype=int)
            clustering_inputs = proposals_positions
        clustering_outputs = DBSCAN(eps=epsilon, min_samples=minpts).fit_predict(clustering_inputs)

        prev_ins_id = next_ins_id       
        for index, label in np.ndenumerate(clustering_outputs):
            if label >= 0: # otherwise outlier
                proposal_ids = proposals_ids[index]
                if use_binary_mask:
                    print('use_binary_mask')
                    proposal_binary_mask = proposals_binary_mask[index]
                    negative_binary_mask = proposal_binary_mask[:, 0]
                    positive_binary_mask = proposal_binary_mask[:, 1]
                    negative_binary_mask_indexes = np.where(negative_binary_mask > positive_binary_mask)[0]
                    positive_binary_mask_indexes = np.where(negative_binary_mask <= positive_binary_mask)[0]
                    proposal_ids = proposal_ids[positive_binary_mask_indexes]
                ins_prediction[proposal_ids] = prev_ins_id + label
                point_semantic_classes_new[proposal_ids] = proposals_semantic_classes[index]
                if prev_ins_id + label > next_ins_id:
                    next_ins_id = prev_ins_id + label
                
        # majority voting
        if use_majority_voting:
            for i in range(prev_ins_id, next_ins_id + 1):
                instance_ids = np.where(ins_prediction == i)
                if instance_ids[0].size == 0:
                    continue
                point_semantic_classes_current_instance = point_semantic_classes_new[instance_ids]
                bincount = np.bincount(point_semantic_classes_current_instance)
                most_frequent = bincount.argmax()
                point_semantic_classes_new[instance_ids] = most_frequent
        
        next_ins_id = next_ins_id + 1 #!!!!!!

        return ins_prediction, next_ins_id, proposals_ids, proposals_positions, point_semantic_classes_new, refined_proposals_positions

    
    