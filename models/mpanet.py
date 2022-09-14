import torch
import torch.nn as nn
import numpy as np
from os.path import exists, join
import time

from models.backbone_module2 import KPFCNN
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.loss_helper import *
from models.losses import *
from models.blocks import UnaryBlock
from models.focal_loss import FocalLoss2
from models.mask_module import MaskModule

from sklearn.cluster import DBSCAN

class MPAnet(nn.Module):

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
            use_mask_module=False,
            use_background_in_proposals=False,
            use_fps=True,                               # FPS or Random
            use_geo_features=True,                      # Geometric-Features or Embedding-Features
            nuscene=False):        

        super().__init__()

        print('NumberProposal: ' + str(num_proposal))
        print('NumberSample: ' + str(nsample))
        print('NumberSampleSub: ' + str(nsample_sub))
        print('Radius: ' + str(radius))

        self.num_proposal = num_proposal
        self.nsample = nsample
        self.use_binary_mask_in_proposal_module = use_binary_mask_in_proposal_module
        self.use_mask_module = use_mask_module
        self.use_background_in_proposals = use_background_in_proposals
        self.use_geo_features = use_geo_features
        self.nuscene = nuscene

        self.C = len(lbl_values) - len(ign_lbls)

        # Backbone point feature learning
        self.backbone_net = KPFCNN(config, lbl_values, ign_lbls)

        # Hough voting
        self.vgen = VotingModule(vote_factor=1, seed_feature_dim=config.first_features_dim)

        # Vote aggregation and proposal generation
        if self.nuscene:
            self.pnet = ProposalModule(10, num_proposal, config.first_features_dim, radius, nsample, nsample_sub, use_fps, use_geo_features, use_binary_mask_in_proposal_module)
        else:
            self.pnet = ProposalModule(8, num_proposal, config.first_features_dim, radius, nsample, nsample_sub, use_fps, use_geo_features, use_binary_mask_in_proposal_module)

        # Semantic classes for background points (FP background points)
        if self.nuscene:
            self.sem_background_net = UnaryBlock(config.first_features_dim, 6, False, 0)
        else:
            self.sem_background_net = UnaryBlock(config.first_features_dim, 11, False, 0)

        if use_mask_module:
            self.mask_net = MaskModule(mlp=[config.first_features_dim + 128, 256, 128, 64, 32, 2])

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

        #t_start_backbone_forward_pass = time.time()
        point_semantic_classes, point_objectness_scores, point_features = self.backbone_net(batch)
        #t_end_backbone_forward_pass = time.time()
        #duration_backbone_forward_pass = t_end_backbone_forward_pass - t_start_backbone_forward_pass
        #print('duration_backbone_forward_pass: ' + str(duration_backbone_forward_pass))

        # Reshape to have a minibatch size of 1
        point_features = torch.transpose(point_features, 0, 1)
        point_features = point_features.unsqueeze(0)
        point_positions = point_positions.unsqueeze(0)

        #t_start_vgen_forward_pass = time.time()
        normalization = True
        # Need to be consistent during training and inference
        if normalization:
            point_votes, point_votes_features = self.vgen(point_positions, point_features)
            point_votes_features_norm = torch.norm(point_votes_features, p=2, dim=1)
            point_votes_features = point_votes_features.div(point_votes_features_norm.unsqueeze(1))
        else:
            point_votes, _ = self.vgen(point_positions, point_features)
            point_votes_features = point_features.contiguous()
        #t_end_vgen_forward_pass = time.time()
        #duration_vgen_forward_pass = t_end_vgen_forward_pass - t_start_vgen_forward_pass
        #print('duration_vgen_forward_pass: ' + str(duration_vgen_forward_pass))


        # Only consider vote points from objects as possible proposal center
        selected_point_semantic_classes = torch.argmax(point_semantic_classes.data, dim=1)
        if self.nuscene:
            object_points_ids = torch.where(selected_point_semantic_classes < 10)[0]
        else:
            object_points_ids = torch.where(selected_point_semantic_classes < 8)[0]
        object_point_votes = point_votes[:, object_points_ids, :]
        object_point_votes_features = point_votes_features[:, :, object_points_ids]
        # Handle the case that all points are predicted to be background
        if object_point_votes.shape[1] == 0:
            print("All points are predicted to be background.")
            object_point_votes = point_votes
            object_point_votes_features = point_votes_features
            object_points_ids = torch.where(selected_point_semantic_classes >= 0)[0]
            # In this case simply conisder all points as possible proposal points


        if not self.pre_train:
            #t_start_pnet_forward_pass = time.time()
            #proposal_semantic_classes, proposal_aggregation_features, proposal_objectness_scores, proposal_positions, proposal_idx, proposal_features = self.pnet(object_point_votes, object_point_votes_features)
            proposal_binary_mask, proposal_semantic_classes, proposal_aggregation_features, proposal_objectness_scores, proposal_positions, proposal_idx, proposal_features = self.pnet(object_point_votes, object_point_votes_features)
            #t_end_pnet_forward_pass = time.time()
            #duration_pnet_forward_pass = t_end_pnet_forward_pass - t_start_pnet_forward_pass
            #print('duration_pnet_forward_pass: ' + str(duration_pnet_forward_pass))
        else:
            batch_size = 1
            number_proposals = self.num_proposal
            n_sample = self.nsample
            proposal_binary_mask = torch.zeros(batch_size, 2, number_proposals, n_sample)
            proposal_semantic_classes = torch.zeros(batch_size, self.C, number_proposals)
            proposal_aggregation_features = torch.zeros(batch_size, 4, number_proposals)
            proposal_objectness_scores = torch.zeros(batch_size, 2, number_proposals)
            proposal_positions = torch.zeros(batch_size, number_proposals, 3)
            proposal_idx = torch.zeros(batch_size, number_proposals, n_sample, dtype=torch.int32)

        proposal_ids = object_points_ids[proposal_idx.long()[:]]

        if self.use_mask_module:
            print('self.use_mask_module')
            if self.use_background_in_proposals:
                print('self.use_background_in_proposals')
                # Maybe also an option to check over all points if they are in the proposal BB
                # Would then not only consider the background points for refinement, but also the foreground points
                proposal_ids_with_background = torch.zeros((1, self.num_proposal, self.nsample), dtype=int).cuda()
                proposal_points_features = torch.zeros((1, 259, self.num_proposal, self.nsample)).cuda()
                background_points_ids = torch.where(selected_point_semantic_classes >= 8)[0]
                background_points = point_positions[0, background_points_ids]
                index = 0
                for prop_ids in proposal_ids[0]:
                    prop_ids = torch.unique(prop_ids) 
                    prop_points = point_positions[0, prop_ids]
                    x_min = torch.min(prop_points[:, 0])
                    y_min = torch.min(prop_points[:, 1])
                    z_min = torch.min(prop_points[:, 2])
                    x_max = torch.max(prop_points[:, 0])
                    y_max = torch.max(prop_points[:, 1])
                    z_max = torch.max(prop_points[:, 2])
                    background_points_in_proposal_ids = torch.where((x_min<=background_points[:,0])&(x_max>=background_points[:,0])&(y_min<=background_points[:,1])&(y_max>=background_points[:,1])&(z_min<=background_points[:,2])&(z_max>=background_points[:,2]))[0]
                    background_points_in_proposal_ids = background_points_ids[background_points_in_proposal_ids]
                    #print(background_points_in_proposal_ids.shape[0])
                    prop_ids = torch.cat((prop_ids, background_points_in_proposal_ids))
                    prop_ids = prop_ids[0:5000]
                    prop_ids_help = torch.full((self.nsample, 1), prop_ids[0], dtype=int).cuda()
                    prop_ids_help = prop_ids_help.squeeze()
                    prop_ids_help[0:prop_ids.shape[0]] = prop_ids
                    prop_ids = prop_ids_help
                    prop_points = point_positions[:, prop_ids]
                    prop_points = torch.transpose(prop_points, 1, 2)
                    prop_features = point_features[:, :, prop_ids]
                    prop_features = torch.cat((prop_points, prop_features), dim=1)
                    proposal_ids_with_background[:, index, :] = prop_ids.unsqueeze(0)
                    proposal_points_features[:, :, index, :] = prop_features
                    index += 1
                proposal_ids = proposal_ids_with_background
            else:
                proposal_points_positions = point_positions[:, proposal_ids[0], :]
                proposal_points_positions = torch.transpose(proposal_points_positions, 2, 3)
                proposal_points_positions = torch.transpose(proposal_points_positions, 1, 2)
                proposal_points_features = point_features[:, :, proposal_ids[0]]
                proposal_points_features = torch.cat((proposal_points_positions, proposal_points_features), dim=1)
                # The proposal_points_features are not the same as the proposal_points_features which are input to the mask-net in the pointnet2_modules.py
                # Reason: Used the point_votes and point_votes_features in pointnet2_modules.py
                # Could also use the point_votes and point_votes_features here
                # But would make no sense for self.use_background_in_proposals

            proposal_features_help = proposal_features.unsqueeze(-1)
            proposal_features_help = proposal_features_help.expand(-1,-1,-1,self.nsample)
            grouped_features_cat = torch.cat([proposal_points_features, proposal_features_help], dim=1)
            proposal_binary_mask = self.mask_net(grouped_features_cat)

        fp_points_features = point_features
        fp_points_features = fp_points_features.squeeze()
        fp_points_features = torch.transpose(fp_points_features, 0, 1)
        fp_points_sem_classes = self.sem_background_net(fp_points_features)
        
        return proposal_binary_mask, fp_points_sem_classes, proposal_semantic_classes, proposal_aggregation_features, proposal_objectness_scores, proposal_positions, proposal_ids, point_semantic_classes, point_votes, point_objectness_scores


    def loss(self, proposal_binary_mask, point_semantic_classes, point_votes, point_objectness_scores, proposal_ids, proposal_objectness_scores, proposal_positions, proposal_semantic_classes, proposal_aggregation_features, fp_points_sem_classes, labels, ins_labels, centers_gt, points=None, times=None):
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

        # Reshape to have a minibatch size of 1
        point_semantic_classes = torch.transpose(point_semantic_classes, 0, 1)
        point_semantic_classes = point_semantic_classes.unsqueeze(0)
        target = target.unsqueeze(0)

        fp_target = - torch.ones_like(labels)
        if self.nuscene:
            for i in range(11,17):
                fp_target[labels == i] = i - 11
        else:
            for i in range(9,20):
                fp_target[labels == i] = i - 9
        fp_target = fp_target.unsqueeze(0)

        fp_points_sem_classes = torch.transpose(fp_points_sem_classes, 0, 1)
        fp_points_sem_classes = fp_points_sem_classes.unsqueeze(0)


        # get ids of points that belongs to an object
        if self.nuscene:
            object_points_ids = torch.where((target.squeeze() < 10) & (target.squeeze() != -1))[0]
        else:
            # object_points_ids = torch.where((labels < 9) & (labels != 0))[0]
            object_points_ids = torch.where(ins_labels != 0)[0]
            # object_points_ids = torch.where((target.squeeze() < 8) & (target.squeeze() != -1))[0]
            # object_points_ids = torch.where((centers_gt[:, 0] != 0) | (centers_gt[:, 1] != 0) | (centers_gt[:, 2] != 0) | (centers_gt[:, 3] != 0))[0] # hier noch Probleme: Sind auch Punkte dabei, die nicht zu Objekten gehören
            # Also need to exclude the ids where all centers_gt are 0. Reason: For object with <5 points, no centers_gt are computed
            # object_points_ids = torch.where((target.squeeze() < 8) & (target.squeeze() != -1) & ((centers_gt[:, 0] != 0) | (centers_gt[:, 1] != 0) | (centers_gt[:, 2] != 0) | (centers_gt[:, 3] != 0)))[0]

        #if True:
        if self.pre_train:
            # PER-POINT SEMANTIC CLASSIFICATION LOSS (cross entropy loss)
            self.point_semantic_classification_loss = self.criterion(point_semantic_classes, target)

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

                # Reshape object_centers_gt to have a minibatch size of 1
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

                # We want to know the center, the radius and the semantic classes of the objects later on
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
                    criterion_sem_cls = nn.CrossEntropyLoss()#!!!
                    proposal_semantic_loss = criterion_sem_cls(proposal_semantic_classes, proposal_sem_gt)#!!!  # (B,K) 
                    #proposal_semantic_loss = torch.sum(proposal_semantic_loss * proposal_objectness_label) / (torch.sum(proposal_objectness_label) + 1e-6)
                    self.proposal_semantic_loss = proposal_semantic_loss#!!!

                    # FP-POINTS-SEMANTIC-LOSS
                    #criterion_sem_fp = torch.nn.CrossEntropyLoss(ignore_index=-1)#!!!
                    #self.fp_points_sem_loss = criterion_sem_fp(fp_points_sem_classes, fp_target)#!!!

                    # AGGREGATION FEATURE LOSS (3D-MPA)
                    if self.use_geo_features:
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
                        self.proposal_agg_feature_loss = self.proposal_agg_feature_loss
                        print(self.proposal_agg_feature_loss)
                    else:
                        sigma_var = sigma_dist = 0.1
                        gamma = 0.001
                        l1_loss = torch.nn.L1Loss()
                        proposal_aggregation_features = torch.transpose(proposal_aggregation_features, 1, 2)
                        number_of_gt_objects = object_centers_pos_rad_sem_gt.shape[1]
                        objects_mean = torch.zeros(1,number_of_gt_objects,5).cuda()
                        for i in range(0, number_of_gt_objects):
                            ids = torch.where(proposal_objectness_assignment[0,:] == i)[0]
                            if ids.shape[0] == 0:
                                mean = torch.zeros(5).cuda()
                            else:
                                mean = torch.mean(proposal_aggregation_features[0,ids,:], dim=0).cuda()
                            objects_mean[0, i, 0:5] = mean

                        variance_loss = torch.tensor(0)
                        dist_loss = torch.tensor(0)
                        reg_loss = torch.tensor(0)

                        for i in range(0, number_of_gt_objects):
                            current_mean = objects_mean[:, i, :]

                            # Regularization-Loss
                            current_reg_loss = torch.abs(current_mean)
                            current_reg_loss = torch.mean(current_reg_loss)
                            reg_loss = reg_loss + current_reg_loss

                            # Variance-Loss
                            ids = torch.where(proposal_objectness_assignment[0,:] == i)[0]
                            number_of_assigned_proposals = ids.shape[0]
                            summed_variance_loss = torch.tensor(0)
                            for j in range(0, number_of_assigned_proposals):
                                current_emb_feature = proposal_aggregation_features[:,ids[j],:]
                                current_variance_loss = l1_loss(current_mean, current_emb_feature)
                                current_variance_loss = current_variance_loss - sigma_var
                                current_variance_loss = max(torch.tensor(0), current_variance_loss)
                                current_variance_loss = current_variance_loss ** 2
                                summed_variance_loss = summed_variance_loss + current_variance_loss
                            if number_of_assigned_proposals != 0:
                                summed_variance_loss = summed_variance_loss / number_of_assigned_proposals
                            variance_loss = variance_loss + summed_variance_loss

                            # Distribution-Loss
                            for j in range(0, number_of_gt_objects):
                                if i != j:
                                    current_mean_1 = objects_mean[:, i, :]
                                    current_mean_2 = objects_mean[:, j, :]
                                    current_dist_loss = l1_loss(current_mean_1, current_mean_2)
                                    current_dist_loss = 2 * sigma_dist - current_dist_loss
                                    current_dist_loss = max(torch.tensor(0), current_dist_loss)
                                    current_dist_loss = current_dist_loss ** 2
                                    dist_loss = dist_loss + current_dist_loss

                        if number_of_gt_objects != 0:
                            variance_loss = variance_loss / number_of_gt_objects
                            reg_loss = reg_loss / number_of_gt_objects

                        dist_help = number_of_gt_objects * (number_of_gt_objects - 1)
                        if dist_help != 0:
                            dist_loss = dist_loss / dist_help
                        
                        print(variance_loss)
                        print(dist_loss)
                        print(reg_loss)
                        #self.proposal_agg_feature_loss = variance_loss + dist_loss + gamma * reg_loss
                        self.proposal_agg_feature_loss = (variance_loss + dist_loss + gamma * reg_loss) * 100
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
    

    def ins_pred_in_time_mpa(self, labels, proposals_binary_mask, fp_points_sem_classes, proposals_aggregation_features, proposals_semantic_classes, proposals_objectness_scores, proposals_positions, proposals_ids, point_semantic_classes, point_positions, next_ins_id):
        """
        Calculate instance probabilities for each point with considering old predictions and using MPA
        Points can be assigned to multiple proposals
        :return: instance ids for all points, and new instances and next available ins_id
        """

        proposals_positions = proposals_positions.squeeze()
        proposals_ids = proposals_ids.squeeze()
        proposals_aggregation_features = proposals_aggregation_features.squeeze()
        proposals_aggregation_features = torch.transpose(proposals_aggregation_features, 0, 1)
        proposals_objectness_scores = proposals_objectness_scores.squeeze()
        proposals_objectness_scores = torch.transpose(proposals_objectness_scores, 0, 1)
        proposals_binary_mask = proposals_binary_mask.squeeze()
        proposals_binary_mask = torch.transpose(proposals_binary_mask, 0, 1)
        proposals_binary_mask = torch.transpose(proposals_binary_mask, 1, 2)

        point_semantic_classes = point_semantic_classes.cpu().detach().numpy()
        proposals_positions = proposals_positions.cpu().detach().numpy()
        proposals_ids = proposals_ids.cpu().detach().numpy()
        proposals_aggregation_features = proposals_aggregation_features.cpu().detach().numpy()
        proposals_objectness_scores = proposals_objectness_scores.cpu().detach().numpy()
        proposals_binary_mask = proposals_binary_mask.cpu().detach().numpy()

        ins_prediction = np.zeros(point_semantic_classes.shape, dtype=int)

        epsilon = 0.8
        print(epsilon)
        minpts = 1
        use_emb = False
        use_rp_r_bb = True
        use_rp_bb = False
        use_rp_r = False
        use_rp = False                    
        use_p = False            
        use_prop_sem_class = False
        use_objectness_scores = False
        use_binary_mask = False
        use_majority_voting = True

        if use_objectness_scores:
            print('use_objectness_scores')
            negative_proposals_objectness_scores = proposals_objectness_scores[:, 0]
            positive_proposals_objectness_scores = proposals_objectness_scores[:, 1]

            negative_proposals_indexes = np.where(negative_proposals_objectness_scores > positive_proposals_objectness_scores)[0]
            positive_proposals_indexes = np.where(negative_proposals_objectness_scores <= positive_proposals_objectness_scores)[0]

            negative_proposals_ids = proposals_ids[negative_proposals_indexes]
            positive_proposals_ids = proposals_ids[positive_proposals_indexes]

            all_negative_proposal_ids = negative_proposals_ids.flatten()
            all_negative_proposal_ids = np.unique(all_negative_proposal_ids)
            all_positive_proposal_ids = positive_proposals_ids.flatten()
            all_positive_proposal_ids = np.unique(all_positive_proposal_ids)
            only_negative_proposal_ids = np.setdiff1d(all_negative_proposal_ids, all_positive_proposal_ids)
            proposals_ids = positive_proposals_ids
            proposals_positions = proposals_positions[positive_proposals_indexes]
            proposals_aggregation_features = proposals_aggregation_features[positive_proposals_indexes]
            proposals_semantic_classes = proposals_semantic_classes[positive_proposals_indexes]
            proposals_binary_mask = proposals_binary_mask[positive_proposals_indexes]

        # Proposal Clustering with DBScan
        if use_emb:
            print('use_emb')
            refined_proposals_positions = np.zeros((proposals_positions.shape[0], 3), dtype=int)
            clustering_inputs = proposals_aggregation_features
        elif use_rp_r_bb:
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

        all_negative_mask_points_ids = []
        all_positive_mask_points_ids = []

        prev_ins_id = next_ins_id
        for index, label in np.ndenumerate(clustering_outputs):
            if label >= 0: # otherwise outlier
                proposal_ids = proposals_ids[index]
                if use_binary_mask:
                    #print('use_binary_mask')
                    proposal_binary_mask = proposals_binary_mask[index]
                    negative_binary_mask = proposal_binary_mask[:, 0]
                    positive_binary_mask = proposal_binary_mask[:, 1]
                    negative_binary_mask_indexes = np.where(negative_binary_mask > positive_binary_mask)[0]
                    positive_binary_mask_indexes = np.where(negative_binary_mask <= positive_binary_mask)[0]
                    if negative_binary_mask_indexes.shape[0] > 0:
                        all_negative_mask_points_ids.append(np.unique(proposal_ids[negative_binary_mask_indexes]))
                    if positive_binary_mask_indexes.shape[0] > 0:
                        all_positive_mask_points_ids.append(np.unique(proposal_ids[positive_binary_mask_indexes]))
                    proposal_ids = proposal_ids[positive_binary_mask_indexes]
                    #print(proposal_ids.shape[0])
                ins_prediction[proposal_ids] = prev_ins_id + label
                if use_prop_sem_class:
                    #print('use_prop_sem_class')
                    point_semantic_classes[proposal_ids] = proposals_semantic_classes[index]           
                if prev_ins_id + label > next_ins_id:
                    next_ins_id = prev_ins_id + label   

        if use_binary_mask:
            print('use_binary_mask')
            help_len_neg = len(all_negative_mask_points_ids)
            help_len_pos = len(all_positive_mask_points_ids)
            if help_len_neg != 0:
                all_negative_mask_points_ids = np.concatenate(all_negative_mask_points_ids, axis=0)
                if help_len_pos != 0:
                    all_positive_mask_points_ids = np.concatenate(all_positive_mask_points_ids, axis=0)
                    only_negative_mask_points_ids = np.setdiff1d(all_negative_mask_points_ids, all_positive_mask_points_ids)
                else: 
                    only_negative_mask_points_ids = all_negative_mask_points_ids
                point_semantic_classes[only_negative_mask_points_ids] = fp_points_sem_classes[only_negative_mask_points_ids]


        # majority voting
        if use_majority_voting:
            print('use_majority_voting')
            for i in range(prev_ins_id, next_ins_id + 1):
                instance_ids = np.where(ins_prediction == i)
                if instance_ids[0].size == 0:
                    continue
                point_semantic_classes_current_instance = point_semantic_classes[instance_ids]
                bincount = np.bincount(point_semantic_classes_current_instance)
                most_frequent = bincount.argmax()
                point_semantic_classes[instance_ids] = most_frequent
        
        next_ins_id = next_ins_id + 1 #!!!!!!

        return ins_prediction, next_ins_id, proposals_ids, proposals_positions, point_semantic_classes, refined_proposals_positions   


    def ins_pred_in_time_nms(self, proposals_objectness_scores, proposals_confidence_score, proposals_positions, proposals_ids, point_semantic_classes, point_positions, next_ins_id):
        """
        Calculate instance probabilities for each point with considering old predictions and using NMS
        Points can be assigned to multiple proposals
        :return: instance ids for all points, and new instances and next available ins_id
        """

        proposals_positions = proposals_positions.squeeze()
        proposals_ids = proposals_ids.squeeze()

        #point_positions = point_positions.cpu().detach().numpy()
        point_positions = point_positions.cpu()
        point_semantic_classes = point_semantic_classes.cpu().detach().numpy()
        proposals_positions = proposals_positions.cpu().detach().numpy()
        proposals_ids = proposals_ids.cpu().detach().numpy()

        proposal_points = point_positions[proposals_ids]

        refined_proposals_positions = proposals_positions

        ins_prediction = np.zeros(point_semantic_classes.shape, dtype=int)

        proposals_objectness_scores = proposals_objectness_scores.squeeze()
        proposals_objectness_scores = torch.transpose(proposals_objectness_scores, 0, 1)
        softmax = torch.nn.Softmax(1)
        proposals_objectness_scores = softmax(proposals_objectness_scores)
        proposals_objectness_scores = proposals_objectness_scores[:, 1]        


        use_majority_voting = False

        thresh_iou = 0.25


        '''
        x1 = np.min(proposal_points[:,:,0], axis=1)
        y1 = np.min(proposal_points[:,:,1], axis=1)
        z1 = np.min(proposal_points[:,:,2], axis=1)
        x2 = np.max(proposal_points[:,:,0], axis=1)
        y2 = np.max(proposal_points[:,:,1], axis=1)
        z2 = np.max(proposal_points[:,:,2], axis=1)
        '''
        x1 = torch.min(proposal_points[:,:,0], dim=1)[0]
        y1 = torch.min(proposal_points[:,:,1], dim=1)[0]
        z1 = torch.min(proposal_points[:,:,2], dim=1)[0]
        x2 = torch.max(proposal_points[:,:,0], dim=1)[0]
        y2 = torch.max(proposal_points[:,:,1], dim=1)[0]
        z2 = torch.max(proposal_points[:,:,2], dim=1)[0]

        #scores = torch.from_numpy(proposals_confidence_score)
        scores = proposals_objectness_scores.cpu()

        areas = (x2 - x1) * (y2 - y1) * (z2 - z1)

        order = scores.argsort()

        keep = []

        while len(order) > 0:

            idx = order[-1]

            keep.append(idx.item())

            order = order[:-1]

            if len(order) == 0:
                break

            xx1 = torch.index_select(x1,dim = 0, index = order)
            xx2 = torch.index_select(x2,dim = 0, index = order)
            yy1 = torch.index_select(y1,dim = 0, index = order)
            yy2 = torch.index_select(y2,dim = 0, index = order)
            zz1 = torch.index_select(z1,dim = 0, index = order)
            zz2 = torch.index_select(z2,dim = 0, index = order)
            
            xx1 = torch.max(xx1, x1[idx])
            yy1 = torch.max(yy1, y1[idx])
            zz1 = torch.max(zz1, z1[idx])
            xx2 = torch.min(xx2, x2[idx])
            yy2 = torch.min(yy2, y2[idx])
            zz2 = torch.min(zz2, z2[idx])

            w = xx2 - xx1
            h = yy2 - yy1
            l = zz2 - zz1

            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            l = torch.clamp(l, min=0.0)

            inter = w*h*l

            rem_areas = torch.index_select(areas, dim = 0, index = order)

            union = (rem_areas - inter) + areas[idx]

            IoU = inter / union

            mask = IoU < thresh_iou
            order = order[mask]


        prev_ins_id = next_ins_id

        for i in keep:
            proposal_ids = proposals_ids[i]
            ins_prediction[proposal_ids] = next_ins_id  
            next_ins_id = next_ins_id + 1

        print(next_ins_id-1)

        # majority voting
        if use_majority_voting:
            print('use_majority_voting')
            for i in range(prev_ins_id, next_ins_id):
                instance_ids = np.where(ins_prediction == i)
                if instance_ids[0].size == 0:
                    continue
                point_semantic_classes_current_instance = point_semantic_classes[instance_ids]
                bincount = np.bincount(point_semantic_classes_current_instance)
                most_frequent = bincount.argmax()
                point_semantic_classes[instance_ids] = most_frequent
        
        return ins_prediction, next_ins_id, proposals_ids, proposals_positions, point_semantic_classes, refined_proposals_positions 


    def ins_pred_in_time_mpa_bb(self, labels, fp_points_sem_classes, proposal_semantic_classes, proposals_objectness_scores, proposals_positions, proposals_ids, point_semantic_classes, point_positions, next_ins_id):
        """
        Calculate instance probabilities for each point with considering old predictions and use the bounding boxes to aggregate proposals
        Points can be assigned to multiple proposals
        :return: instance ids for all points, and new instances and next available ins_id
        """

        #proposals_positions_torch = proposals_positions
        
        proposals_objectness_scores = proposals_objectness_scores.squeeze()
        proposals_objectness_scores = torch.transpose(proposals_objectness_scores, 0, 1)
        proposals_positions = proposals_positions.squeeze()
        proposals_ids = proposals_ids.squeeze()

        point_semantic_classes = point_semantic_classes.cpu().detach().numpy()
        point_positions = point_positions.cpu().detach().numpy()
        proposals_objectness_scores = proposals_objectness_scores.cpu().detach().numpy()
        proposals_positions = proposals_positions.cpu().detach().numpy()
        proposals_ids = proposals_ids.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        ins_prediction = np.zeros(point_semantic_classes.shape)

        iou_treshold = 0.1
        use_all_proposals = True
        use_majority_voting = True
        use_prop_sem_classes = False
        use_fp_sem_classes = False
        store_number_proposals = False
        file_folder = '/globalwork/kreuzberg/4D-PLS/test/Log_2022-05-12_16-20-31_importance_None_str1_bigpug_2_chkp_1400/'
        
        final_objects_ids = []

        proposals_positions_help = []

        positive_proposals_counter = 0
        
        proposal_background_counter = 0
        wrong_proposal_sem_counter = 0
        correct_proposal_sem_counter = 0
        wrong_point_sem_counter = 0
        correct_point_sem_counter = 0

        wrong_positive_proposals_counter = 0
        correct_positive_proposals_counter = 0
        wrong_negative_proposals_counter = 0
        correct_negative_proposals_counter = 0

        positive_proposals_points_ids = np.zeros(1, int)
        negative_proposals_points_ids = np.zeros(1, int)

        '''
        object_points_ids = torch.where(ins_labels != 0)[0]
        object_centers_gt = object_centers_gt[object_points_ids, 4:7]
        object_centers_gt = object_centers_gt.unsqueeze(0)
        object_centers_gt = torch.unique(object_centers_gt, dim=1)
        dist1, ind1, _, _ = nn_distance(proposals_positions_torch, object_centers_gt)
        '''

        i1 = 0
        for associated_proposal_ids in proposals_ids:
            associated_proposal_ids = np.unique(associated_proposal_ids)
            help1 = point_semantic_classes[associated_proposal_ids]
            help2 = np.bincount(help1).argmax()
            help3 = proposal_semantic_classes[i1]
            help4 = labels[associated_proposal_ids]
            help5 = np.bincount(help4).argmax()
            help6 = np.in1d([1,2,3,4,5,6,7,8], help4)
            #'''
            if help5 == 0 or help5 > 8:
                # Proposal in Background
                proposal_background_counter += 1
            else:
                # Proposal in Foreground
                if help3 != help5:
                    wrong_proposal_sem_counter += 1
                if help3 == help5:
                    correct_proposal_sem_counter += 1
                if help2 != help5:
                    wrong_point_sem_counter += 1
                if help2 == help5: 
                    correct_point_sem_counter += 1
            #'''
            if proposals_objectness_scores[i1][1] >= proposals_objectness_scores[i1][0] or use_all_proposals:
            #if help5 > 0 and help5 < 9:
            #if help6.any():
                positive_proposals_counter += 1
                positive_proposals_points_ids = np.append(positive_proposals_points_ids, associated_proposal_ids)
                #'''
                if help5 == 0 or help5 > 8:
                    wrong_positive_proposals_counter += 1
                else:
                    correct_positive_proposals_counter += 1
                #'''
                proposals_positions_help.append(proposals_positions[i1])
                if use_prop_sem_classes:
                    point_semantic_classes[associated_proposal_ids] = proposal_semantic_classes[i1]
                #else: 
                #    if help5 > 0 and help5 < 9:
                #        point_semantic_classes[associated_proposal_ids] = help5
                if i1 == 0:
                    final_objects_ids.append(associated_proposal_ids)
                    i1 += 1
                    continue
                bb1_points = point_positions[associated_proposal_ids]
                merged = False
                for i2, final_object_ids in enumerate(final_objects_ids):
                    bb2_points = point_positions[final_object_ids]
                    iou = get_iou(bb1_points, bb2_points)
                    if iou >= iou_treshold:
                        final_objects_ids[i2] = np.concatenate((final_object_ids, associated_proposal_ids))
                        merged = True
                        break
                if not merged:
                    final_objects_ids.append(associated_proposal_ids)
            else:
                negative_proposals_points_ids = np.append(negative_proposals_points_ids, associated_proposal_ids)
                #'''
                if help5 == 0 or help5 > 8:
                    correct_negative_proposals_counter += 1
                else:
                    wrong_negative_proposals_counter += 1
                #'''
                #if use_fp_sem_classes:
                #    point_semantic_classes[associated_proposal_ids] = fp_points_sem_classes[associated_proposal_ids]
                #else:
                    # Set all class labels of the points belonging to a negative proposal to 0 (ignored label)
                    #point_semantic_classes[associated_proposal_ids] = 0 #ATTENTION: EVAL METRIC does not allow this
                    # Set all class labels of the points belonging to a negative proposal to 9 (road)
                    #point_semantic_classes[associated_proposal_ids] = 9
                    # Set all class labels of the points belonging to a negative proposal to GT label
                    #point_semantic_classes[associated_proposal_ids] = labels[associated_proposal_ids] #Problem: Some points would get label 0
                    #point_semantic_classes = np.where(point_semantic_classes == 0, 9, point_semantic_classes)
            i1 += 1

        print(positive_proposals_counter)

        negative_proposals_points_ids = np.setdiff1d(negative_proposals_points_ids, positive_proposals_points_ids)
        if use_fp_sem_classes:
            help7 = point_semantic_classes[negative_proposals_points_ids]
            point_semantic_classes[negative_proposals_points_ids] = 9
            help10 = point_semantic_classes[negative_proposals_points_ids]
            point_semantic_classes[negative_proposals_points_ids] = fp_points_sem_classes[negative_proposals_points_ids]
            help8 = point_semantic_classes[negative_proposals_points_ids]
            help9 = labels[negative_proposals_points_ids]
            correct_before = (help7 == help9).sum().item()
            correct_after = (help8 == help9).sum().item()
            correct_9 = (help10 == help9).sum().item()
            print(correct_before)
            print(correct_after)
            print(correct_9)
        #else:
        #    point_semantic_classes[negative_proposals_points_ids] = 9
        
        prev_ins_id = next_ins_id
        for i, instance_ids in enumerate(final_objects_ids):
            ins_prediction[instance_ids] = next_ins_id
            next_ins_id += 1


        # majority voting
        if use_majority_voting:
            for i in range(prev_ins_id, next_ins_id + 1):
                instance_ids = np.where(ins_prediction == i)
                if instance_ids[0].size == 0:
                    continue
                point_semantic_classes_current_instance = point_semantic_classes[instance_ids]
                bincount = np.bincount(point_semantic_classes_current_instance)
                most_frequent = bincount.argmax()
                point_semantic_classes[instance_ids] = most_frequent

        return ins_prediction, next_ins_id, proposals_ids, proposals_positions, point_semantic_classes, np.asarray(proposals_positions_help)


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


