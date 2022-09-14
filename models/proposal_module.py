import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2.pointnet2_modules import PointnetSAModuleVotes

class ProposalModule(nn.Module):
    def __init__(self, num_class, num_proposal, seed_feat_dim, radius, nsample, nsample_sub, use_fps, use_geo_features, use_binary_mask):
        super().__init__()

        self.num_class = num_class
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim # F (dimension of the per-point features)
        self.use_fps = use_fps
        self.use_geo_features = use_geo_features

        # Proposal Generation

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposal,
            radius=radius,
            nsample=nsample,
            nsample_sub=nsample_sub,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            mlp_binary_mask=[self.seed_feat_dim + 128, 256, 128, 64, 32, 2],#
            use_binary_mask = use_binary_mask,#
            use_xyz=True,
            normalize_xyz=True
        )

        # Object proposal/detection (MLP2)
        # Objectness scores (2), aggregation features (4/5/7), semantic classes
        self.conv1 = torch.nn.Conv1d(128, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        if use_geo_features:
            # geometric features
            #self.conv3 = torch.nn.Conv1d(128, 2 + 4 + self.num_class, 1)
            # geometric features + bb-size
            self.conv3 = torch.nn.Conv1d(128, 2 + 7 + self.num_class, 1)
        else:
            # embedding features
            self.conv3 = torch.nn.Conv1d(128, 2 + 5 + self.num_class, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, point_xyz, point_features):

        # Random sampling from the votes
        num_seed = point_xyz.shape[1]
        batch_size = point_xyz.shape[0] 
        sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()

        # VOTE AGGREGATION / PROPOSAL GENERATION
        if self.use_fps:
            # FPS
            #proposal_xyz, proposal_features, _, proposal_idx = self.vote_aggregation(point_xyz, point_features)
            proposal_xyz, proposal_features, _, proposal_idx, proposal_binary_mask = self.vote_aggregation(point_xyz, point_features)
        else:
            # Random sampling
            #proposal_xyz, proposal_features, _, proposal_idx = self.vote_aggregation(point_xyz, point_features, sample_inds)
            proposal_xyz, proposal_features, _, proposal_idx, proposal_binary_mask = self.vote_aggregation(point_xyz, point_features, sample_inds)        


        # OBJECT GENERATION
        net = F.relu(self.bn1(self.conv1(proposal_features)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)  # (B, 2 + 4 + num_class, num_proposal)

        if self.use_geo_features:
            proposal_objectness_scores = net [:, 0:2, :]
            #proposal_aggregation_features = net [:, 2:6, :]
            proposal_aggregation_features = net [:, 2:9, :]
            #proposal_sematic_class = net [:, 6:6+self.num_class, :]
            proposal_sematic_class = net [:, 9:9+self.num_class, :]
        else:
            proposal_objectness_scores = net [:, 0:2, :]
            proposal_aggregation_features = net [:, 2:7, :]
            proposal_sematic_class = net [:, 7:7+self.num_class, :]
        

        #return proposal_sematic_class, proposal_aggregation_features, proposal_objectness_scores, proposal_xyz, proposal_idx, proposal_features
        return proposal_binary_mask, proposal_sematic_class, proposal_aggregation_features, proposal_objectness_scores, proposal_xyz, proposal_idx, proposal_features
