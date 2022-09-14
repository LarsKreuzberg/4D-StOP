#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import sys

from torch.utils.tensorboard import SummaryWriter

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from utils.config import Config
from sklearn.neighbors import KDTree

from models.blocks import BatchNormBlock, KPConv


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#


class ModelTrainer:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, chkp_path=None, finetune=False, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """

        ############
        # Parameters
        ############

        # Epoch index
        self.epoch = 0
        self.step = 0

        # During Pre-Training
        var_params = [v for k, v in net.named_parameters() if 'head_var' in k]
        # Optimizer with specific learning rate for deformable KPConv
        deform_params = [v for k, v in net.named_parameters() if 'offset' in k and not 'head_var' in k]
        other_params = [v for k, v in net.named_parameters() if 'offset' not in k and not 'head_var' in k]
        deform_lr = config.learning_rate * config.deform_lr_factor
        var_lr =  1e-3
        self.optimizer = torch.optim.SGD([{'params': other_params},
                                          {'params': var_params, 'lr': var_lr},
                                          {'params': deform_params, 'lr': deform_lr}],
                                         lr=config.learning_rate,
                                         momentum=config.momentum,
                                         weight_decay=config.weight_decay)


        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        if (chkp_path is not None):
            if finetune:
                checkpoint = torch.load(chkp_path)
                if checkpoint['model_state_dict']['head_var.mlp.weight'].shape[0] == \
                        checkpoint['model_state_dict']['head_var.mlp.weight'].shape[1] and config.free_dim != 0:
                    checkpoint['model_state_dict']['head_var.mlp.weight'] = net.head_var.mlp.weight
                    checkpoint['model_state_dict']['head_var.batch_norm.bias'] = net.head_var.batch_norm.bias

                if checkpoint['model_state_dict']['head_var.mlp.weight'].shape[0] -  \
                    checkpoint['model_state_dict']['head_var.mlp.weight'].shape[1] != config.free_dim:
                    checkpoint['model_state_dict']['head_var.mlp.weight'] = net.head_var.mlp.weight
                    checkpoint['model_state_dict']['head_var.batch_norm.bias'] = net.head_var.batch_norm.bias

                if checkpoint['model_state_dict']['head_var.mlp.weight'].shape[0] != net.head_var.mlp.weight.shape[0] \
                        or checkpoint['model_state_dict']['head_var.mlp.weight'].shape[1] !=net.head_var.mlp.weight.shape[1]:
                    checkpoint['model_state_dict']['head_var.mlp.weight'] = net.head_var.mlp.weight
                    checkpoint['model_state_dict']['head_var.batch_norm.bias'] = net.head_var.batch_norm.bias

                if config.reinit_var:
                    checkpoint['model_state_dict']['head_var.mlp.weight'] = net.head_var.mlp.weight
                    checkpoint['model_state_dict']['head_var.batch_norm.bias'] = net.head_var.batch_norm.bias

                net.load_state_dict(checkpoint['model_state_dict'])
                net.train()
                print("Model restored and ready for finetuning.")
            elif config.freeze:
                checkpoint = torch.load(chkp_path)
                pretrained_dict = checkpoint['model_state_dict']
                net_dict = net.state_dict() 
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if "pnet" not in k}
                net_dict.update(pretrained_dict)
                net.load_state_dict(pretrained_dict, strict=False)
                #net.load_state_dict(checkpoint['model_state_dict'])
                #net.load_state_dict(checkpoint['model_state_dict'], strict=False)
                net.train()
                child_counter = 0
                for child in net.children():
                    if child_counter < 2:
                        for param in child.parameters():
                            param.requires_grad = False
                        for module in child.modules():
                            if isinstance(module, nn.BatchNorm1d) or isinstance(module, BatchNormBlock):
                                module.eval()
                    child_counter += 1  
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=config.learning_rate)
                #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                child_counter = 0
                for child in net.children():
                    print(" child", child_counter, "is -")
                    print(child)
                    for param in child.parameters():
                        print(param.requires_grad)
                    child_counter += 1  
                self.epoch = checkpoint['epoch']
                #net.train()
                print("Model and training state restored.")
            else:
                checkpoint = torch.load(chkp_path)
                if config.reinit_var:
                    checkpoint['model_state_dict']['head_var.mlp.weight'] = net.head_var.mlp.weight
                    checkpoint['model_state_dict']['head_var.batch_norm.bias'] = net.head_var.batch_norm.bias
                if not config.reinit_var:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                net.load_state_dict(checkpoint['model_state_dict'])
                self.epoch = checkpoint['epoch']
                net.train()
                print("Model and training state restored (4D-PLS).")


        # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                current_log = time.strftime('Log_%Y-%m-%d_%H-%M-%S', time.localtime())
                config.saving_path = os.path.join(config.train_path, current_log)
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, net, training_loader, val_loader, config):
        """
        Train the model on a particular dataset.
        """
        ################
        # Initialization
        ################

        writer = SummaryWriter()

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                file.write('epochs steps loss out_loss center_loss instance_loss variance_loss train_accuracy time\n')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)


        # Start training loop
        for epoch in range(config.max_epoch):

            self.step = 0
            acc_summed = 0
            loss_summed = 0
            output_loss_summed = 0
            center_loss_summed = 0
            instance_loss_summed = 0
            variance_loss_summed = 0

            for batch in training_loader:

                ##################
                # Processing batch
                ##################

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs, centers_output, var_output, embedding = net(batch, config)

                loss = net.loss(outputs, centers_output, var_output, embedding, batch.labels, batch.ins_labels, batch.centers, batch.points, batch.times.unsqueeze(1))

                acc = net.accuracy(outputs, batch.labels)

                acc_summed = acc_summed + acc
                loss_summed = loss_summed + loss
                output_loss_summed = output_loss_summed + net.output_loss
                center_loss_summed = center_loss_summed + net.center_loss
                instance_loss_summed = instance_loss_summed + net.instance_loss
                variance_loss_summed = variance_loss_summed + net.variance_loss

                t += [time.time()]

                # Backward + optimize
                loss.backward()

                if config.grad_clip_norm > 0:
                    # torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Average timing
                if self.step < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => L={:.3f} L_O={:.3f} L_C={:.3f} L_I={:.3f} L_V={:.3f} acc={:3.0f}%'
                    print(message.format(self.epoch+1, self.step+1,
                                         loss.item(),
                                         net.output_loss.item(),
                                         net.center_loss.item(),
                                         net.instance_loss.item(),
                                         net.variance_loss.item(),
                                         100 * acc))

                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'
                        file.write(message.format(self.epoch+1,
                                                  self.step+1,
                                                  loss,
                                                  net.output_loss,
                                                  net.center_loss,
                                                  net.instance_loss,
                                                  net.variance_loss,
                                                  acc,
                                                  t[-1] - t0))

                self.step += 1

            ##############
            # End of epoch
            ##############

            writer.add_scalar("Loss", loss_summed/config.epoch_steps, self.epoch+1)
            writer.add_scalar("Output_Loss", output_loss_summed/config.epoch_steps, self.epoch+1)
            writer.add_scalar("Center_Loss", center_loss_summed/config.epoch_steps, self.epoch+1)
            writer.add_scalar("Instance_Loss", instance_loss_summed/config.epoch_steps, self.epoch+1)
            writer.add_scalar("Variance_Loss", variance_loss_summed/config.epoch_steps, self.epoch+1)
            writer.add_scalar("Accuracy", acc_summed/config.epoch_steps, self.epoch+1)
            i = 0
            for param_group in self.optimizer.param_groups:
                writer.add_scalar("Learning_Rate_" + str(i), param_group['lr'], self.epoch+1)
                i += 1

            # Update learning rate
            if self.epoch in config.lr_decays:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= config.lr_decays[self.epoch]

            # Saving
            if config.saving:
                # Get current state dict
                save_dict = {'epoch': self.epoch+1,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}

                # Save current state of the network (for restoring purposes)
                checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path)

                # Save checkpoints occasionally
                if (self.epoch + 1) % config.checkpoint_gap == 0:
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
                    torch.save(save_dict, checkpoint_path)


            # Validation
            #if self.epoch+1 % 40 == 0:
            #    net.eval()
            #    self.optimizer.zero_grad()
            #    self.validation(net, val_loader, config)
            #    net.train()
            
            # Update epoch
            self.epoch += 1

        
        writer.flush()
        writer.close()

        print('Finished Training')
        return

    def train_mpa(self, net, training_loader, val_loader, config):
        for param_group in self.optimizer.param_groups:
            print(param_group['lr'])
        """
        Train the model on a particular dataset.
        """
        ################
        # Initialization
        ################

        writer = SummaryWriter()

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                file.write('epochs steps loss out_loss center_loss instance_loss variance_loss train_accuracy time\n')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start training loop
        for epoch in range(config.max_epoch):

            self.step = 0
            acc_summed = 0
            loss_summed = 0
            point_semantic_classification_loss_summed = 0
            point_objectness_loss_summed = 0
            point_center_regression_loss_summed = 0
            proposal_objectness_loss_summed = 0
            proposal_semantic_loss_summed = 0
            fp_points_sem_loss_summed = 0
            proposal_agg_feature_loss_summed = 0
            proposal_mask_loss_summed = 0
            instance_loss_summed = 0
            variance_loss_summed = 0

            for batch in training_loader:

                ##################
                # Processing batch
                ##################

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                proposal_binary_mask, fp_points_sem_classes, proposal_semantic_classes, proposal_aggregation_features, proposal_objectness_scores, proposal_positions, proposal_ids, outputs, point_votes, point_objectness_scores = net(batch)
                loss = net.loss(proposal_binary_mask, outputs, point_votes, point_objectness_scores, proposal_ids, proposal_objectness_scores, proposal_positions, proposal_semantic_classes, proposal_aggregation_features, fp_points_sem_classes, batch.labels, batch.ins_labels, batch.centers, batch.points[0], batch.times.unsqueeze(1))
                
                acc = net.accuracy(outputs, batch.labels)

                acc_summed = acc_summed + acc
                loss_summed = loss_summed + loss.item()
                point_semantic_classification_loss_summed = point_semantic_classification_loss_summed + net.point_semantic_classification_loss.item()
                point_objectness_loss_summed = point_objectness_loss_summed + net.point_objectness_loss.item()
                point_center_regression_loss_summed = point_center_regression_loss_summed + net.point_center_regression_loss.item()
                proposal_objectness_loss_summed = proposal_objectness_loss_summed + net.proposal_objectness_loss.item()
                proposal_semantic_loss_summed = proposal_semantic_loss_summed + net.proposal_semantic_loss.item()
                fp_points_sem_loss_summed = fp_points_sem_loss_summed + net.fp_points_sem_loss.item()
                proposal_agg_feature_loss_summed = proposal_agg_feature_loss_summed + net.proposal_agg_feature_loss.item()
                proposal_mask_loss_summed = proposal_mask_loss_summed + net.proposal_mask_loss.item()
                instance_loss_summed = instance_loss_summed + net.instance_loss.item()
                variance_loss_summed = variance_loss_summed + net.variance_loss.item()


                t += [time.time()]

                # Backward + optimize
                if loss != torch.tensor(0):
                    print("Backward + optimize")
                    loss.backward()

                    if config.grad_clip_norm > 0:
                        # torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                        torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                    self.optimizer.step()
                else:
                    print("Don't backward + optimize")
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Average timing
                if self.step < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => L={:.3f} L_O={:.3f} L_C={:.3f} L_I={:.3f} L_V={:.3f} acc={:3.0f}%'
                    print(message.format(self.epoch + 1, self.step + 1,
                                         loss.item(),
                                         net.point_semantic_classification_loss.item(),
                                         net.point_objectness_loss.item(),
                                         net.instance_loss.item(),
                                         net.variance_loss.item(),
                                         100 * acc))

                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'
                        file.write(message.format(self.epoch + 1,
                                                  self.step + 1,
                                                  loss,
                                                  net.point_semantic_classification_loss,
                                                  net.point_objectness_loss,
                                                  net.instance_loss,
                                                  net.variance_loss,
                                                  acc,
                                                  t[-1] - t0))

                self.step += 1

            ##############
            # End of epoch
            ##############
            
            writer.add_scalar("Loss", loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Point_Semantic_Classification_Loss", point_semantic_classification_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Point_Objectness_Loss", point_objectness_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Point_Center_Regression_Loss", point_center_regression_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Proposal_Objectness_Loss", proposal_objectness_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Proposal_Semantic_Loss", proposal_semantic_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("FP-Semantic-Loss", fp_points_sem_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Proposal_Aggregation_Feature_Loss", proposal_agg_feature_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Proposal_Mask_Loss", proposal_mask_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Instance_Loss", instance_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Variance_Loss", variance_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Accuracy", acc_summed / config.epoch_steps, self.epoch + 1)
            i = 0
            for param_group in self.optimizer.param_groups:
                writer.add_scalar("Learning_Rate_" + str(i), param_group['lr'], self.epoch + 1)
                i += 1

            # Update learning rate
            if config.lr_exponential_decrease:
                if self.epoch in config.lr_decays:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= config.lr_decays[self.epoch]
            else:
                if (self.epoch + 1) % 40 == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] / 2

            # Saving
            if config.saving:
                # Get current state dict
                save_dict = {'epoch': self.epoch + 1,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}

                # Save current state of the network (for restoring purposes)
                checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path)

                # Save checkpoints occasionally
                if (self.epoch + 1) % config.checkpoint_gap == 0:
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
                    torch.save(save_dict, checkpoint_path)

            # Validation
            # if self.epoch+1 % 40 == 0:
            #    net.eval()
            #    self.optimizer.zero_grad()
            #    self.validation(net, val_loader, config)
            #    net.train()

            # Update epoch
            self.epoch += 1

        writer.flush()
        writer.close()

        print('Finished Training')
        return
    
    def train_prototype(self, net, training_loader, val_loader, config):
        """
        Train the model on a particular dataset.
        """
        ################
        # Initialization
        ################

        writer = SummaryWriter()

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                file.write('epochs steps loss out_loss center_loss instance_loss variance_loss train_accuracy time\n')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start training loop
        for epoch in range(config.max_epoch):

            self.step = 0
            acc_summed = 0
            loss_summed = 0
            point_semantic_classification_loss_summed = 0
            point_objectness_loss_summed = 0
            point_center_regression_loss_summed = 0
            proposal_objectness_loss_summed = 0
            proposal_semantic_loss_summed = 0
            proposal_agg_feature_loss_summed = 0
            instance_loss_summed = 0
            variance_loss_summed = 0

            for batch in training_loader:

                ##################
                # Processing batch
                ##################

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs, point_votes, point_objectness_scores = net(batch)

                loss = net.loss(outputs, point_votes, point_objectness_scores, batch.labels, batch.ins_labels, batch.centers)

                acc = net.accuracy(outputs, batch.labels)

                acc_summed = acc_summed + acc
                loss_summed = loss_summed + loss.item()
                point_semantic_classification_loss_summed = point_semantic_classification_loss_summed + net.point_semantic_classification_loss.item()
                point_objectness_loss_summed = point_objectness_loss_summed + net.point_objectness_loss.item()
                point_center_regression_loss_summed = point_center_regression_loss_summed + net.point_center_regression_loss.item()
                proposal_objectness_loss_summed = proposal_objectness_loss_summed + net.proposal_objectness_loss.item()
                proposal_semantic_loss_summed = proposal_semantic_loss_summed + net.proposal_semantic_loss.item()
                proposal_agg_feature_loss_summed = proposal_agg_feature_loss_summed + net.proposal_agg_feature_loss.item()
                instance_loss_summed = instance_loss_summed + net.instance_loss.item()
                variance_loss_summed = variance_loss_summed + net.variance_loss.item()


                t += [time.time()]

                # Backward + optimize
                loss.backward()

                if config.grad_clip_norm > 0:
                    # torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Average timing
                if self.step < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => L={:.3f} L_O={:.3f} L_C={:.3f} L_I={:.3f} L_V={:.3f} acc={:3.0f}%'
                    print(message.format(self.epoch + 1, self.step + 1,
                                         loss.item(),
                                         net.point_semantic_classification_loss.item(),
                                         net.point_objectness_loss.item(),
                                         net.instance_loss.item(),
                                         net.variance_loss.item(),
                                         100 * acc))

                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'
                        file.write(message.format(self.epoch + 1,
                                                  self.step + 1,
                                                  loss,
                                                  net.point_semantic_classification_loss,
                                                  net.point_objectness_loss,
                                                  net.instance_loss,
                                                  net.variance_loss,
                                                  acc,
                                                  t[-1] - t0))

                self.step += 1

            ##############
            # End of epoch
            ##############
            
            writer.add_scalar("Loss", loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Point_Semantic_Classification_Loss", point_semantic_classification_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Point_Objectness_Loss", point_objectness_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Point_Center_Regression_Loss", point_center_regression_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Proposal_Objectness_Loss", proposal_objectness_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Proposal_Semantic_Loss", proposal_semantic_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Proposal_Aggregation_Feature_Loss", proposal_agg_feature_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Instance_Loss", instance_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Variance_Loss", variance_loss_summed / config.epoch_steps, self.epoch + 1)
            writer.add_scalar("Accuracy", acc_summed / config.epoch_steps, self.epoch + 1)
            i = 0
            for param_group in self.optimizer.param_groups:
                writer.add_scalar("Learning_Rate_" + str(i), param_group['lr'], self.epoch + 1)
                i += 1

            # Update learning rate
            if self.epoch in config.lr_decays:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= config.lr_decays[self.epoch]

            # Saving
            if config.saving:
                # Get current state dict
                save_dict = {'epoch': self.epoch + 1,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}

                # Save current state of the network (for restoring purposes)
                checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path)

                # Save checkpoints occasionally
                if (self.epoch + 1) % config.checkpoint_gap == 0:
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
                    torch.save(save_dict, checkpoint_path)

            # Validation
            # if self.epoch+1 % 40 == 0:
            #    net.eval()
            #    self.optimizer.zero_grad()
            #    self.validation(net, val_loader, config)
            #    net.train()

            # Update epoch
            self.epoch += 1

        writer.flush()
        writer.close()

        print('Finished Training')
        return
    
    
    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def validation(self, net, val_loader, config: Config):

        if config.dataset_task == 'classification':
            self.object_classification_validation(net, val_loader, config)
        elif config.dataset_task == 'segmentation':
            self.object_segmentation_validation(net, val_loader, config)
        elif config.dataset_task == 'cloud_segmentation':
            self.cloud_segmentation_validation(net, val_loader, config)
        elif config.dataset_task == 'slam_segmentation':
            #self.slam_segmentation_validation(net, val_loader, config)
            self.my_slam_segmentation_validation(net, val_loader, config)
        else:
            raise ValueError('No validation method implemented for this network type')

    def object_classification_validation(self, net, val_loader, config):
        """
        Perform a round of validation and show/save results
        :param net: network object
        :param val_loader: data loader for validation set
        :param config: configuration object
        """

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Number of classes predicted by the model
        nc_model = config.num_classes
        softmax = torch.nn.Softmax(1)

        # Initialize global prediction over all models
        if not hasattr(self, 'val_probs'):
            self.val_probs = np.zeros((val_loader.dataset.num_models, nc_model))

        #####################
        # Network predictions
        #####################

        probs = []
        targets = []
        obj_inds = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start validation loop
        for batch in val_loader:

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            # Get probs and labels
            probs += [softmax(outputs).cpu().detach().numpy()]
            targets += [batch.labels.cpu().numpy()]
            obj_inds += [batch.model_inds.cpu().numpy()]
            torch.cuda.synchronize(self.device)

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * len(obj_inds) / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        # Stack all validation predictions
        probs = np.vstack(probs)
        targets = np.hstack(targets)
        obj_inds = np.hstack(obj_inds)

        ###################
        # Voting validation
        ###################

        self.val_probs[obj_inds] = val_smooth * self.val_probs[obj_inds] + (1 - val_smooth) * probs

        ############
        # Confusions
        ############

        validation_labels = np.array(val_loader.dataset.label_values)

        # Compute classification results
        C1 = fast_confusion(targets,
                            np.argmax(probs, axis=1),
                            validation_labels)

        # Compute votes confusion
        C2 = fast_confusion(val_loader.dataset.input_labels,
                            np.argmax(self.val_probs, axis=1),
                            validation_labels)

        # Saving (optionnal)
        if config.saving:
            print("Save confusions")
            conf_list = [C1, C2]
            file_list = ['val_confs.txt', 'vote_confs.txt']
            for conf, conf_file in zip(conf_list, file_list):
                test_file = join(config.saving_path, conf_file)
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        for line in conf:
                            for value in line:
                                text_file.write('%d ' % value)
                        text_file.write('\n')
                else:
                    with open(test_file, "w") as text_file:
                        for line in conf:
                            for value in line:
                                text_file.write('%d ' % value)
                        text_file.write('\n')

        val_ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
        vote_ACC = 100 * np.sum(np.diag(C2)) / (np.sum(C2) + 1e-6)
        print('Accuracies : val = {:.1f}% / vote = {:.1f}%'.format(val_ACC, vote_ACC))

        return C1

    def cloud_segmentation_validation(self, net, val_loader, config, debug=False):
        """
        Validation method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Do not validate if dataset has no validation cloud
        if val_loader.dataset.validation_split not in val_loader.dataset.all_splits:
            return

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # print(nc_tot)
        # print(nc_model)

        # Initiate global prediction over validation clouds
        if not hasattr(self, 'validation_probs'):
            self.validation_probs = [np.zeros((l.shape[0], nc_model))
                                     for l in val_loader.dataset.input_labels]
            self.val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in val_loader.dataset.label_values:
                if label_value not in val_loader.dataset.ignored_labels:
                    self.val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                      for labels in val_loader.dataset.validation_labels])
                    i += 1

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        t1 = time.time()

        # Start validation loop
        for i, batch in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            # Get probs and labels
            stacked_probs = softmax(outputs).cpu().detach().numpy()
            labels = batch.labels.cpu().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            in_inds = batch.input_inds.cpu().numpy()
            cloud_inds = batch.cloud_inds.cpu().numpy()
            torch.cuda.synchronize(self.device)

            # Get predictions and labels per instance
            # ***************************************

            i0 = 0
            for b_i, length in enumerate(lengths):
                # Get prediction
                target = labels[i0:i0 + length]
                probs = stacked_probs[i0:i0 + length]
                inds = in_inds[i0:i0 + length]
                c_i = cloud_inds[b_i]

                # Update current probs in whole cloud
                self.validation_probs[c_i][inds] = val_smooth * self.validation_probs[c_i][inds] \
                                                   + (1 - val_smooth) * probs

                # Stack all prediction for this epoch
                predictions.append(probs)
                targets.append(target)
                i0 += length

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * i / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (probs, truth) in enumerate(zip(predictions, targets)):

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                if label_value in val_loader.dataset.ignored_labels:
                    probs = np.insert(probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)

        t3 = time.time()

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Balance with real validation proportions
        C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        t4 = time.time()

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        t5 = time.time()

        # Saving (optionnal)
        if config.saving:

            # Name of saving file
            test_file = join(config.saving_path, 'val_IoUs.txt')

            # Line to write:
            line = ''
            for IoU in IoUs:
                line += '{:.3f} '.format(IoU)
            line = line + '\n'

            # Write in file
            if exists(test_file):
                with open(test_file, "a") as text_file:
                    text_file.write(line)
            else:
                with open(test_file, "w") as text_file:
                    text_file.write(line)

            # Save potentials
            pot_path = join(config.saving_path, 'potentials')
            if not exists(pot_path):
                makedirs(pot_path)
            files = val_loader.dataset.files
            for i, file_path in enumerate(files):
                pot_points = np.array(val_loader.dataset.pot_trees[i].data, copy=False)
                cloud_name = file_path.split('/')[-1]
                pot_name = join(pot_path, cloud_name)
                pots = val_loader.dataset.potentials[i].numpy().astype(np.float32)
                write_ply(pot_name,
                          [pot_points.astype(np.float32), pots],
                          ['x', 'y', 'z', 'pots'])

        t6 = time.time()

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} mean IoU = {:.1f}%'.format(config.dataset, mIoU))

        # Save predicted cloud occasionally
        if config.saving and (self.epoch + 1) % config.checkpoint_gap == 0:
            val_path = join(config.saving_path, 'val_preds_{:d}'.format(self.epoch + 1))
            if not exists(val_path):
                makedirs(val_path)
            files = val_loader.dataset.files
            for i, file_path in enumerate(files):

                # Get points
                points = val_loader.dataset.load_evaluation_points(file_path)

                # Get probs on our own ply points
                sub_probs = self.validation_probs[i]

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)

                # Get the predicted labels
                sub_preds = val_loader.dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]

                # Reproject preds on the evaluations points
                preds = (sub_preds[val_loader.dataset.test_proj[i]]).astype(np.int32)

                # Path of saved validation file
                cloud_name = file_path.split('/')[-1]
                val_name = join(val_path, cloud_name)

                # Save file
                labels = val_loader.dataset.validation_labels[i].astype(np.int32)
                write_ply(val_name,
                          [points, preds, labels],
                          ['x', 'y', 'z', 'preds', 'class'])

        # Display timings
        t7 = time.time()
        if debug:
            print('\n************************\n')
            print('Validation timings:')
            print('Init ...... {:.1f}s'.format(t1 - t0))
            print('Loop ...... {:.1f}s'.format(t2 - t1))
            print('Confs ..... {:.1f}s'.format(t3 - t2))
            print('Confs bis . {:.1f}s'.format(t4 - t3))
            print('IoU ....... {:.1f}s'.format(t5 - t4))
            print('Save1 ..... {:.1f}s'.format(t6 - t5))
            print('Save2 ..... {:.1f}s'.format(t7 - t6))
            print('\n************************\n')

        return

    def slam_segmentation_validation(self, net, val_loader, config, debug=True):
        """
        Validation method for slam segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Do not validate if dataset has no validation cloud
        if val_loader is None:
            return

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Create folder for validation predictions
        if not exists(join(config.saving_path, 'val_preds')):
            makedirs(join(config.saving_path, 'val_preds'))

        # initiate the dataset validation containers
        val_loader.dataset.val_points = []
        val_loader.dataset.val_labels = []

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        inds = []
        val_i = 0
        c_ious = []
        s_ious = []
        o_scores = []
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)
        accuracies = []

        t1 = time.time()

        # Start validation loop
        for i, batch in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs, centers_output, var_output, embedding = net(batch, config)
                acc = net.accuracy(outputs, batch.labels)
                accuracies.append(acc)
                probs = softmax(outputs).cpu().detach().numpy()

                if not config.pre_train and self.epoch > 50:
                    for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                        if label_value in val_loader.dataset.ignored_labels:
                            probs = np.insert(probs, l_ind, 0, axis=1)
                    preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]
                    preds = torch.from_numpy(preds)
                    preds.to(outputs.device)

                    ins_preds = net.ins_pred(preds, centers_output, var_output, embedding, batch.points, batch.times.unsqueeze(1))
                else:
                    ins_preds = torch.zeros(outputs.shape[0])
            # Get probs and labels
            stk_probs = softmax(outputs).cpu().detach().numpy()
            centers_output = centers_output.cpu().detach().numpy()
            centers_output = centers_output
            ins_preds = ins_preds.cpu().detach().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            f_inds = batch.frame_inds.cpu().numpy()
            r_inds_list = batch.reproj_inds
            r_mask_list = batch.reproj_masks
            labels_list = batch.val_labels

            torch.cuda.synchronize(self.device)

            # Get predictions and labels per instance
            # ***************************************

            i0 = 0
            for b_i, length in enumerate(lengths):

                # Get prediction
                probs = stk_probs[i0:i0 + length]
                center_props = centers_output[i0:i0 + length]
                ins_probs = ins_preds[i0:i0 + length]
                proj_inds = r_inds_list[b_i]
                proj_mask = r_mask_list[b_i]
                frame_labels = labels_list[b_i]
                s_ind = f_inds[b_i, 0]
                f_ind = f_inds[b_i, 1]

                # Project predictions on the frame points
                proj_probs = probs[proj_inds]
                proj_center_probs = center_props[proj_inds]
                proj_ins_probs = ins_probs[proj_inds]
                #proj_offset_probs = offset_probs[proj_inds]

                # Safe check if only one point:
                if proj_probs.ndim < 2:
                    proj_probs = np.expand_dims(proj_probs, 0)
                    proj_center_probs = np.expand_dims(proj_center_probs, 0)
                    proj_ins_probs = np.expand_dims(proj_ins_probs, 0)

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        proj_probs = np.insert(proj_probs, l_ind, 0, axis=1)

                # Predicted labels
                preds = val_loader.dataset.label_values[np.argmax(proj_probs, axis=1)]

                # Save predictions in a binary file
                filename = '{:s}_{:07d}.npy'.format(val_loader.dataset.sequences[s_ind], f_ind)
                filename_c = '{:s}_{:07d}_c.npy'.format(val_loader.dataset.sequences[s_ind], f_ind)
                filename_i = '{:s}_{:07d}_i.npy'.format(val_loader.dataset.sequences[s_ind], f_ind)
                filepath = join(config.saving_path, 'val_preds', filename)
                filepath_c = join(config.saving_path, 'val_preds', filename_c)
                filepath_i = join(config.saving_path, 'val_preds', filename_i)

                if exists(filepath):
                    frame_preds = np.load(filepath)
                    center_preds = np.load(filepath_c)
                    ins_preds = np.load(filepath_i)

                else:
                    frame_preds = np.zeros(frame_labels.shape, dtype=np.uint8)
                    center_preds = np.zeros(frame_labels.shape, dtype=np.float32)
                    ins_preds = np.zeros(frame_labels.shape, dtype=np.uint8)

                center_preds[proj_mask] = proj_center_probs[:, 0]
                frame_preds[proj_mask] = preds.astype(np.uint8)
                ins_preds[proj_mask] = proj_ins_probs
                np.save(filepath, frame_preds)
                np.save(filepath_c, center_preds)
                np.save(filepath_i, ins_preds)


                centers_gt = batch.centers.cpu().detach().numpy()
                #ins_label_gt = batch.ins_labels.cpu().detach().numpy()

                center_gt = centers_gt[:, 0]

                c_iou = (np.sum(np.logical_and(center_preds > 0.5, center_gt > 0.5))) / \
                        (np.sum(center_preds > 0.5) + np.sum(center_gt > 0.5) + 1e-10)
                c_ious.append(c_iou)
                s_ious.append(np.sum(center_preds > 0.5))

                # Save some of the frame pots
                if f_ind % 20 == 0:
                    seq_path = join(val_loader.dataset.path, 'sequences', val_loader.dataset.sequences[s_ind])
                    velo_file = join(seq_path, 'velodyne', val_loader.dataset.frames[s_ind][f_ind] + '.bin')
                    frame_points = np.fromfile(velo_file, dtype=np.float32)
                    frame_points = frame_points.reshape((-1, 4))
                    write_ply(filepath[:-4] + '_pots.ply',
                              [frame_points[:, :3], frame_labels, frame_preds],
                              ['x', 'y', 'z', 'gt', 'pre'])

                # Update validation confusions
                frame_C = fast_confusion(frame_labels,
                                         frame_preds.astype(np.int32),
                                         val_loader.dataset.label_values)
                val_loader.dataset.val_confs[s_ind][f_ind, :, :] = frame_C

                # Stack all prediction for this epoch
                predictions += [preds]
                targets += [frame_labels[proj_mask]]
                inds += [f_inds[b_i, :]]
                val_i += 1
                i0 += length

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * i / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (preds, truth) in enumerate(zip(predictions, targets)):
            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)

        t3 = time.time()

        #######################################
        # Results on this subpart of validation
        #######################################

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Balance with real validation proportions
        C *= np.expand_dims(val_loader.dataset.class_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        #####################################
        # Results on the whole validation set
        #####################################

        t4 = time.time()

        # Sum all validation confusions
        C_tot = [np.sum(seq_C, axis=0) for seq_C in val_loader.dataset.val_confs if len(seq_C) > 0]
        C_tot = np.sum(np.stack(C_tot, axis=0), axis=0)

        if debug:
            s = '\n'
            for cc in C_tot:
                for c in cc:
                    s += '{:8.1f} '.format(c)
                s += '\n'
            print(s)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C_tot = np.delete(C_tot, l_ind, axis=0)
                C_tot = np.delete(C_tot, l_ind, axis=1)

        # Objects IoU
        val_IoUs = IoU_from_confusions(C_tot)

        t5 = time.time()

        # Saving (optionnal)
        if config.saving:

            IoU_list = [IoUs, val_IoUs]
            file_list = ['subpart_IoUs.txt', 'val_IoUs.txt']
            for IoUs_to_save, IoU_file in zip(IoU_list, file_list):

                # Name of saving file
                test_file = join(config.saving_path, IoU_file)

                # Line to write:
                line = ''
                for IoU in IoUs_to_save:
                    line += '{:.3f} '.format(IoU)
                line = line + '\n'

                # Write in file
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        text_file.write(line)
                else:
                    with open(test_file, "w") as text_file:
                        text_file.write(line)

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} : subpart mIoU = {:.1f} %'.format(config.dataset, mIoU))
        mIoU = 100 * np.mean(val_IoUs)
        print('{:s} :     val mIoU = {:.1f} %'.format(config.dataset, mIoU))
        cIoU = 200 * np.mean(c_ious)
        print('{:s} :     val center mIoU = {:.1f} %'.format(config.dataset, cIoU))
        sIoU = np.mean(s_ious)
        print('{:s} :     val centers sum  = {:.1f} %'.format(config.dataset, sIoU))

        # Print accuracy mean
        accuracy_mean = sum(accuracies)/len(accuracies)
        print('\n************************\n')
        print("Accuracy" + str(accuracy_mean))
        print('\n************************\n')


        t6 = time.time()

        # Display timings
        if debug:
            print('\n************************\n')
            print('Validation timings:')
            print('Init ...... {:.1f}s'.format(t1 - t0))
            print('Loop ...... {:.1f}s'.format(t2 - t1))
            print('Confs ..... {:.1f}s'.format(t3 - t2))
            print('IoU1 ...... {:.1f}s'.format(t4 - t3))
            print('IoU2 ...... {:.1f}s'.format(t5 - t4))
            print('Save ...... {:.1f}s'.format(t6 - t5))
            print('\n************************\n')

        return

    def my_slam_segmentation_validation(self, net, val_loader, config, debug=True):
        """
        Validation method for slam segmentation models
        """
        t1 = time.time()

        accuracies = []

        # Start validation loop
        for i, batch in enumerate(val_loader):

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs, centers_output, var_output, embedding = net(batch, config)
                acc = net.accuracy(outputs, batch.labels)
                accuracies.append(acc)

        # Print accuracy mean
        accuracy_mean = (sum(accuracies)/len(accuracies))*100
        print('\n************************\n')
        print("Accuracy: " + str(accuracy_mean) + "%")
        print('\n************************\n')

        t2 = time.time()
        print("Time: " + str(t2-t1))
        print('\n************************\n')

        return