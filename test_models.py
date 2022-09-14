# Common libs
import signal
import os
import numpy as np
import sys
import torch

# Dataset
from datasets.SemanticKitti import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN

from models.prototype import PrototypeNet
from models.mpanet import MPAnet
from models.mpanet_binary import MPAnetBinary

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def model_choice(chosen_log):

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    chosen_log = '/globalwork/kreuzberg/4D-PLS/results/Log_2022-06-17_12-16-59'

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None

    # Choose to test on validation or test split
    on_val = True#False

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'
    if torch.cuda.device_count() > 1:
        GPU_ID = '0, 1'

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chkp = chosen_chkp.split('.')[0]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)


    ##################################
    # Change model parameters for test
    ##################################

    config.global_fet = False
    config.validation_size = 200
    config.input_threads = 16
    config.n_frames = 4
    #config.n_frames = 2
    config.n_test_frames = 4
    #config.n_test_frames = 2
    if config.n_frames < config.n_test_frames:
        config.n_frames = config.n_test_frames
    config.big_gpu = True
    config.dataset_task = '4d_panoptic'
    #config.sampling = 'density'
    config.sampling = 'importance'
    config.decay_sampling = 'None'
    config.stride = 1
    config.chosen_chkp = chkp
    config.pre_train = False

    config.dataset_path = '/globalwork/kreuzberg/SemanticKITTI/dataset'
    config.test_path = '/globalwork/kreuzberg/4D-PLS/test'


    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if config.dataset == 'SemanticKitti':
        test_dataset = SemanticKittiDataset(config, set=set, balance_classes=False, seqential_batch=True)
        test_sampler = SemanticKittiSampler(test_dataset)
        collate_fn = SemanticKittiCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=0,#config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()

    #net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    #net = PrototypeNet(config, test_dataset.label_values, test_dataset.ignored_labels)
    net = MPAnet(config, test_dataset.label_values, test_dataset.ignored_labels)
    #net = MPAnetBinary(config, test_dataset.label_values, test_dataset.ignored_labels)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')
    
    config.dataset_task = '4d_panoptic'
    
    # Testing

    #tester.panoptic_4d_test(net, test_loader, config)
    #tester.panoptic_4d_test_prototype(net, test_loader, config)
    tester.panoptic_4d_test_mpa(net, test_loader, config)