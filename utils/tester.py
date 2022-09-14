#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
from sympy import sequence
import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import json
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from sklearn.metrics import confusion_matrix

#from utils.visualizer import show_ModelNet_models

# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, chkp_path=None, on_gpu=True):

        ############
        # Parameters
        ############

        self.instances = {} #store instance ids and mean, cov fors sequantial prediction
        self.next_ins_id = 1 #next ins id for new instance

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        checkpoint = torch.load(chkp_path)
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        #net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")

        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def cloud_segmentation_test(self, net, test_loader, config, num_votes=100, debug=False):
        """
        Test method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        self.test_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_labels]

        # Test saving path
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'predictions')):
                makedirs(join(test_path, 'predictions'))
            if not exists(join(test_path, 'probs')):
                makedirs(join(test_path, 'probs'))
            if not exists(join(test_path, 'potentials')):
                makedirs(join(test_path, 'potentials'))
        else:
            test_path = None

        # If on validation directly compute score
        if test_loader.dataset.set == 'validation':
            val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in test_loader.dataset.label_values:
                if label_value not in test_loader.dataset.ignored_labels:
                    val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                 for labels in test_loader.dataset.validation_labels])
                    i += 1
        else:
            val_proportions = None

        #####################
        # Network predictions
        #####################

        test_epoch = 0
        last_min = -0.5

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                t += [time.time()]

                # Get probs and labels
                stacked_probs = softmax(outputs).cpu().detach().numpy()
                s_points = batch.points[0].cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()
                torch.cuda.synchronize(self.device)

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    points = s_points[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]
                    c_i = cloud_inds[b_i]

                    if 0 < test_radius_ratio < 1:
                        mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                        inds = inds[mask]
                        probs = probs[mask]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                    i0 += length

                # Average timing
                t += [time.time()]
                if i < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})'
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2])))

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.min_potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
            #print([np.mean(pots) for pots in test_loader.dataset.potentials])

            # Save predicted cloud
            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                # Show vote results (On subcloud so it is not the good values here)
                if test_loader.dataset.set == 'validation':
                    print('\nConfusion on sub clouds')
                    Confs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Insert false columns for ignored labels
                        probs = np.array(self.test_probs[i], copy=True)
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        # Predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                        # Targets
                        targets = test_loader.dataset.input_labels[i]

                        # Confs
                        Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print(s + '\n')

                # Save real IoU once in a while
                if int(np.ceil(new_min)) % 10 == 0:

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    proj_probs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        print(i, file_path, test_loader.dataset.test_proj[i].shape, self.test_probs[i].shape)

                        print(test_loader.dataset.test_proj[i].dtype, np.max(test_loader.dataset.test_proj[i]))
                        print(test_loader.dataset.test_proj[i][:5])

                        # Reproject probs on the evaluations points
                        probs = self.test_probs[i][test_loader.dataset.test_proj[i], :]
                        proj_probs += [probs]

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

                    # Show vote results
                    if test_loader.dataset.set == 'validation':
                        print('Confusion on full clouds')
                        t1 = time.time()
                        Confs = []
                        for i, file_path in enumerate(test_loader.dataset.files):

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    proj_probs[i] = np.insert(proj_probs[i], l_ind, 0, axis=1)

                            # Get the predicted labels
                            preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                            # Confusion
                            targets = test_loader.dataset.validation_labels[i]
                            Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                            if label_value in test_loader.dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        IoUs = IoU_from_confusions(C)
                        mIoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * mIoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        print('-' * len(s))
                        print(s)
                        print('-' * len(s) + '\n')

                    # Save predictions
                    print('Saving clouds')
                    t1 = time.time()
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Get file
                        points = test_loader.dataset.load_evaluation_points(file_path)

                        # Get the predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                        # Save plys
                        cloud_name = file_path.split('/')[-1]
                        test_name = join(test_path, 'predictions', cloud_name)
                        write_ply(test_name,
                                  [points, preds],
                                  ['x', 'y', 'z', 'preds'])
                        test_name2 = join(test_path, 'probs', cloud_name)
                        prob_names = ['_'.join(test_loader.dataset.label_to_names[label].split())
                                      for label in test_loader.dataset.label_values]
                        write_ply(test_name2,
                                  [points, proj_probs[i]],
                                  ['x', 'y', 'z'] + prob_names)

                        # Save potentials
                        pot_points = np.array(test_loader.dataset.pot_trees[i].data, copy=False)
                        pot_name = join(test_path, 'potentials', cloud_name)
                        pots = test_loader.dataset.potentials[i].numpy().astype(np.float32)
                        write_ply(pot_name,
                                  [pot_points.astype(np.float32), pots],
                                  ['x', 'y', 'z', 'pots'])

                        # Save ascii preds
                        if test_loader.dataset.set == 'test':
                            if test_loader.dataset.name.startswith('Semantic3D'):
                                ascii_name = join(test_path, 'predictions', test_loader.dataset.ascii_files[cloud_name])
                            else:
                                ascii_name = join(test_path, 'predictions', cloud_name[:-4] + '.txt')
                            np.savetxt(ascii_name, preds, fmt='%d')

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break

        return

    def slam_segmentation_test(self, net, test_loader, config, num_votes=100, debug=True):
        """
        Test method for slam segmentation models
        """

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.5
        last_min = -0.5
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes
        nc_model = net.C

        # Test saving path
        test_path = None
        report_path = None
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            report_path = join(test_path, 'reports')
            if not exists(report_path):
                makedirs(report_path)

        if test_loader.dataset.set == 'validation':
            for folder in ['val_predictions', 'val_probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))
        else:
            for folder in ['predictions', 'probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))

        # Init validation container
        all_f_preds = []
        all_f_labels = []
        if test_loader.dataset.set == 'validation':
            for i, seq_frames in enumerate(test_loader.dataset.frames):
                all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
                all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        test_epoch = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                if 'cuda' in self.device.type:
                    batch.to(self.device)
                with torch.no_grad():
                    outputs, centers_output, var_output, embedding = net(batch, config)
                    #ins_preds = torch.zeros(outputs.shape[0])

                    probs = softmax(outputs).cpu().detach().numpy()

                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            probs = np.insert(probs, l_ind, 0, axis=1)
                    preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)]
                    preds = torch.from_numpy(preds)
                    preds.to(outputs.device)
                    sequence = test_loader.dataset.sequences[batch.frame_inds[0][0]]
                    if sequence not in self.instances:
                        self.instances[sequence] = {}
                    ins_preds = net.ins_pred(preds, centers_output, var_output, embedding, batch.points, batch.times.unsqueeze(1))
                    #ins_preds, new_instances, ins_id = net.ins_pred_in_time(preds, centers_output, var_output, embedding, self.instances[sequence],
                    #                                 self.next_ins_id, batch.points)

                    #self.next_ins_id = ins_id#update next available ins id
                    #for ins_id, instance in new_instances.items(): #add new instances to history
                    #    self.instances[sequence][ins_id] = instance

                # Get probs and labels
                stk_probs = softmax(outputs).cpu().detach().numpy()
                ins_preds = ins_preds.cpu().detach().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                f_inds = batch.frame_inds.cpu().numpy()
                r_inds_list = batch.reproj_inds
                r_mask_list = batch.reproj_masks
                labels_list = batch.val_labels
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    probs = stk_probs[i0:i0 + length]
                    ins_probs = ins_preds[i0:i0 + length]
                    proj_inds = r_inds_list[b_i]
                    proj_mask = r_mask_list[b_i]
                    frame_labels = labels_list[b_i]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Project predictions on the frame points
                    proj_probs = probs[proj_inds]
                    proj_ins_probs = ins_probs[proj_inds]

                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_probs = np.expand_dims(proj_probs, 0)
                        proj_ins_probs = np.expand_dims(proj_ins_probs, 0)

                    # Save probs in a binary file (uint8 format for lighter weight)
                    seq_name = test_loader.dataset.sequences[s_ind]
                    if test_loader.dataset.set == 'validation':
                        folder = 'val_probs'
                        pred_folder = 'val_predictions'
                    else:
                        folder = 'probs'
                        pred_folder = 'predictions'
                    filename = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
                    filepath = join(test_path, folder, filename)
                    filename_i = '{:s}_{:07d}_i.npy'.format(seq_name, f_ind)
                    filepath_i = join(test_path, folder, filename_i)
                    #if exists(filepath):
                    #    frame_probs_uint8 = np.load(filepath)
                    #    ins_preds = np.load(filepath_i)
                    #else:
                    frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                    ins_preds = np.zeros((proj_mask.shape[0]))

                    frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
                    frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                    frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)
                    ins_preds[proj_mask] = proj_ins_probs

                    #np.save(filepath, frame_probs_uint8)
                    np.save(filepath_i, ins_preds)



                    # Save some prediction in ply format for visual
                    if test_loader.dataset.set == 'validation':

                        # Insert false columns for ignored labels
                        frame_probs_uint8_bis = frame_probs_uint8.copy()
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8_bis = np.insert(frame_probs_uint8_bis, l_ind, 0, axis=1)

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_bis,
                                                                                 axis=1)].astype(np.int32)

                        np.save(filepath, frame_preds)
                        # Save some of the frame pots
                        if f_ind % 20 == 0:
                            seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                            velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds, pots],
                                          ['x', 'y', 'z', 'gt', 'pre', 'pots'])
                            else:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds],
                                          ['x', 'y', 'z', 'gt', 'pre'])

                            # Also Save lbl probabilities
                            probpath = join(test_path, folder, filename[:-4] + '_probs.ply')
                            lbl_names = [test_loader.dataset.label_to_names[l]
                                         for l in test_loader.dataset.label_values
                                         if l not in test_loader.dataset.ignored_labels]
                            write_ply(probpath,
                                      [frame_points[:, :3], frame_probs_uint8],
                                      ['x', 'y', 'z'] + lbl_names)

                        # keep frame preds in memory
                        all_f_preds[s_ind][f_ind] = frame_preds
                        all_f_labels[s_ind][f_ind] = frame_labels

                    else:

                        # Insert false columns for ignored labels
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8 = np.insert(frame_probs_uint8, l_ind, 0, axis=1)

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8,
                                                                                 axis=1)].astype(np.int32)
                        np.save(filepath, frame_preds)
                        if f_inds[b_i, 1] % 100 == 0:
                            # Load points
                            seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                            velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds, pots],
                                          ['x', 'y', 'z', 'pre', 'pots'])
                            else:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds],
                                          ['x', 'y', 'z', 'pre'])

                    # Stack all prediction for this epoch
                    i0 += length

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f}) / pots {:d} => {:.1f}%'
                    min_pot = int(torch.floor(torch.min(test_loader.dataset.potentials)))
                    pot_num = torch.sum(test_loader.dataset.potentials > min_pot + 0.5).type(torch.int32).item()
                    current_num = pot_num + (i + 1 - config.validation_size) * config.val_batch_num
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2]),
                                         min_pot,
                                         100.0 * current_num / len(test_loader.dataset.potentials)))


            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))

            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                if test_loader.dataset.set == 'validation' and last_min % 1 == 0:

                    #####################################
                    # Results on the whole validation set
                    #####################################

                    # Confusions for our subparts of validation set
                    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
                    for i, (preds, truth) in enumerate(zip(predictions, targets)):

                        # Confusions
                        Confs[i, :, :] = fast_confusion(truth, preds, test_loader.dataset.label_values).astype(np.int32)


                    # Show vote results
                    print('\nCompute confusion')

                    val_preds = []
                    val_labels = []
                    t1 = time.time()
                    for i, seq_frames in enumerate(test_loader.dataset.frames):
                        val_preds += [np.hstack(all_f_preds[i])]
                        val_labels += [np.hstack(all_f_labels[i])]
                    val_preds = np.hstack(val_preds)
                    val_labels = np.hstack(val_labels)
                    t2 = time.time()
                    C_tot = fast_confusion(val_labels, val_preds, test_loader.dataset.label_values)
                    t3 = time.time()
                    print(' Stacking time : {:.1f}s'.format(t2 - t1))
                    print('Confusion time : {:.1f}s'.format(t3 - t2))

                    s1 = '\n'
                    for cc in C_tot:
                        for c in cc:
                            s1 += '{:7.0f} '.format(c)
                        s1 += '\n'
                    if debug:
                        print(s1)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C_tot = np.delete(C_tot, l_ind, axis=0)
                            C_tot = np.delete(C_tot, l_ind, axis=1)

                    # Objects IoU
                    val_IoUs = IoU_from_confusions(C_tot)

                    # Compute IoUs
                    mIoU = np.mean(val_IoUs)
                    s2 = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in val_IoUs:
                        s2 += '{:5.2f} '.format(100 * IoU)
                    print(s2 + '\n')

                    # Save a report
                    report_file = join(report_path, 'report_{:04d}.txt'.format(int(np.floor(last_min))))
                    str = 'Report of the confusion and metrics\n'
                    str += '***********************************\n\n\n'
                    str += 'Confusion matrix:\n\n'
                    str += s1
                    str += '\nIoU values:\n\n'
                    str += s2
                    str += '\n\n'
                    with open(report_file, 'w') as f:
                        f.write(str)

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break

        return

    def panoptic_4d_test(self, net, test_loader, config, num_votes=100, debug=True):
        """
        Test method for slam segmentation models
        """

        summed_number_of_clusters = 0
        t_start_inference = time.time()
        duration_relevant_summed = 0
        duration_relevant_help_summed = 0
        duration_relevant_help_get_item_summed = 0

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.5
        last_min = -0.5
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes
        nc_model = net.C
        # Test saving path
        test_path = None
        report_path = None

        if config.dataset_task == '4d_panoptic':

            #assoc_saving = [asc_type for idx, asc_type in enumerate(config.association_types) if config.association_weights[idx] > 0]
            #assoc_saving.append(str(config.n_test_frames))
            #assoc_saving = '_'.join(assoc_saving)
            assoc_saving = config.sampling
            config.assoc_saving = config.sampling+'_'+ config.decay_sampling
            if hasattr(config, 'stride'):
                config.assoc_saving = config.sampling + '_' + config.decay_sampling+ '_str' + str(config.stride) +'_'
            if hasattr(config, 'big_gpu') and config.big_gpu:
                config.assoc_saving = config.assoc_saving + 'bigpug_'


        if config.saving:
            test_path = join(config.test_path, config.saving_path.split('/')[-1]+ '_'+config.assoc_saving+str(config.n_test_frames) + '_' + config.chosen_chkp)
            if not exists(test_path):
                makedirs(test_path)
            report_path = join(test_path, 'reports')
            if not exists(report_path):
                makedirs(report_path)

        if test_loader.dataset.set == 'validation':
            for folder in ['val_predictions', 'val_probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))
        else:
            for folder in ['predictions', 'probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))

        # Init validation container
        all_f_preds = []
        all_f_labels = []
        if test_loader.dataset.set == 'validation':
            for i, seq_frames in enumerate(test_loader.dataset.frames):
                all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
                all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        test_epoch = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        processed = 0 #number of frames that processed

        # Start test loop
        while True:
            print('Initialize workers')
            t_start_step = time.time()
            for i, batch in enumerate(test_loader):
                t_end_get_item = time.time()
                duration_get_item = t_end_get_item - t_start_step
                print('duration_get_item: ' + str(duration_get_item))

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                flag = True
                if config.n_test_frames > 1:
                    lengths = batch.lengths[0].cpu().numpy()
                    for b_i, length in enumerate(lengths):
                        f_inds = batch.frame_inds.cpu().numpy()
                        f_ind = f_inds[b_i, 1]
                        if f_ind % config.n_test_frames != config.n_test_frames-1:
                             flag = False

                if processed == test_loader.dataset.all_inds.shape[0]:
                    return
                #if not flag:
                #    continue
                #else:
                processed +=1

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                with torch.no_grad():
                    
                    t_start_forward_pass = time.time()
                    outputs, centers_output, var_output, embedding = net(batch, config)
                    t_end_forward_pass = time.time()
                    duration_forward_pass = t_end_forward_pass - t_start_forward_pass
                    print('duration_forward_pass: ' + str(duration_forward_pass))
                    #ins_preds = torch.zeros(outputs.shape[0])

                    probs = softmax(outputs)
                    probs = probs.cpu().detach().numpy()

                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            probs = np.insert(probs, l_ind, 0, axis=1)
                    preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)]
                    preds = torch.from_numpy(preds)
                    preds.to(outputs.device)
                    sequence = test_loader.dataset.sequences[batch.frame_inds[0][0]]
                    pose = test_loader.dataset.poses[batch.frame_inds[0][0]][batch.frame_inds[0][1]]
                    if sequence not in self.instances:
                        self.instances[sequence] = {}

                    nuscene = False

                    t_start_ins_pred = time.time()
                    #ins_preds = net.ins_pred(preds, centers_output, var_output, embedding, batch.points)
                    #ins_preds, new_instances, ins_id = net.ins_pred_in_time(config, preds, centers_output, var_output, embedding, self.instances[sequence], self.next_ins_id, batch.points, batch.times.unsqueeze(1), pose)
                    ins_preds, new_instances, ins_id, center_points = net.ins_pred_in_time(config, preds, centers_output, var_output, embedding, self.instances[sequence], self.next_ins_id, batch.points, batch.times.unsqueeze(1), pose)
                    #ins_preds, new_instances, ins_id, center_points, proposals, proposals_center = net.ins_pred_in_time_mpa(config, preds, centers_output, var_output, embedding, self.next_ins_id, batch.points, batch.times.unsqueeze(1))
                    #ins_preds, new_instances, ins_id, center_points, proposals_center = net.ins_pred_in_time_mpa_new(config, preds, centers_output, var_output, embedding, self.next_ins_id, batch.points, batch.times.unsqueeze(1))

                    t_end_ins_pred = time.time()
                    duration_ins_pred = t_end_ins_pred - t_start_ins_pred
                    print('duration_ins_pred: ' + str(duration_ins_pred))

                    duration_relevant = duration_forward_pass + duration_ins_pred
                    print('duration_relevant: ' + str(duration_relevant))

                    duration_relevant_summed = duration_relevant_summed + duration_relevant
                    print('duration_relevant_summed: ' + str(duration_relevant_summed))

                    duration_relevant_help = t_end_ins_pred - t_start_forward_pass
                    print('duration_relevant_help: ' + str(duration_relevant_help))

                    duration_relevant_help_summed = duration_relevant_help_summed + duration_relevant_help
                    print('duration_relevant_help_summed: ' + str(duration_relevant_help_summed))

                    duration_relevant_help_get_item = t_end_ins_pred - t_start_step
                    print('duration_relevant_help_get_item: ' + str(duration_relevant_help_get_item))

                    duration_relevant_help_get_item_summed = duration_relevant_help_get_item_summed + duration_relevant_help_get_item
                    print('duration_relevant_help_get_item_summed: ' + str(duration_relevant_help_get_item_summed))

                    number_of_clusters = ins_id - self.next_ins_id
                    summed_number_of_clusters = summed_number_of_clusters + number_of_clusters
                    print('summed_number_of_clusters: ' + str(summed_number_of_clusters))

                    self.next_ins_id = ins_id#update next available ins id
                    for ins_id, instance in new_instances.items(): #add new instances to history
                        self.instances[sequence][ins_id] = instance

                    dont_track_ids = []
                    for ins_id in self.instances[sequence].keys():
                        if self.instances[sequence][ins_id]['life'] == 0:
                            dont_track_ids.append(ins_id)
                        self.instances[sequence][ins_id]['life'] -= 1

                    for ins_id in dont_track_ids:
                        del self.instances[sequence][ins_id]
                
                t_start_store_data = time.time()

                # Get probs and labels
                stk_probs = softmax(outputs).cpu().detach().numpy()
                ins_preds = ins_preds.cpu().detach().numpy()
                centers_output = centers_output.cpu().detach().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                f_inds = batch.frame_inds.cpu().numpy()
                r_inds_list = batch.reproj_inds
                r_mask_list = batch.reproj_masks
                f_inc_r_inds_list = batch.f_inc_reproj_inds
                f_inc_r_mask_list = batch.f_inc_reproj_masks

                labels_list = batch.val_labels
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    probs = stk_probs[i0:i0 + length]
                    c_probs = centers_output[i0:i0 + length]
                    ins_probs = ins_preds[i0:i0 + length]
                    proj_inds = r_inds_list[b_i]
                    proj_mask = r_mask_list[b_i]
                    frame_labels = labels_list[b_i]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Add 4D visualization here
                    #'''
                    folder_4d = "4D_val_probs"
                    if not exists(join(test_path, folder_4d)):
                        makedirs(join(test_path, folder_4d))
                    seq_name_4d = test_loader.dataset.sequences[s_ind]
                    filename_4d = '{:s}_{:07d}.npy'.format(seq_name_4d, f_ind)
                    filename_i_4d = '{:s}_{:07d}_i.npy'.format(seq_name_4d, f_ind)
                    filename_c_4d = '{:s}_{:07d}_c.npy'.format(seq_name_4d, f_ind)
                    filename_p_4d = '{:s}_{:07d}_p.npy'.format(seq_name_4d, f_ind)
                    filename_cp_4d = '{:s}_{:07d}_cp.npy'.format(seq_name_4d, f_ind)
                    filename_gt_4d = '{:s}_{:07d}_gt.npy'.format(seq_name_4d, f_ind)
                    filename_gt_i_4d = '{:s}_{:07d}_gt_i.npy'.format(seq_name_4d, f_ind)
                    filename_gt_c_4d = '{:s}_{:07d}_gt_c.npy'.format(seq_name_4d, f_ind)
                    filename_prop_ids_4d = '{:s}_{:07d}_prop_ids.npy'.format(seq_name_4d, f_ind)
                    filename_prop_c_4d = '{:s}_{:07d}_prop_c.npy'.format(seq_name_4d, f_ind)
                    filepath_4d = join(test_path, folder_4d, filename_4d)
                    filepath_i_4d = join(test_path, folder_4d, filename_i_4d)
                    filepath_c_4d = join(test_path, folder_4d, filename_c_4d)
                    filepath_p_4d = join(test_path, folder_4d, filename_p_4d)
                    filepath_cp_4d = join(test_path, folder_4d, filename_cp_4d)
                    filepath_gt_4d = join(test_path, folder_4d, filename_gt_4d)
                    filepath_gt_i_4d = join(test_path, folder_4d, filename_gt_i_4d)
                    filepath_gt_c_4d = join(test_path, folder_4d, filename_gt_c_4d)
                    filepath_prop_ids_4d = join(test_path, folder_4d, filename_prop_ids_4d)
                    filepath_prop_c_4d = join(test_path, folder_4d, filename_prop_c_4d)

                    batch_labels = batch.labels.cpu().numpy()
                    #batch_ins_labels = batch.ins_labels.cpu().numpy()
                    batch_ins_labels_help = batch.ins_labels_help
                    #help = batch_ins_labels_help[batch_ins_labels_help != 0]
                    #help = np.unique(help)
                    #help = np.sort(help)

                    #np.save(filepath_p_4d, batch.points[0].cpu().numpy())
                    #np.save(filepath_c_4d, c_probs)
                    #np.save(filepath_i_4d, ins_probs)
                    #np.save(filepath_cp_4d, center_points)
                    #np.save(filepath_gt_4d, batch_labels)
                    #np.save(filepath_gt_i_4d, batch_ins_labels_help)
                    #np.save(filepath_gt_c_4d, batch.centers.cpu().numpy())

                    sem_probs_uint8 = np.zeros(probs.shape, dtype=np.uint8)
                    sem_probs = sem_probs_uint8.astype(np.float32) / 255
                    sem_probs = test_smooth * sem_probs + (1 - test_smooth) * probs
                    sem_probs_uint8 = (sem_probs * 255).astype(np.uint8)
                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            sem_probs_uint8 = np.insert(sem_probs_uint8, l_ind, 0, axis=1)
                    sem_preds = test_loader.dataset.label_values[np.argmax(sem_probs_uint8,
                                                                           axis=1)].astype(np.int32)
                    #np.save(filepath_4d, sem_preds)

                    #np.save(filepath_prop_ids_4d, proposals)
                    #np.save(filepath_prop_c_4d, proposals_center)
                    #'''
                    # end added

                    # Project predictions on the frame points
                    proj_probs = probs[proj_inds]
                    proj_c_probs = c_probs[proj_inds]
                    proj_ins_probs = ins_probs[proj_inds]

                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_c_probs = np.expand_dims(proj_c_probs, 0)
                        proj_probs = np.expand_dims(proj_probs, 0)
                        proj_ins_probs = np.expand_dims(proj_ins_probs, 0)

                    # Save probs in a binary file (uint8 format for lighter weight)
                    seq_name = test_loader.dataset.sequences[s_ind]
                    if test_loader.dataset.set == 'validation':
                        folder = 'val_probs'
                        pred_folder = 'val_predictions'
                    else:
                        folder = 'probs'
                        pred_folder = 'predictions'
                    filename = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
                    filepath = join(test_path, folder, filename)
                    filename_i = '{:s}_{:07d}_i.npy'.format(seq_name, f_ind)
                    filename_c = '{:s}_{:07d}_c.npy'.format(seq_name, f_ind)
                    filepath_i = join(test_path, folder, filename_i)
                    filepath_c = join(test_path, folder, filename_c)

                    #if exists(filepath):
                    #    frame_probs_uint8 = np.load(filepath)
                    #    ins_preds = np.load(filepath_i)
                    #else:
                    frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                    frame_c_probs = np.zeros((proj_mask.shape[0], 1))
                    ins_preds = np.zeros((proj_mask.shape[0]))

                    frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
                    frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                    frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)
                    ins_preds[proj_mask] = proj_ins_probs
                    frame_c_probs[proj_mask] = proj_c_probs

                    #np.save(filepath, frame_probs_uint8)
                    #print ('Saving {}'.format(filepath_i))
                    np.save(filepath_i, ins_preds)
                    np.save(filepath_c, frame_c_probs)

                    ins_features = {}
                    for ins_id in np.unique(ins_preds):
                        if int(ins_id) in self.instances[sequence]:
                            ins_features[int(ins_id)] = self.instances[sequence][int(ins_id)]['mean']
                    filename_f = '{:s}_{:07d}_f.npy'.format(seq_name, f_ind)
                    filepath_f = join(test_path, folder, filename_f)
                    #np.save(filepath_f, ins_features)

                    if config.n_test_frames > 1:
                        for fi in range(len(f_inc_r_inds_list[b_i])):
                            proj_inds = f_inc_r_inds_list[b_i][fi]
                            proj_mask = f_inc_r_mask_list[b_i][fi]
                            proj_ins_probs = ins_probs[proj_inds]
                            proj_probs = probs[proj_inds]
                            if proj_probs.ndim < 2:
                                proj_ins_probs = np.expand_dims(proj_ins_probs, 0)
                                proj_probs = np.expand_dims(proj_probs, 0)

                            frame_probs_uint8_p = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                            frame_probs = frame_probs_uint8_p[proj_mask, :].astype(np.float32) / 255
                            frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                            frame_probs_uint8_p[proj_mask, :] = (frame_probs * 255).astype(np.uint8)

                            ins_preds = np.zeros((proj_mask.shape[0]))
                            ins_preds[proj_mask] = proj_ins_probs

                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    frame_probs_uint8_p = np.insert(frame_probs_uint8_p, l_ind, 0, axis=1)

                                # Predicted labels
                            frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_p,
                                                                                     axis=1)].astype(np.int32)

                            filename_i = '{:s}_{:07d}_{}_i.npy'.format(seq_name, f_ind - fi - 1, f_ind)
                            filepath_i = join(test_path, folder, filename_i)
                            #('Saving {}'.format(filepath_i))
                            np.save(filepath_i, ins_preds)

                            filename_p = '{:s}_{:07d}_{}.npy'.format(seq_name, f_ind-fi-1, f_ind)
                            filepath_p = join(test_path, folder, filename_p)
                            #print('Saving {}'.format(filepath_p))
                            np.save(filepath_p, frame_preds)

                            ins_features = {}
                            for ins_id in np.unique(ins_preds):
                                if int(ins_id) in self.instances[sequence]:
                                    ins_features[int(ins_id)] = self.instances[sequence][int(ins_id)]['mean']
                            filename_f = '{:s}_{:07d}_{}_f.npy'.format(seq_name, f_ind-fi-1, f_ind)
                            filepath_f = join(test_path, folder, filename_f)
                            #np.save(filepath_f, ins_features)

                    # Save some prediction in ply format for visual
                    if test_loader.dataset.set == 'validation':

                        # Insert false columns for ignored labels
                        frame_probs_uint8_bis = frame_probs_uint8.copy()
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8_bis = np.insert(frame_probs_uint8_bis, l_ind, 0, axis=1)

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_bis,
                                                                                 axis=1)].astype(np.int32)

                        np.save(filepath, frame_preds)
                        #print('Saving {}'.format(filepath))
                        # Save some of the frame pots
                        if f_ind % 20 == 0:
                            seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                            #velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                            if nuscene:
                                velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.npy')
                                frame_points = np.load(velo_file)                                
                            else:
                                velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                                frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds, pots],
                                          ['x', 'y', 'z', 'gt', 'pre', 'pots'])
                            else:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds],
                                          ['x', 'y', 'z', 'gt', 'pre'])

                            # Also Save lbl probabilities
                            probpath = join(test_path, folder, filename[:-4] + '_probs.ply')
                            lbl_names = [test_loader.dataset.label_to_names[l]
                                         for l in test_loader.dataset.label_values
                                         if l not in test_loader.dataset.ignored_labels]
                            write_ply(probpath,
                                      [frame_points[:, :3], frame_probs_uint8],
                                      ['x', 'y', 'z'] + lbl_names)

                        # keep frame preds in memory
                        all_f_preds[s_ind][f_ind] = frame_preds
                        all_f_labels[s_ind][f_ind] = frame_labels

                    else:

                        # Insert false columns for ignored labels
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8 = np.insert(frame_probs_uint8, l_ind, 0, axis=1)

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8,
                                                                                 axis=1)].astype(np.int32)
                        np.save(filepath, frame_preds)
                        if f_inds[b_i, 1] % 100 == 0:
                            # Load points
                            seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                            velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                            if nuscene:
                                velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.npy')
                                frame_points = np.load(velo_file)
                            else:
                                velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                                frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds, pots],
                                          ['x', 'y', 'z', 'pre', 'pots'])
                            else:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds],
                                          ['x', 'y', 'z', 'pre'])

                    # Stack all prediction for this epoch
                    i0 += length

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                t_end = time.time()
                
                duration_store_data = t_end - t_start_store_data
                print('duration_store_data: ' + str(duration_store_data))

                duration_step = t_end - t_start_step
                print('duration_step: ' + str(duration_step))

                duration_inference = t_end - t_start_inference
                print('duration_inference: ' + str(duration_inference))

                t_start_step = time.time()

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f}) / pots {:d} => {:.1f}%'
                    min_pot = int(torch.floor(torch.min(test_loader.dataset.potentials)))
                    pot_num = torch.sum(test_loader.dataset.potentials > min_pot + 0.5).type(torch.int32).item()
                    current_num = pot_num + (i + 1 - config.validation_size) * config.val_batch_num
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2]),
                                         min_pot,
                                         100.0 * current_num / len(test_loader.dataset.potentials)))


            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))

            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                if test_loader.dataset.set == 'validation' and last_min % 1 == 0:

                    #####################################
                    # Results on the whole validation set
                    #####################################

                    # Confusions for our subparts of validation set
                    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
                    for i, (preds, truth) in enumerate(zip(predictions, targets)):

                        # Confusions
                        Confs[i, :, :] = fast_confusion(truth, preds, test_loader.dataset.label_values).astype(np.int32)


                    # Show vote results
                    print('\nCompute confusion')

                    val_preds = []
                    val_labels = []
                    t1 = time.time()
                    for i, seq_frames in enumerate(test_loader.dataset.frames):
                        val_preds += [np.hstack(all_f_preds[i])]
                        val_labels += [np.hstack(all_f_labels[i])]
                    val_preds = np.hstack(val_preds)
                    val_labels = np.hstack(val_labels)
                    t2 = time.time()
                    C_tot = fast_confusion(val_labels, val_preds, test_loader.dataset.label_values)
                    t3 = time.time()
                    print(' Stacking time : {:.1f}s'.format(t2 - t1))
                    print('Confusion time : {:.1f}s'.format(t3 - t2))

                    s1 = '\n'
                    for cc in C_tot:
                        for c in cc:
                            s1 += '{:7.0f} '.format(c)
                        s1 += '\n'
                    if debug:
                        print(s1)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C_tot = np.delete(C_tot, l_ind, axis=0)
                            C_tot = np.delete(C_tot, l_ind, axis=1)

                    # Objects IoU
                    val_IoUs = IoU_from_confusions(C_tot)

                    # Compute IoUs
                    mIoU = np.mean(val_IoUs)
                    s2 = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in val_IoUs:
                        s2 += '{:5.2f} '.format(100 * IoU)
                    print(s2 + '\n')

                    # Save a report
                    report_file = join(report_path, 'report_{:04d}.txt'.format(int(np.floor(last_min))))
                    strg = 'Report of the confusion and metrics\n'
                    strg += '***********************************\n\n\n'
                    strg += 'Confusion matrix:\n\n'
                    strg += s1
                    strg += '\nIoU values:\n\n'
                    strg += s2
                    strg += '\n\n'
                    with open(report_file, 'w') as f:
                        f.write(strg)

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break
            #break

        averaged_number_of_clusters = summed_number_of_clusters / 4071
        print('averaged_number_of_clusters: ' + str(averaged_number_of_clusters))

        return

    
    def panoptic_4d_test_prototype(self, net, test_loader, config, num_votes=100, debug=True):
        """
        Test method for slam segmentation models
        """

        t_start_inference = time.time()
        duration_relevant_summed = 0
        duration_relevant_help_summed = 0
        duration_relevant_help_get_item_summed = 0

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.5
        last_min = -0.5
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes
        nc_model = net.C
        # Test saving pathtest_prot
        test_path = None
        report_path = None

        if config.dataset_task == '4d_panoptic':

            #assoc_saving = [asc_type for idx, asc_type in enumerate(config.association_types) if config.association_weights[idx] > 0]
            #assoc_saving.append(str(config.n_test_frames))
            #assoc_saving = '_'.join(assoc_saving)
            assoc_saving = config.sampling
            config.assoc_saving = config.sampling+'_'+ config.decay_sampling
            if hasattr(config, 'stride'):
                config.assoc_saving = config.sampling + '_' + config.decay_sampling+ '_str' + str(config.stride) +'_'
            if hasattr(config, 'big_gpu') and config.big_gpu:
                config.assoc_saving = config.assoc_saving + 'bigpug_'


        if config.saving:
            test_path = join(config.test_path, config.saving_path.split('/')[-1]+ '_'+config.assoc_saving+str(config.n_test_frames) + '_' + config.chosen_chkp)
            if not exists(test_path):
                makedirs(test_path)
            report_path = join(test_path, 'reports')
            if not exists(report_path):
                makedirs(report_path)

        if test_loader.dataset.set == 'validation':
            for folder in ['val_predictions', 'val_probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))
        else:
            for folder in ['predictions', 'probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))

        # Init validation container
        all_f_preds = []
        all_f_labels = []
        if test_loader.dataset.set == 'validation':
            for i, seq_frames in enumerate(test_loader.dataset.frames):
                all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
                all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        test_epoch = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        processed = 0 #number of frames that processed

        # Start test loop
        while True:
            print('Initialize workers')
            t_start_step = time.time()
            for i, batch in enumerate(test_loader):
                t_end_get_item = time.time()
                duration_get_item = t_end_get_item - t_start_step
                print('duration_get_item: ' + str(duration_get_item))

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                flag = True
                if config.n_test_frames > 1:
                    lengths = batch.lengths[0].cpu().numpy()
                    for b_i, length in enumerate(lengths):
                        f_inds = batch.frame_inds.cpu().numpy()
                        f_ind = f_inds[b_i, 1]
                        if f_ind % config.n_test_frames != config.n_test_frames-1:
                             flag = False

                if processed == test_loader.dataset.all_inds.shape[0]:
                    return
                #if not flag:
                #    continue
                #else:
                processed +=1

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                with torch.no_grad():
                    
                    t_start_forward_pass = time.time()
                    outputs, point_votes, centers_output = net(batch)
                    t_end_forward_pass = time.time()
                    duration_forward_pass = t_end_forward_pass - t_start_forward_pass
                    print('duration_forward_pass: ' + str(duration_forward_pass))

                    #ins_preds = torch.zeros(outputs.shape[0])

                    probs = softmax(outputs)
                    probs = probs.cpu().detach().numpy()

                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            probs = np.insert(probs, l_ind, 0, axis=1)
                    preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)]
                    preds = torch.from_numpy(preds)
                    preds.to(outputs.device)
                    
                    sequence = test_loader.dataset.sequences[batch.frame_inds[0][0]]
                    pose = test_loader.dataset.poses[batch.frame_inds[0][0]][batch.frame_inds[0][1]]
                    if sequence not in self.instances:
                        self.instances[sequence] = {}

                    t_start_ins_pred = time.time()

                    #ins_preds = net.ins_pred(preds, centers_output, var_output, embedding, batch.points)
                    #ins_preds, new_instances, ins_id, center_points = net.ins_pred_in_time(config, preds, centers_output, var_output, embedding, self.instances[sequence], self.next_ins_id, batch.points, batch.times.unsqueeze(1), pose)
                    #ins_preds, ins_id, proposals, proposals_center = net.ins_pred_in_time_mpa(preds, point_votes, self.next_ins_id)
                    ins_preds, ins_id, proposals, proposals_center, point_semantic_classses = net.ins_pred_in_time_mpa(preds, point_votes, self.next_ins_id)
                    #ins_preds, ins_id, proposals, proposals_center, point_semantic_classses = net.ins_pred_in_time_mpa_bb(preds, point_votes, batch.points[0], self.next_ins_id)
                    preds = point_semantic_classses #!!!
                    majority_voting = True #!!!

                    t_end_ins_pred = time.time()
                    duration_ins_pred = t_end_ins_pred - t_start_ins_pred
                    print('duration_ins_pred: ' + str(duration_ins_pred))

                    duration_relevant = duration_forward_pass + duration_ins_pred
                    print('duration_relevant: ' + str(duration_relevant))

                    duration_relevant_summed = duration_relevant_summed + duration_relevant
                    print('duration_relevant_summed: ' + str(duration_relevant_summed))

                    duration_relevant_help = t_end_ins_pred - t_start_forward_pass
                    print('duration_relevant_help: ' + str(duration_relevant_help))

                    duration_relevant_help_summed = duration_relevant_help_summed + duration_relevant_help
                    print('duration_relevant_help_summed: ' + str(duration_relevant_help_summed))

                    duration_relevant_help_get_item = t_end_ins_pred - t_start_step
                    print('duration_relevant_help_get_item: ' + str(duration_relevant_help_get_item))

                    duration_relevant_help_get_item_summed = duration_relevant_help_get_item_summed + duration_relevant_help_get_item
                    print('duration_relevant_help_get_item_summed: ' + str(duration_relevant_help_get_item_summed))

                    self.next_ins_id = ins_id#update next available ins id
                    #for ins_id, instance in new_instances.items(): #add new instances to history
                    #    self.instances[sequence][ins_id] = instance

                    dont_track_ids = []
                    for ins_id in self.instances[sequence].keys():
                        if self.instances[sequence][ins_id]['life'] == 0:
                            dont_track_ids.append(ins_id)
                        self.instances[sequence][ins_id]['life'] -= 1

                    for ins_id in dont_track_ids:
                        del self.instances[sequence][ins_id]

                t_start_store_data = time.time()
                
                # Get probs and labels
                stk_probs = softmax(outputs).cpu().detach().numpy()
                #ins_preds = ins_preds.cpu().detach().numpy()
                centers_output = centers_output.cpu().detach().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                f_inds = batch.frame_inds.cpu().numpy()
                r_inds_list = batch.reproj_inds
                r_mask_list = batch.reproj_masks
                f_inc_r_inds_list = batch.f_inc_reproj_inds
                f_inc_r_mask_list = batch.f_inc_reproj_masks

                labels_list = batch.val_labels
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    probs = stk_probs[i0:i0 + length]
                    c_probs = centers_output[i0:i0 + length]
                    ins_probs = ins_preds[i0:i0 + length]
                    semantic_predictions_mv = preds[i0:i0 + length] #!!!
                    proj_inds = r_inds_list[b_i]
                    proj_mask = r_mask_list[b_i]
                    frame_labels = labels_list[b_i]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Add 4D visualization here
                    #'''
                    folder_4d = "4D_val_probs"
                    if not exists(join(test_path, folder_4d)):
                        makedirs(join(test_path, folder_4d))
                    seq_name_4d = test_loader.dataset.sequences[s_ind]
                    filename_4d = '{:s}_{:07d}.npy'.format(seq_name_4d, f_ind)
                    filename_i_4d = '{:s}_{:07d}_i.npy'.format(seq_name_4d, f_ind)
                    filename_c_4d = '{:s}_{:07d}_c.npy'.format(seq_name_4d, f_ind)
                    filename_p_4d = '{:s}_{:07d}_p.npy'.format(seq_name_4d, f_ind)
                    filename_cp_4d = '{:s}_{:07d}_cp.npy'.format(seq_name_4d, f_ind)
                    filename_gt_4d = '{:s}_{:07d}_gt.npy'.format(seq_name_4d, f_ind)
                    filename_gt_i_4d = '{:s}_{:07d}_gt_i.npy'.format(seq_name_4d, f_ind)
                    filename_gt_c_4d = '{:s}_{:07d}_gt_c.npy'.format(seq_name_4d, f_ind)
                    filename_prop_ids_4d = '{:s}_{:07d}_prop_ids.npy'.format(seq_name_4d, f_ind)
                    filename_prop_c_4d = '{:s}_{:07d}_prop_c.npy'.format(seq_name_4d, f_ind)
                    filename_v_4d = '{:s}_{:07d}_v.npy'.format(seq_name_4d, f_ind)
                    filepath_4d = join(test_path, folder_4d, filename_4d)
                    filepath_i_4d = join(test_path, folder_4d, filename_i_4d)
                    filepath_c_4d = join(test_path, folder_4d, filename_c_4d)
                    filepath_p_4d = join(test_path, folder_4d, filename_p_4d)
                    filepath_cp_4d = join(test_path, folder_4d, filename_cp_4d)
                    filepath_gt_4d = join(test_path, folder_4d, filename_gt_4d)
                    filepath_gt_i_4d = join(test_path, folder_4d, filename_gt_i_4d)
                    filepath_gt_c_4d = join(test_path, folder_4d, filename_gt_c_4d)
                    filepath_prop_ids_4d = join(test_path, folder_4d, filename_prop_ids_4d)
                    filepath_prop_c_4d = join(test_path, folder_4d, filename_prop_c_4d)
                    filepath_v_4d = join(test_path, folder_4d, filename_v_4d)

                    batch_labels = batch.labels.cpu().numpy()
                    #batch_ins_labels = batch.ins_labels.cpu().numpy()
                    batch_ins_labels_help = batch.ins_labels_help
                    #help = batch_ins_labels_help[batch_ins_labels_help != 0]
                    #help = np.unique(help)
                    #help = np.sort(help)

                    #np.save(filepath_p_4d, batch.points[0].cpu().numpy())
                    #np.save(filepath_c_4d, c_probs)
                    #np.save(filepath_i_4d, ins_probs)
                    #np.save(filepath_cp_4d, center_points)
                    #np.save(filepath_gt_4d, batch_labels)
                    #np.save(filepath_gt_i_4d, batch_ins_labels_help)
                    #np.save(filepath_gt_c_4d, batch.centers.cpu().numpy())

                    sem_probs_uint8 = np.zeros(probs.shape, dtype=np.uint8)
                    sem_probs = sem_probs_uint8.astype(np.float32) / 255
                    sem_probs = test_smooth * sem_probs + (1 - test_smooth) * probs
                    sem_probs_uint8 = (sem_probs * 255).astype(np.uint8)
                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            sem_probs_uint8 = np.insert(sem_probs_uint8, l_ind, 0, axis=1)
                    sem_preds = test_loader.dataset.label_values[np.argmax(sem_probs_uint8,
                                                                           axis=1)].astype(np.int32)
                    #if majority_voting:
                    #    np.save(filepath_4d, semantic_predictions_mv) #!!!
                    #else:
                    #    np.save(filepath_4d, sem_preds) #!!!

                    #np.save(filepath_prop_ids_4d, proposals)
                    #np.save(filepath_prop_c_4d, proposals_center)
    
                    point_votes = point_votes.squeeze()
                    point_votes = point_votes.cpu().detach().numpy()
                    ins_ids = np.where((point_semantic_classses < 9) & (point_semantic_classses != 0))[0]
                    point_votes = point_votes[ins_ids]
                    #np.save(filepath_v_4d, point_votes)
                    #'''
                    # end added

                    # Project predictions on the frame points
                    proj_probs = probs[proj_inds]
                    proj_c_probs = c_probs[proj_inds]
                    proj_ins_probs = ins_probs[proj_inds]
                    proj_semantic_predictions_mv = semantic_predictions_mv[proj_inds] #!!!

                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_c_probs = np.expand_dims(proj_c_probs, 0)
                        proj_probs = np.expand_dims(proj_probs, 0)
                        proj_ins_probs = np.expand_dims(proj_ins_probs, 0)
                        proj_semantic_predictions_mv = np.expand_dims(proj_semantic_predictions_mv, 0)

                    # Save probs in a binary file (uint8 format for lighter weight)
                    seq_name = test_loader.dataset.sequences[s_ind]
                    if test_loader.dataset.set == 'validation':
                        folder = 'val_probs'
                        pred_folder = 'val_predictions'
                    else:
                        folder = 'probs'
                        pred_folder = 'predictions'
                    filename = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
                    filepath = join(test_path, folder, filename)
                    filename_i = '{:s}_{:07d}_i.npy'.format(seq_name, f_ind)
                    filename_c = '{:s}_{:07d}_c.npy'.format(seq_name, f_ind)
                    filepath_i = join(test_path, folder, filename_i)
                    filepath_c = join(test_path, folder, filename_c)

                    #if exists(filepath):
                    #    frame_probs_uint8 = np.load(filepath)
                    #    ins_preds = np.load(filepath_i)
                    #else:
                    frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                    frame_c_probs = np.zeros((proj_mask.shape[0], 1))
                    ins_preds = np.zeros((proj_mask.shape[0]))
                    frame_semantic_predictions = np.zeros((proj_mask.shape[0])) #!!!

                    frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
                    frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                    frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)
                    ins_preds[proj_mask] = proj_ins_probs
                    frame_c_probs[proj_mask] = proj_c_probs
                    frame_semantic_predictions[proj_mask] = proj_semantic_predictions_mv #!!!

                    #np.save(filepath, frame_probs_uint8)
                    #print ('Saving {}'.format(filepath_i))
                    np.save(filepath_i, ins_preds)
                    np.save(filepath_c, frame_c_probs)
                    if majority_voting:
                        np.save(filepath, frame_semantic_predictions) #!!!

                    ins_features = {}
                    for ins_id in np.unique(ins_preds):
                        if int(ins_id) in self.instances[sequence]:
                            ins_features[int(ins_id)] = self.instances[sequence][int(ins_id)]['mean']
                    filename_f = '{:s}_{:07d}_f.npy'.format(seq_name, f_ind)
                    filepath_f = join(test_path, folder, filename_f)
                    #np.save(filepath_f, ins_features)

                    if config.n_test_frames > 1:
                        for fi in range(len(f_inc_r_inds_list[b_i])):
                            proj_inds = f_inc_r_inds_list[b_i][fi]
                            proj_mask = f_inc_r_mask_list[b_i][fi]
                            proj_ins_probs = ins_probs[proj_inds]
                            proj_probs = probs[proj_inds]
                            if proj_probs.ndim < 2:
                                proj_ins_probs = np.expand_dims(proj_ins_probs, 0)
                                proj_probs = np.expand_dims(proj_probs, 0)

                            frame_probs_uint8_p = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                            frame_probs = frame_probs_uint8_p[proj_mask, :].astype(np.float32) / 255
                            frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                            frame_probs_uint8_p[proj_mask, :] = (frame_probs * 255).astype(np.uint8)

                            ins_preds = np.zeros((proj_mask.shape[0]))
                            ins_preds[proj_mask] = proj_ins_probs

                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    frame_probs_uint8_p = np.insert(frame_probs_uint8_p, l_ind, 0, axis=1)

                            # Predicted labels
                            frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_p,
                                                                                     axis=1)].astype(np.int32)

                            filename_i = '{:s}_{:07d}_{}_i.npy'.format(seq_name, f_ind - fi - 1, f_ind)
                            filepath_i = join(test_path, folder, filename_i)
                            #('Saving {}'.format(filepath_i))
                            np.save(filepath_i, ins_preds)

                            filename_p = '{:s}_{:07d}_{}.npy'.format(seq_name, f_ind-fi-1, f_ind)
                            filepath_p = join(test_path, folder, filename_p)
                            #print('Saving {}'.format(filepath_p))
                            np.save(filepath_p, frame_preds)

                            ins_features = {}
                            for ins_id in np.unique(ins_preds):
                                if int(ins_id) in self.instances[sequence]:
                                    ins_features[int(ins_id)] = self.instances[sequence][int(ins_id)]['mean']
                            filename_f = '{:s}_{:07d}_{}_f.npy'.format(seq_name, f_ind-fi-1, f_ind)
                            filepath_f = join(test_path, folder, filename_f)
                            #np.save(filepath_f, ins_features)

                    # Save some prediction in ply format for visual
                    if test_loader.dataset.set == 'validation':

                        # Insert false columns for ignored labels
                        frame_probs_uint8_bis = frame_probs_uint8.copy()
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8_bis = np.insert(frame_probs_uint8_bis, l_ind, 0, axis=1)

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_bis,
                                                                                 axis=1)].astype(np.int32)

                        if not majority_voting:
                            np.save(filepath, frame_preds) #!!!
                        #print('Saving {}'.format(filepath))
                        # Save some of the frame pots
                        if f_ind % 20 == 0:
                            seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                            velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds, pots],
                                          ['x', 'y', 'z', 'gt', 'pre', 'pots'])
                            else:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds],
                                          ['x', 'y', 'z', 'gt', 'pre'])

                            # Also Save lbl probabilities
                            probpath = join(test_path, folder, filename[:-4] + '_probs.ply')
                            lbl_names = [test_loader.dataset.label_to_names[l]
                                         for l in test_loader.dataset.label_values
                                         if l not in test_loader.dataset.ignored_labels]
                            write_ply(probpath,
                                      [frame_points[:, :3], frame_probs_uint8],
                                      ['x', 'y', 'z'] + lbl_names)

                        # keep frame preds in memory
                        all_f_preds[s_ind][f_ind] = frame_preds
                        all_f_labels[s_ind][f_ind] = frame_labels

                    else:

                        # Insert false columns for ignored labels
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8 = np.insert(frame_probs_uint8, l_ind, 0, axis=1)

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8,
                                                                                 axis=1)].astype(np.int32)
                        if not majority_voting:
                            np.save(filepath, frame_preds) #!!!
                        if f_inds[b_i, 1] % 100 == 0:
                            # Load points
                            seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                            velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds, pots],
                                          ['x', 'y', 'z', 'pre', 'pots'])
                            else:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds],
                                          ['x', 'y', 'z', 'pre'])

                    # Stack all prediction for this epoch
                    i0 += length

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                t_end = time.time()
                
                duration_store_data = t_end - t_start_store_data
                print('duration_store_data: ' + str(duration_store_data))

                duration_step = t_end - t_start_step
                print('duration_step: ' + str(duration_step))

                duration_inference = t_end - t_start_inference
                print('duration_inference: ' + str(duration_inference))

                t_start_step = time.time()

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f}) / pots {:d} => {:.1f}%'
                    min_pot = int(torch.floor(torch.min(test_loader.dataset.potentials)))
                    pot_num = torch.sum(test_loader.dataset.potentials > min_pot + 0.5).type(torch.int32).item()
                    current_num = pot_num + (i + 1 - config.validation_size) * config.val_batch_num
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2]),
                                         min_pot,
                                         100.0 * current_num / len(test_loader.dataset.potentials)))
                

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))

            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                if test_loader.dataset.set == 'validation' and last_min % 1 == 0:

                    #####################################
                    # Results on the whole validation set
                    #####################################

                    # Confusions for our subparts of validation set
                    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
                    for i, (preds, truth) in enumerate(zip(predictions, targets)):

                        # Confusions
                        Confs[i, :, :] = fast_confusion(truth, preds, test_loader.dataset.label_values).astype(np.int32)


                    # Show vote results
                    print('\nCompute confusion')

                    val_preds = []
                    val_labels = []
                    t1 = time.time()
                    for i, seq_frames in enumerate(test_loader.dataset.frames):
                        val_preds += [np.hstack(all_f_preds[i])]
                        val_labels += [np.hstack(all_f_labels[i])]
                    val_preds = np.hstack(val_preds)
                    val_labels = np.hstack(val_labels)
                    t2 = time.time()
                    C_tot = fast_confusion(val_labels, val_preds, test_loader.dataset.label_values)
                    t3 = time.time()
                    print(' Stacking time : {:.1f}s'.format(t2 - t1))
                    print('Confusion time : {:.1f}s'.format(t3 - t2))

                    s1 = '\n'
                    for cc in C_tot:
                        for c in cc:
                            s1 += '{:7.0f} '.format(c)
                        s1 += '\n'
                    if debug:
                        print(s1)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C_tot = np.delete(C_tot, l_ind, axis=0)
                            C_tot = np.delete(C_tot, l_ind, axis=1)

                    # Objects IoU
                    val_IoUs = IoU_from_confusions(C_tot)

                    # Compute IoUs
                    mIoU = np.mean(val_IoUs)
                    s2 = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in val_IoUs:
                        s2 += '{:5.2f} '.format(100 * IoU)
                    print(s2 + '\n')

                    # Save a report
                    report_file = join(report_path, 'report_{:04d}.txt'.format(int(np.floor(last_min))))
                    strg = 'Report of the confusion and metrics\n'
                    strg += '***********************************\n\n\n'
                    strg += 'Confusion matrix:\n\n'
                    strg += s1
                    strg += '\nIoU values:\n\n'
                    strg += s2
                    strg += '\n\n'
                    with open(report_file, 'w') as f:
                        f.write(strg)

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break
            #break

        return

    def panoptic_4d_test_mpa(self, net, test_loader, config, num_votes=100, debug=True):
        """
        Test method for slam segmentation models
        """

        t_start_inference = time.time()
        duration_relevant_summed = 0

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.5
        last_min = -0.5
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes
        nc_model = net.C
        # Test saving path
        test_path = None
        report_path = None

        if config.dataset_task == '4d_panoptic':

            #assoc_saving = [asc_type for idx, asc_type in enumerate(config.association_types) if config.association_weights[idx] > 0]
            #assoc_saving.append(str(config.n_test_frames))
            #assoc_saving = '_'.join(assoc_saving)
            assoc_saving = config.sampling
            config.assoc_saving = config.sampling+'_'+ config.decay_sampling
            if hasattr(config, 'stride'):
                config.assoc_saving = config.sampling + '_' + config.decay_sampling+ '_str' + str(config.stride) +'_'
            if hasattr(config, 'big_gpu') and config.big_gpu:
                config.assoc_saving = config.assoc_saving + 'bigpug_'


        if config.saving:
            test_path = join(config.test_path, config.saving_path.split('/')[-1]+ '_'+config.assoc_saving+str(config.n_test_frames) + '_' + config.chosen_chkp)
            if not exists(test_path):
                makedirs(test_path)
            report_path = join(test_path, 'reports')
            if not exists(report_path):
                makedirs(report_path)

        if test_loader.dataset.set == 'validation':
            for folder in ['val_predictions', 'val_probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))
        else:
            for folder in ['predictions', 'probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))

        # Init validation container
        all_f_preds = []
        all_f_labels = []
        if test_loader.dataset.set == 'validation':
            for i, seq_frames in enumerate(test_loader.dataset.frames):
                all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
                all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        test_epoch = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        processed = 0 #number of frames that processed

        # Start test loop
        while True:
            print('Initialize workers')
            t_start_step = time.time()
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                flag = True
                if config.n_test_frames > 1:
                    lengths = batch.lengths[0].cpu().numpy()
                    for b_i, length in enumerate(lengths):
                        f_inds = batch.frame_inds.cpu().numpy()
                        f_ind = f_inds[b_i, 1]
                        if f_ind % config.n_test_frames != config.n_test_frames-1:
                             flag = False

                if processed == test_loader.dataset.all_inds.shape[0]:
                    return
                #if not flag:
                #    continue
                #else:
                processed +=1

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                with torch.no_grad():
                    
                    nuscene = False
                    binary = False
                    if binary:
                        t_start_forward_pass = time.time()
                        proposal_binary_mask, fp_points_sem_classes, proposal_semantic_classes, proposal_aggregation_features, proposal_objectness_scores, proposal_positions, proposal_ids, point_semantic_classses_binary, point_votes, centers_output = net(batch)
                        outputs = torch.zeros(fp_points_sem_classes.shape[0], 19)
                        t_end_forward_pass = time.time()
                        duration_forward_pass = t_end_forward_pass - t_start_forward_pass
                        print('duration_forward_pass: ' + str(duration_forward_pass))
                    else:
                        t_start_forward_pass = time.time()
                        proposal_binary_mask, fp_points_sem_classes, proposal_semantic_classes, proposal_aggregation_features, proposal_objectness_scores, proposal_positions, proposal_ids, outputs, point_votes, centers_output = net(batch)
                        t_end_forward_pass = time.time()
                        duration_forward_pass = t_end_forward_pass - t_start_forward_pass
                        print('duration_forward_pass: ' + str(duration_forward_pass))

                        probs = softmax(outputs)
                        probs = probs.cpu().detach().numpy()

                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)
                        preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)]
                        preds = torch.from_numpy(preds)
                        preds.to(outputs.device)

                    proposal_semantic_classes = proposal_semantic_classes.squeeze()
                    proposal_semantic_classes = torch.transpose(proposal_semantic_classes, 0, 1)
                    proposal_semantic_classes = softmax(proposal_semantic_classes)
                    proposal_semantic_classes = proposal_semantic_classes.cpu().detach().numpy()

                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            proposal_semantic_classes = np.insert(proposal_semantic_classes, l_ind, 0, axis=1)
                    proposal_semantic_classes = proposal_semantic_classes[:, 0:9]
                    proposal_confidence_score = np.max(proposal_semantic_classes, axis=1)
                    proposal_semantic_classes = test_loader.dataset.label_values[np.argmax(proposal_semantic_classes, axis=1)]

                    fp_points_sem_classes = softmax(fp_points_sem_classes)
                    fp_points_sem_classes = fp_points_sem_classes.cpu().detach().numpy()

                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            fp_points_sem_classes = np.insert(fp_points_sem_classes, l_ind, 0, axis=1)
                    fp_points_sem_classes = test_loader.dataset.label_values[np.argmax(fp_points_sem_classes, axis=1)]
                    # map to background labels
                    if nuscene:
                        fp_points_sem_classes = fp_points_sem_classes + 10
                    else:
                        fp_points_sem_classes = fp_points_sem_classes + 8

                    sequence = test_loader.dataset.sequences[batch.frame_inds[0][0]]
                    pose = test_loader.dataset.poses[batch.frame_inds[0][0]][batch.frame_inds[0][1]]
                    if sequence not in self.instances:
                        self.instances[sequence] = {}

                    t_start_ins_pred = time.time()
                    
                    if binary:
                        ins_preds, ins_id, proposals, proposals_center, point_semantic_classses, refined_proposals_positions = net.ins_pred_in_time_mpa(batch.labels, proposal_binary_mask, fp_points_sem_classes, proposal_aggregation_features, proposal_semantic_classes, proposal_objectness_scores, proposal_positions, proposal_ids, batch.points[0], self.next_ins_id)
                    else:
                        ins_preds, ins_id, proposals, proposals_center, point_semantic_classses, refined_proposals_positions = net.ins_pred_in_time_mpa(batch.labels, proposal_binary_mask, fp_points_sem_classes, proposal_aggregation_features, proposal_semantic_classes, proposal_objectness_scores, proposal_positions, proposal_ids, preds, batch.points[0], self.next_ins_id)
                        #ins_preds, ins_id, proposals, proposals_center, point_semantic_classses, proposals_center_help = net.ins_pred_in_time_mpa_bb(batch.labels, fp_points_sem_classes, proposal_semantic_classes, proposal_objectness_scores, proposal_positions, proposal_ids, preds, batch.points[0], self.next_ins_id)
                        #ins_preds, ins_id, proposals, proposals_center, point_semantic_classses, refined_proposals_positions = net.ins_pred_in_time_nms(proposal_objectness_scores, proposal_confidence_score, proposal_positions, proposal_ids, preds, batch.points[0], self.next_ins_id)

                    t_end_ins_pred = time.time()
                    duration_ins_pred = t_end_ins_pred - t_start_ins_pred
                    print('duration_ins_pred: ' + str(duration_ins_pred))

                    duration_relevant = duration_forward_pass + duration_ins_pred
                    print('duration_relevant: ' + str(duration_relevant))

                    duration_relevant_summed = duration_relevant_summed + duration_relevant
                    print('duration_relevant_summed: ' + str(duration_relevant_summed))
                    
                    #help_labels = batch.labels.cpu().detach().numpy()
                    
                    preds = point_semantic_classses #!!!
                    majority_voting_or_use_sem_prop = True #!!!

                    self.next_ins_id = ins_id#update next available ins id
                    #for ins_id, instance in new_instances.items(): #add new instances to history
                    #    self.instances[sequence][ins_id] = instance

                    dont_track_ids = []
                    for ins_id in self.instances[sequence].keys():
                        if self.instances[sequence][ins_id]['life'] == 0:
                            dont_track_ids.append(ins_id)
                        self.instances[sequence][ins_id]['life'] -= 1

                    for ins_id in dont_track_ids:
                        del self.instances[sequence][ins_id]

                # Get probs and labels
                stk_probs = softmax(outputs).cpu().detach().numpy()
                #ins_preds = ins_preds.cpu().detach().numpy()
                centers_output = centers_output.cpu().detach().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                f_inds = batch.frame_inds.cpu().numpy()
                r_inds_list = batch.reproj_inds
                r_mask_list = batch.reproj_masks
                f_inc_r_inds_list = batch.f_inc_reproj_inds
                f_inc_r_mask_list = batch.f_inc_reproj_masks

                labels_list = batch.val_labels
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    probs = stk_probs[i0:i0 + length]
                    c_probs = centers_output[i0:i0 + length]
                    ins_probs = ins_preds[i0:i0 + length]
                    semantic_predictions_mv = preds[i0:i0 + length] #!!!
                    proj_inds = r_inds_list[b_i]
                    proj_mask = r_mask_list[b_i]
                    frame_labels = labels_list[b_i]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Add 4D visualization here
                    folder_4d = "4D_val_probs"
                    if not exists(join(test_path, folder_4d)):
                        makedirs(join(test_path, folder_4d))
                    seq_name_4d = test_loader.dataset.sequences[s_ind]
                    filename_4d = '{:s}_{:07d}.npy'.format(seq_name_4d, f_ind)
                    filename_i_4d = '{:s}_{:07d}_i.npy'.format(seq_name_4d, f_ind)
                    filename_c_4d = '{:s}_{:07d}_c.npy'.format(seq_name_4d, f_ind)
                    filename_p_4d = '{:s}_{:07d}_p.npy'.format(seq_name_4d, f_ind)
                    filename_cp_4d = '{:s}_{:07d}_cp.npy'.format(seq_name_4d, f_ind)
                    filename_v_4d = '{:s}_{:07d}_v.npy'.format(seq_name_4d, f_ind)
                    filename_gt_4d = '{:s}_{:07d}_gt.npy'.format(seq_name_4d, f_ind)
                    filename_gt_i_4d = '{:s}_{:07d}_gt_i.npy'.format(seq_name_4d, f_ind)
                    filename_gt_c_4d = '{:s}_{:07d}_gt_c.npy'.format(seq_name_4d, f_ind)
                    filename_prop_ids_4d = '{:s}_{:07d}_prop_ids.npy'.format(seq_name_4d, f_ind)
                    filename_prop_c_4d = '{:s}_{:07d}_prop_c.npy'.format(seq_name_4d, f_ind)
                    filename_prop_c_help_4d = '{:s}_{:07d}_prop_c_help.npy'.format(seq_name_4d, f_ind)
                    filename_prop_c_ref_4d = '{:s}_{:07d}_prop_c_ref.npy'.format(seq_name_4d, f_ind)
                    filename_gt_center_4d = '{:s}_{:07d}_gt_center.npy'.format(seq_name_4d, f_ind)
                    filepath_4d = join(test_path, folder_4d, filename_4d)
                    filepath_i_4d = join(test_path, folder_4d, filename_i_4d)
                    filepath_c_4d = join(test_path, folder_4d, filename_c_4d)
                    filepath_p_4d = join(test_path, folder_4d, filename_p_4d)
                    filepath_cp_4d = join(test_path, folder_4d, filename_cp_4d)
                    filepath_v_4d = join(test_path, folder_4d, filename_v_4d)
                    filepath_gt_4d = join(test_path, folder_4d, filename_gt_4d)
                    filepath_gt_i_4d = join(test_path, folder_4d, filename_gt_i_4d)
                    filepath_gt_c_4d = join(test_path, folder_4d, filename_gt_c_4d)
                    filepath_prop_ids_4d = join(test_path, folder_4d, filename_prop_ids_4d)
                    filepath_prop_c_4d = join(test_path, folder_4d, filename_prop_c_4d)
                    filepath_prop_c_help_4d = join(test_path, folder_4d, filename_prop_c_help_4d)
                    filepath_prop_c_ref_4d = join(test_path, folder_4d, filename_prop_c_ref_4d)
                    filepath_gt_center_4d = join(test_path, folder_4d, filename_gt_center_4d)

                    batch_labels = batch.labels.cpu().numpy()
                    #batch_ins_labels = batch.ins_labels.cpu().numpy()
                    batch_ins_labels_help = batch.ins_labels_help
                    #help = batch_ins_labels_help[batch_ins_labels_help != 0]
                    #help = np.unique(help)
                    #help = np.sort(help)

                    #np.save(filepath_p_4d, batch.points[0].cpu().numpy())
                    #np.save(filepath_c_4d, c_probs)
                    #np.save(filepath_i_4d, ins_probs)
                    #np.save(filepath_cp_4d, center_points)
                    #np.save(filepath_gt_4d, batch_labels)
                    #np.save(filepath_gt_i_4d, batch_ins_labels_help)
                    #np.save(filepath_gt_c_4d, batch.centers.cpu().numpy())

                    sem_probs_uint8 = np.zeros(probs.shape, dtype=np.uint8)
                    sem_probs = sem_probs_uint8.astype(np.float32) / 255
                    sem_probs = test_smooth * sem_probs + (1 - test_smooth) * probs
                    sem_probs_uint8 = (sem_probs * 255).astype(np.uint8)
                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            sem_probs_uint8 = np.insert(sem_probs_uint8, l_ind, 0, axis=1)
                    sem_preds = test_loader.dataset.label_values[np.argmax(sem_probs_uint8,
                                                                           axis=1)].astype(np.int32)
                    #if majority_voting_or_use_sem_prop:
                    #    np.save(filepath_4d, semantic_predictions_mv) #!!!
                    #else:
                    #    np.save(filepath_4d, sem_preds) #!!!

                    #np.save(filepath_prop_ids_4d, proposals)!!!
                    #np.save(filepath_prop_c_4d, proposals_center)
                    #np.save(filepath_prop_c_help_4d, proposals_center_help)
                    #np.save(filepath_prop_c_ref_4d, refined_proposals_positions)
                    #np.save(filepath_gt_center_4d, gt_center)

                    point_votes = point_votes.squeeze()
                    point_votes = point_votes.cpu().detach().numpy()
                    ins_ids = np.where((point_semantic_classses < 9) & (point_semantic_classses != 0))[0]
                    point_votes = point_votes[ins_ids]
                    #np.save(filepath_v_4d, point_votes)

                    # end added

                    # Project predictions on the frame points
                    proj_probs = probs[proj_inds]
                    proj_c_probs = c_probs[proj_inds]
                    proj_ins_probs = ins_probs[proj_inds]
                    proj_semantic_predictions_mv = semantic_predictions_mv[proj_inds] #!!!

                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_c_probs = np.expand_dims(proj_c_probs, 0)
                        proj_probs = np.expand_dims(proj_probs, 0)
                        proj_ins_probs = np.expand_dims(proj_ins_probs, 0)
                        proj_semantic_predictions_mv = np.expand_dims(proj_semantic_predictions_mv, 0)

                    # Save probs in a binary file (uint8 format for lighter weight)
                    seq_name = test_loader.dataset.sequences[s_ind]
                    if test_loader.dataset.set == 'validation':
                        folder = 'val_probs'
                        pred_folder = 'val_predictions'
                    else:
                        folder = 'probs'
                        pred_folder = 'predictions'
                    filename = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
                    filepath = join(test_path, folder, filename)
                    filename_i = '{:s}_{:07d}_i.npy'.format(seq_name, f_ind)
                    filename_c = '{:s}_{:07d}_c.npy'.format(seq_name, f_ind)
                    filepath_i = join(test_path, folder, filename_i)
                    filepath_c = join(test_path, folder, filename_c)

                    #if exists(filepath):
                    #    frame_probs_uint8 = np.load(filepath)
                    #    ins_preds = np.load(filepath_i)
                    #else:
                    frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                    frame_c_probs = np.zeros((proj_mask.shape[0], 1))
                    ins_preds = np.zeros((proj_mask.shape[0]))
                    frame_semantic_predictions = np.zeros((proj_mask.shape[0])) #!!!

                    frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
                    frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                    frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)
                    ins_preds[proj_mask] = proj_ins_probs
                    frame_c_probs[proj_mask] = proj_c_probs
                    frame_semantic_predictions[proj_mask] = proj_semantic_predictions_mv #!!!

                    #np.save(filepath, frame_probs_uint8)
                    #print ('Saving {}'.format(filepath_i))
                    np.save(filepath_i, ins_preds)
                    np.save(filepath_c, frame_c_probs)
                    if majority_voting_or_use_sem_prop:
                        np.save(filepath, frame_semantic_predictions) #!!!

                    ins_features = {}
                    for ins_id in np.unique(ins_preds):
                        if int(ins_id) in self.instances[sequence]:
                            ins_features[int(ins_id)] = self.instances[sequence][int(ins_id)]['mean']
                    filename_f = '{:s}_{:07d}_f.npy'.format(seq_name, f_ind)
                    filepath_f = join(test_path, folder, filename_f)
                    #np.save(filepath_f, ins_features)

                    if config.n_test_frames > 1:
                        for fi in range(len(f_inc_r_inds_list[b_i])):
                            proj_inds = f_inc_r_inds_list[b_i][fi]
                            proj_mask = f_inc_r_mask_list[b_i][fi]
                            proj_ins_probs = ins_probs[proj_inds]
                            proj_probs = probs[proj_inds]
                            if proj_probs.ndim < 2:
                                proj_ins_probs = np.expand_dims(proj_ins_probs, 0)
                                proj_probs = np.expand_dims(proj_probs, 0)

                            frame_probs_uint8_p = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                            frame_probs = frame_probs_uint8_p[proj_mask, :].astype(np.float32) / 255
                            frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                            frame_probs_uint8_p[proj_mask, :] = (frame_probs * 255).astype(np.uint8)

                            ins_preds = np.zeros((proj_mask.shape[0]))
                            ins_preds[proj_mask] = proj_ins_probs

                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    frame_probs_uint8_p = np.insert(frame_probs_uint8_p, l_ind, 0, axis=1)

                            # Predicted labels
                            frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_p,
                                                                                     axis=1)].astype(np.int32)

                            filename_i = '{:s}_{:07d}_{}_i.npy'.format(seq_name, f_ind - fi - 1, f_ind)
                            filepath_i = join(test_path, folder, filename_i)
                            #('Saving {}'.format(filepath_i))
                            np.save(filepath_i, ins_preds)

                            filename_p = '{:s}_{:07d}_{}.npy'.format(seq_name, f_ind-fi-1, f_ind)
                            filepath_p = join(test_path, folder, filename_p)
                            #print('Saving {}'.format(filepath_p))
                            np.save(filepath_p, frame_preds)

                            ins_features = {}
                            for ins_id in np.unique(ins_preds):
                                if int(ins_id) in self.instances[sequence]:
                                    ins_features[int(ins_id)] = self.instances[sequence][int(ins_id)]['mean']
                            filename_f = '{:s}_{:07d}_{}_f.npy'.format(seq_name, f_ind-fi-1, f_ind)
                            filepath_f = join(test_path, folder, filename_f)
                            #np.save(filepath_f, ins_features)

                    # Save some prediction in ply format for visual
                    if test_loader.dataset.set == 'validation':

                        # Insert false columns for ignored labels
                        frame_probs_uint8_bis = frame_probs_uint8.copy()
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8_bis = np.insert(frame_probs_uint8_bis, l_ind, 0, axis=1)

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_bis,
                                                                                 axis=1)].astype(np.int32)

                        if not majority_voting_or_use_sem_prop:
                            np.save(filepath, frame_preds) #!!!
                        #print('Saving {}'.format(filepath))
                        # Save some of the frame pots
                        if f_ind % 20 == 0:
                            seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                            if nuscene:
                                velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.npy')
                                frame_points = np.load(velo_file)                                
                            else:
                                velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                                frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds, pots],
                                          ['x', 'y', 'z', 'gt', 'pre', 'pots'])
                            else:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds],
                                          ['x', 'y', 'z', 'gt', 'pre'])

                            # Also Save lbl probabilities
                            probpath = join(test_path, folder, filename[:-4] + '_probs.ply')
                            lbl_names = [test_loader.dataset.label_to_names[l]
                                         for l in test_loader.dataset.label_values
                                         if l not in test_loader.dataset.ignored_labels]
                            write_ply(probpath,
                                      [frame_points[:, :3], frame_probs_uint8],
                                      ['x', 'y', 'z'] + lbl_names)

                        # keep frame preds in memory
                        all_f_preds[s_ind][f_ind] = frame_preds
                        all_f_labels[s_ind][f_ind] = frame_labels

                    else:

                        # Insert false columns for ignored labels
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8 = np.insert(frame_probs_uint8, l_ind, 0, axis=1)

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8,
                                                                                 axis=1)].astype(np.int32)
                        if not majority_voting_or_use_sem_prop:
                            np.save(filepath, frame_preds) #!!!
                        if f_inds[b_i, 1] % 100 == 0:
                            # Load points
                            seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                            if nuscene:
                                velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.npy')
                                frame_points = np.load(velo_file)
                            else:
                                velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                                frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds, pots],
                                          ['x', 'y', 'z', 'pre', 'pots'])
                            else:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds],
                                          ['x', 'y', 'z', 'pre'])

                    # Stack all prediction for this epoch
                    i0 += length

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                t_end = time.time()
                
                duration_step = t_end - t_start_step
                print('duration_step: ' + str(duration_step))

                duration_inference = t_end - t_start_inference
                print('duration_inference: ' + str(duration_inference))

                t_start_step = time.time()

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f}) / pots {:d} => {:.1f}%'
                    min_pot = int(torch.floor(torch.min(test_loader.dataset.potentials)))
                    pot_num = torch.sum(test_loader.dataset.potentials > min_pot + 0.5).type(torch.int32).item()
                    current_num = pot_num + (i + 1 - config.validation_size) * config.val_batch_num
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2]),
                                         min_pot,
                                         100.0 * current_num / len(test_loader.dataset.potentials)))
                

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))

            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                if test_loader.dataset.set == 'validation' and last_min % 1 == 0:

                    #####################################
                    # Results on the whole validation set
                    #####################################

                    # Confusions for our subparts of validation set
                    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
                    for i, (preds, truth) in enumerate(zip(predictions, targets)):

                        # Confusions
                        Confs[i, :, :] = fast_confusion(truth, preds, test_loader.dataset.label_values).astype(np.int32)


                    # Show vote results
                    print('\nCompute confusion')

                    val_preds = []
                    val_labels = []
                    t1 = time.time()
                    for i, seq_frames in enumerate(test_loader.dataset.frames):
                        val_preds += [np.hstack(all_f_preds[i])]
                        val_labels += [np.hstack(all_f_labels[i])]
                    val_preds = np.hstack(val_preds)
                    val_labels = np.hstack(val_labels)
                    t2 = time.time()
                    C_tot = fast_confusion(val_labels, val_preds, test_loader.dataset.label_values)
                    t3 = time.time()
                    print(' Stacking time : {:.1f}s'.format(t2 - t1))
                    print('Confusion time : {:.1f}s'.format(t3 - t2))

                    s1 = '\n'
                    for cc in C_tot:
                        for c in cc:
                            s1 += '{:7.0f} '.format(c)
                        s1 += '\n'
                    if debug:
                        print(s1)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C_tot = np.delete(C_tot, l_ind, axis=0)
                            C_tot = np.delete(C_tot, l_ind, axis=1)

                    # Objects IoU
                    val_IoUs = IoU_from_confusions(C_tot)

                    # Compute IoUs
                    mIoU = np.mean(val_IoUs)
                    s2 = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in val_IoUs:
                        s2 += '{:5.2f} '.format(100 * IoU)
                    print(s2 + '\n')

                    # Save a report
                    report_file = join(report_path, 'report_{:04d}.txt'.format(int(np.floor(last_min))))
                    strg = 'Report of the confusion and metrics\n'
                    strg += '***********************************\n\n\n'
                    strg += 'Confusion matrix:\n\n'
                    strg += s1
                    strg += '\nIoU values:\n\n'
                    strg += s2
                    strg += '\n\n'
                    with open(report_file, 'w') as f:
                        f.write(strg)

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break
            #break

        return
    