#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
#https://github.com/PRBonn/semantic-kitti-ap
import argparse
import os
import yaml
import sys
import numpy as np
import time
import json

from eval_np import Panoptic4DEval


# possible splits
splits = ["train", "valid", "test"]

# added by me
# take second element for sort
def takeSecond(elem):
    return elem[1]
# end added

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./evaluate_panoptic.py")
  parser.add_argument(
      '--dataset',
      '-d',
      type=str,
      #required=True,
      default='/globalwork/kreuzberg/SemanticKITTI/dataset',
      help='Dataset dir. No Default',
  )
  parser.add_argument(
      '--predictions',
      '-p',
      type=str,
      #required=True,
      default='/globalwork/kreuzberg/4D-PLS/test/Log_2022-06-17_12-16-59_importance_None_str1_bigpug_4_current_chkp/stitch4', 
      help='Prediction dir. Same organization as dataset, but predictions in'
      'each sequences "prediction" directory. No Default. If no option is set'
      ' we look for the labels in the same directory as dataset'
  )
  parser.add_argument(
      '--split',
      '-s',
      type=str,
      required=False,
      choices=["train", "valid", "test"],
      default="valid",
      help='Split to evaluate on. One of ' + str(splits) + '. Defaults to %(default)s',
  )
  parser.add_argument(
      '--data_cfg',
      '-dc',
      type=str,
      #required=True,
      default='/globalwork/kreuzberg/SemanticKITTI/dataset/semantic-kitti.yaml',
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--limit',
      '-l',
      type=int,
      required=False,
      default=None,
      help='Limit to the first "--limit" points of each scan. Useful for'
      ' evaluating single scan from aggregated pointcloud.'
      ' Defaults to %(default)s',
  )
  parser.add_argument(
      '--min_inst_points',
      type=int,
      required=False,
      default=50,
      help='Lower bound for the number of points to be considered instance',
  )
  parser.add_argument(
      '--output',
      type=str,
      required=False,
      default=None,
      help='Output directory for scores.txt and detailed_results.html.',
  )

  start_time = time.time()

  FLAGS, unparsed = parser.parse_known_args()



  # fill in real predictions dir
  if FLAGS.predictions is None:
    FLAGS.predictions = FLAGS.dataset

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Data: ", FLAGS.dataset)
  print("Predictions: ", FLAGS.predictions)
  print("Split: ", FLAGS.split)
  print("Config: ", FLAGS.data_cfg)
  print("Limit: ", FLAGS.limit)
  print("Min instance points: ", FLAGS.min_inst_points)
  print("Output directory", FLAGS.output)
  print("*" * 80)

  # assert split
  assert (FLAGS.split in splits)

  # open data config file
  DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))

  # get number of interest classes, and the label mappings
  # class
  class_remap = DATA["learning_map"]
  class_inv_remap = DATA["learning_map_inv"]
  class_ignore = DATA["learning_ignore"]
  nr_classes = len(class_inv_remap)
  class_strings = DATA["labels"]

  # make lookup table for mapping
  # class
  maxkey = max(class_remap.keys())

  # +100 hack making lut bigger just in case there are unknown labels
  class_lut = np.zeros((maxkey + 100), dtype=np.int32)
  class_lut[list(class_remap.keys())] = list(class_remap.values())

  # class
  ignore_class = [cl for cl, ignored in class_ignore.items() if ignored]

  print("Ignoring classes: ", ignore_class)

  # get test set
  test_sequences = DATA["split"][FLAGS.split]

  # create evaluator
  class_evaluator = Panoptic4DEval(nr_classes, None, ignore_class, offset = 2 ** 32, min_points=FLAGS.min_inst_points)

  # get label paths
  label_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    label_paths = os.path.join(FLAGS.dataset, "sequences", sequence, "labels")
    # populate the label names
    seq_label_names = sorted([os.path.join(label_paths, fn) for fn in os.listdir(label_paths) if fn.endswith(".label")])
    label_names.extend(seq_label_names)
  # print(label_names)

  # get predictions paths
  pred_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    pred_paths = os.path.join(FLAGS.predictions, "sequences", sequence, "predictions")
    # populate the label names
    seq_pred_names = sorted([os.path.join(pred_paths, fn) for fn in os.listdir(pred_paths) if fn.endswith(".label")])
    pred_names.extend(seq_pred_names)
  # print(pred_names)

  # check that I have the same number of files
  assert (len(label_names) == len(pred_names))
  print("Evaluating sequences: ", end="", flush=True)
  # open each file, get the tensor, and make the iou comparison

  complete = len(label_names)

  count = 0
  percent = 10

  # added by me
  false_positive_list = []
  false_negative_list = []
  false_positive_count_summed = 0
  false_negative_count_summed = 0
  false_negative_clust_count_summed = 0
  wrong_centers = []
  wrong_centers_classes = [0, 0, 0, 0, 0, 0, 0, 0, 0]
  # end added

  for label_file, pred_file in zip(label_names, pred_names):
    count = count + 1
    if 100 * count / complete > percent:
      print("{}% ".format(percent), end="", flush=True)
      percent = percent + 10
    # print("evaluating label ", label_file, "with", pred_file)
    # open label

    label = np.fromfile(label_file, dtype=np.uint32)

    u_label_sem_class = class_lut[label & 0xFFFF]  # remap to xentropy format
    u_label_inst = label >> 16
    if FLAGS.limit is not None:
      u_label_sem_class = u_label_sem_class[:FLAGS.limit]
      u_label_inst = u_label_inst[:FLAGS.limit]

    label = np.fromfile(pred_file, dtype=np.uint32)

    u_pred_sem_class = class_lut[label & 0xFFFF]  # remap to xentropy format
    u_pred_inst = label >> 16
    if FLAGS.limit is not None:
      u_pred_sem_class = u_pred_sem_class[:FLAGS.limit]
      u_pred_inst = u_pred_inst[:FLAGS.limit]

    # added by me

    # compute false positve and false negative
    u_pred_inst_help = u_pred_inst.copy()
    u_pred_inst_help[u_pred_sem_class == 0] = 0
    u_pred_inst_help[u_pred_sem_class >= 9] = 0
    false_positive_inds = np.where((u_pred_inst_help > 0) & (u_label_inst == 0))
    false_negative_inds = np.where((u_pred_inst_help == 0) & (u_label_inst > 0)) #these FN can result from wrong foreground/backround predictions and from wrong clustering
    false_negative_inds_clust = np.where((u_pred_inst_help == 0) & (u_label_inst > 0) & (u_pred_sem_class < 9)) #these FN only result from wrong clustering
    false_positive_count = false_positive_inds[0].shape[0]
    false_negative_count = false_negative_inds[0].shape[0]
    false_negative_clust_count = false_negative_inds_clust[0].shape[0]
    false_positive_list.append((count-1, false_positive_count))
    false_negative_list.append((count-1, false_negative_count))
    false_positive_count_summed += false_positive_count
    false_negative_count_summed += false_negative_count
    false_negative_clust_count_summed += false_negative_clust_count

    # find partially predicted instances
    instances_gt = np.unique(u_label_inst)
    instances_pred = np.unique(u_pred_inst_help)
    for instance in np.nditer(instances_gt):
      if instance == 0:
        continue
      instance_ids = np.where(u_label_inst == instance)[0]
      sem_classes = u_label_sem_class[instance_ids]
      if np.unique(u_label_sem_class[instance_ids]).shape[0] > 1:
        continue
      help = u_pred_inst_help[instance_ids]
      corresponding_pred_instances = np.unique(u_pred_inst_help[instance_ids])
      zero = np.array([0])
      corresponding_pred_instances = np.setdiff1d(corresponding_pred_instances,zero)
      if corresponding_pred_instances.shape[0] > 1:
        #print(label_file + " Instance: " + str(instance) + ", divided into " + str(corresponding_pred_instances.shape[0]))
        dict = {}
        for i in np.nditer(corresponding_pred_instances):
          ids = np.where((u_pred_inst_help == i) & (u_label_inst == instance))[0]
          n = ids.shape[0]
          dict[int(i)] = n
        do_print = True
        for key in dict:
          if dict[key] < 500:
            do_print = False
        if do_print:
          print(label_file + " Instance: " + str(instance) + " (" + str(instance_ids.shape[0]) + ") " + ", divided into " + str(corresponding_pred_instances.shape[0]) + ", " + str(dict))

    # find merged predicted instances      
    for instance in np.nditer(instances_pred):
      if instance == 0:
        continue
      instance_ids = np.where(u_pred_inst_help == instance)[0]
      corresponding_gt_instances = np.unique(u_label_inst[instance_ids])
      zero = np.array([0])
      corresponding_gt_instances = np.setdiff1d(corresponding_gt_instances,zero)
      if corresponding_gt_instances.shape[0] > 1:
        #print(label_file + " Instance: " + str(instance) + ", divided into " + str(corresponding_pred_instances.shape[0]))
        dict = {}
        for i in np.nditer(corresponding_gt_instances):
          ids = np.where((u_label_inst == i) & (u_pred_inst_help == instance))[0]
          n = ids.shape[0]
          dict[int(i)] = n
        do_print = True
        for key in dict:
          if dict[key] < 500:
            do_print = False
        if do_print:
          print(label_file + " Instances: " + str(dict) + ", merged into " + str(instance) + " (" + str(instance_ids.shape[0]) + " )")

    # end added

    class_evaluator.addBatch(label_file.split('/')[-3], u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)

  print("100%")
  print('FP: ' + str(false_positive_count_summed))
  print('FN: ' + str(false_negative_count_summed))

  # added by me
  """
  false_positive_list_sorted = false_positive_list.copy()
  false_negative_list_sorted = false_negative_list.copy()
  false_positive_list_sorted.sort(key=takeSecond, reverse=True)
  false_negative_list_sorted.sort(key=takeSecond, reverse=True)

  f1 = open("wrong_centers.txt", "w")
  for element in wrong_centers:
      f1.write(str(element) + "\n")
  f1.close()
  f2 = open("false_positive.txt", "w")
  for element in false_positive_list_sorted:
      f2.write(str(element) + "\n")
  f2.close()
  f3 = open("false_negative.txt", "w")
  for element in false_negative_list_sorted:
      f3.write(str(element) + "\n")
  f3.close()
  """
  # end added

  complete_time = time.time() - start_time
  LSTQ, LAQ_ovr, LAQ, AQ_p, AQ_r,  iou, iou_mean, iou_p, iou_r = class_evaluator.getPQ4D()
  things_iou = iou[1:9].mean()
  stuff_iou = iou[9:].mean()
  print ("=== Results ===")
  print ("LSTQ:", LSTQ)
  print("S_assoc (LAQ):", LAQ_ovr)
  float_formatter = "{:.2f}".format
  np.set_printoptions(formatter={'float_kind': float_formatter})
  print ("Assoc:", LAQ)
  print ("iou:", iou)
  print("things_iou:", things_iou)
  print("stuff_iou:", stuff_iou)

  print ("S_cls (LSQ):", iou_mean)

