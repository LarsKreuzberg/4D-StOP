import numpy as np
import pyviz3d.visualizer as viz

def create_color_palette_semantics_old():
    palette = [(255, 255, 255),     #0 
               (174, 199, 232),     #1
               (152, 223, 138),     #2
               (31, 119, 180),      #3
               (255, 187, 120),     #4
               (188, 189, 34),      #5
               (140, 86, 75),       #6
               (255, 152, 150),     #7
               (214, 39, 40),       #8
               (197, 176, 213),     #9
               (148, 103, 189),     #10
               (196, 156, 148),     #11
               (23, 190, 207),      #12
               (178, 76, 76),       #13
               (247, 182, 210),     #14
               (66, 188, 102),      #15
               (219, 219, 141),     #16
               (140, 57, 197),      #17
               (202, 185, 52),      #18
               (51, 176, 203)]      #19

    palette = np.asarray(palette)
    return palette

def create_color_palette_instances_old():
    palette = [(255, 255, 255),
               (174, 199, 232),
               (152, 223, 138),
               (31, 119, 180),
               (255, 187, 120),
               (188, 189, 34),
               (140, 86, 75),
               (255, 152, 150),
               (214, 39, 40),
               (197, 176, 213),
               (148, 103, 189),
               (196, 156, 148),
               (23, 190, 207),
               (178, 76, 76),
               (247, 182, 210),
               (66, 188, 102),
               (219, 219, 141),
               (140, 57, 197),
               (202, 185, 52),
               (51, 176, 203),
               (200, 54, 131),
               (92, 193, 61),
               (78, 71, 183),
               (172, 114, 82),
               (255, 127, 14),
               (91, 163, 138),
               (153, 98, 156),
               (140, 153, 101),
               (158, 218, 229),
               (100, 125, 154),
               (178, 127, 135),
               (120, 185, 128),
               (146, 111, 194),
               (44, 160, 44),
               (112, 128, 144),
               (96, 207, 209),
               (227, 119, 194),
               (213, 92, 176),
               (94, 106, 211),
               (82, 84, 163),
               (100, 85, 144)]

    palette = np.asarray(palette)
    return palette

def create_color_palette_help():
    palette = [(255, 255, 255),         # white
               (0, 0, 0),               # black - votes/proposals-center
               (255, 0, 0),             # red - FP
               (0, 0, 255),             # blue - FN
               (100, 100, 100),         # grey
               (211, 211, 211),         # light-grey
               (255,165,0)]             # orange
               #(224,255,255)]           # light cyan

    palette = np.asarray(palette)
    return palette

def create_color_palette_instances():
    palette = [(255, 255, 255),         # white
               (255,0,255),             # magenta
               (240,128,128),           # light coral
               (0,250,154),             # medium spring green       
               (0,255,0),               # lime                     
               (127,255,212),           # aqua marine            
               (0,255,255),             # cyan                  
               #(255,255,0),             # yellow 
               (152,251,152),           # pale green                   
               (138,43,226),            # blue violet  
               (186,85,211),            # medium orchid             
               #(152,251,152),           # pale green   
               (255,165,0),             # orange
               (175,238,238),           # pale turquoise           
               (154,205,50),            # yellow green
               #(255,165,0),             # orange
               (255,255,0),             # yellow 
               (32,178,170),            # light sea green
               (70,130,180),            # steel blue
               #(0,191,255),             # deep sky blue
               (135,206,250),           # light sky blue
               (173,216,230),           # light blue
               (255,99,71),             # tomato
               (30,144,255),            # dodger blue
               #(135,206,235),           # sky blue
               (0,191,255),             # deep sky blue
               (218,112,214),           # orchid
               (255,20,147),            # deep pink
               (100,149,237),           # corn flower blue
               (255,105,180),           # hot pink
               (255,192,203),           # pink
               (255,127,80),            # coral
               (0,206,209),             # dark turquoise
               (233,150,122),           # dark salmon
               (23, 190, 207),
               (178, 76, 76),
               (247, 182, 210),
               (66, 188, 102),
               (219, 219, 141),
               (140, 57, 197),
               (202, 185, 52),
               (51, 176, 203),
               (200, 54, 131),
               (92, 193, 61),
               (78, 71, 183),
               (172, 114, 82),
               (255, 127, 14),
               (91, 163, 138),
               (153, 98, 156),
               (140, 153, 101),
               (158, 218, 229),
               (100, 125, 154),
               (178, 127, 135),
               (120, 185, 128),
               (146, 111, 194),
               (44, 160, 44),
               (112, 128, 144),
               (96, 207, 209),
               (227, 119, 194),
               (213, 92, 176),
               (94, 106, 211),
               (82, 84, 163),
               (100, 85, 144), ###
               (174, 199, 232),
               (152, 223, 138),
               (31, 119, 180),
               (255, 187, 120),
               (188, 189, 34),
               (140, 86, 75),
               (255, 152, 150),
               (214, 39, 40),
               (197, 176, 213),
               (148, 103, 189),
               (196, 156, 148),
               (255,215,0),
               (240,230,140),
               (245,222,179),
               (0,128,128),
               (173,255,47),
               (106,90,205),
               (102,205,170),
               (221,160,221),
               (219,112,147),
               (127,255,0),
               (148,0,211),
               ]

    palette = np.asarray(palette)
    return palette

def create_color_palette_semantics():
    palette = [(255, 255, 255),         # white
               (255,0,255),             # magenta                   #1      # car
               (0,255,0),               # lime                      #2      # bicycle
               (127,255,212),           # aqua marine               #3      # motorcycle
               (0,255,255),             # cyan                      #4      # truck
               (255,255,0),             # yellow                    #5      # other-vehicle
               (138,43,226),            # blue violet               #6      # person
               (0,250,154),             # medium spring green       #7      # bicyclist
               (152,251,152),           # pale green                #8      # motorcyclist
               (255,215,0),             # gold                      #9      # road
               (240,230,140),           # khaki                     #10     # parking
               (245,222,179),           # wheat                     #11     # sidewalk
               (0,128,128),             # teal                      #12     # other-ground
               (173,255,47),            # green yellow              #13     # building
               (106,90,205),            # slate blue                #14     # fence
               (102,205,170),           # medium aqua marine        #15     # vegetation
               (221,160,221),           # plum                      #16     # trunk
               (219,112,147),           # pale violet red           #17     # terrain
               (127,255,0),             # chartreuse                #18     # pole
               (148,0,211),             # dark violet               #19     # traffic-sign
               ]

    palette = np.asarray(palette)
    return palette

def create_color_palette_instances_help():
    palette = [(255, 255, 255),         # white
               (238,130,238),           # violet
               (135,206,250),           # light sky blue
               (135,206,250),           # light sky blue
               (135,206,250),           # light sky blue
               (135,206,250),           # light sky blue
               (135,206,250),           # light sky blue
               (255,165,0),             # orange 
               ]

    palette = np.asarray(palette)
    return palette

def main():

    color_palette_semantics = create_color_palette_semantics()
    color_palette_instances = create_color_palette_instances()
    color_palette_help = create_color_palette_help()

    v = viz.Visualizer()

    approach = '4D-StOP' 

    frame_number = '0000588'
    seq_number = '08'
    log_folder = 'Log_2022-06-13_17-33-24_importance_None_str1_bigpug_2_current_chkp'

    path = '/globalwork/kreuzberg/4D-PLS/test/' + log_folder + '/4D_val_probs/' + seq_number + '_' + frame_number

    panoptic_visualization = False
    semantic_visualization = False
    instance_visualizations = True
    instance_one_color_visualizations = False
    raw_data = True

    use_votes = False
    use_prop_centers = True
    use_ref_prop_centers = True
    use_center_points = False
    use_lines = False

    stuff_things_boundary = 8#10

    point_size = 35.0

    point_positions = np.load(path + '_p.npy')
    mean = np.mean(point_positions, axis=0) 
    point_positions = point_positions - mean

    # raw-data
    if raw_data:
        point_raw_data_colors = np.ones(point_positions.shape[0], int)
        point_raw_data_colors = point_raw_data_colors + 4
        point_raw_data_colors = color_palette_help[point_raw_data_colors]
        v.add_points('RawData', point_positions, point_raw_data_colors, point_size=point_size) 


    # semantics
    if panoptic_visualization or semantic_visualization:
        point_semantics_gt = np.load(path + '_gt.npy')
        point_semantics_pred = np.load(path + '.npy')
        sem_ids_gt = np.where(point_semantics_gt != 0)[0]
        sem_ids_pred = np.where(point_semantics_pred != 0)[0]
        #if semantic_visualization:
        #    sem_ids_gt = np.where(point_semantics_gt != 0)[0]
        #    sem_ids_pred = np.where(point_semantics_pred != 0)[0]
        #elif panoptic_visualization:
        #    sem_ids_gt = np.where(point_semantics_gt > stuff_things_boundary)[0]
        #    sem_ids_pred = np.where(point_semantics_pred > stuff_things_boundary)[0]
        point_semantics_gt_colors = color_palette_semantics[point_semantics_gt[sem_ids_gt].astype(int)]
        point_semantics_pred_colors = color_palette_semantics[point_semantics_pred[sem_ids_pred].astype(int)]
        v.add_points('SemGT', point_positions[sem_ids_gt], point_semantics_gt_colors, point_size=point_size)
        v.add_points('SemPred', point_positions[sem_ids_pred], point_semantics_pred_colors, point_size=point_size)
    
    # instances
    if instance_one_color_visualizations:
        point_ins_pred = np.load(path + '_i.npy')
        ins_ids_pred = np.where(point_ins_pred != 0)[0]
        ins_colors = np.ones(ins_ids_pred.shape[0], int)
        ins_colors = ins_colors + 5
        ins_colors = color_palette_help[ins_colors]
        v.add_points('Ins1', point_positions[ins_ids_pred], ins_colors, point_size=point_size)

    
    # instances 
    if panoptic_visualization or instance_visualizations:
        # gt
        point_ins_gt = np.load(path + '_gt_i.npy')
        point_ins_gt_help = np.zeros(point_ins_gt.shape)
        point_ins_gt = np.where(point_ins_gt > 0, point_ins_gt, np.inf)
        min = np.amin(point_ins_gt)
        counter = 1#5
        while (min != np.inf):
            point_ins_gt_help = np.where(point_ins_gt == min, counter, point_ins_gt_help)
            point_ins_gt = np.where(point_ins_gt > min, point_ins_gt, np.inf)
            min = np.amin(point_ins_gt)
            counter += 1
        point_instances_gt_colors = color_palette_instances[point_ins_gt_help[point_ins_gt_help != 0].astype(int)]
        v.add_points('InsGT', point_positions[point_ins_gt_help != 0], point_instances_gt_colors, point_size=point_size)

        #pred
        point_ins_pred = np.load(path + '_i.npy')
        point_ins_pred_help = np.zeros(point_ins_pred.shape)
        point_ins_pred = np.where(point_ins_pred > 0, point_ins_pred, np.inf)
        min = np.amin(point_ins_pred)
        counter = 1#5
        while (min != np.inf):
            point_ins_pred_help = np.where(point_ins_pred == min, counter, point_ins_pred_help)
            point_ins_pred = np.where(point_ins_pred > min, point_ins_pred, np.inf)
            min = np.amin(point_ins_pred)
            counter += 1
        point_instances_pred_colors = color_palette_instances[point_ins_pred_help[point_ins_pred_help != 0].astype(int)]
        v.add_points('InsPred', point_positions[point_ins_pred_help != 0], point_instances_pred_colors, point_size=point_size)


    if use_votes:
        point_votes = np.load(path + '_v.npy')
        point_votes = point_votes - mean
        point_votes_colors = np.ones(point_votes.shape[0], int)
        point_votes_colors = color_palette_help[point_votes_colors]
        v.add_points('Votes', point_votes, point_votes_colors, point_size=point_size) 

    if use_prop_centers:
        point_proposals_center = np.load(path + '_prop_c.npy')
        point_proposals_center = point_proposals_center - mean
        point_proposals_center_colors = np.ones(point_proposals_center.shape[0], int)
        point_proposals_center_colors = color_palette_help[point_proposals_center_colors]
        v.add_points('ProposalsCenter', point_proposals_center, point_proposals_center_colors, point_size=100)#150) 

    if use_ref_prop_centers:
        point_proposals_center_ref = np.load(path + '_prop_c_ref.npy')
        point_proposals_center_ref = point_proposals_center_ref - mean
        point_proposals_center_ref_colors = np.ones(point_proposals_center_ref.shape[0], int)
        point_proposals_center_ref_colors = point_proposals_center_ref_colors + 3
        point_proposals_center_ref_colors = color_palette_help[point_proposals_center_ref_colors]
        v.add_points('RefProposalsCenter', point_proposals_center_ref, point_proposals_center_ref_colors, point_size=100)#150) 

    if panoptic_visualization or instance_visualizations:
    #if False:
        false_points = np.zeros(point_ins_pred.shape, int)
        false_positive_inds = np.where((point_ins_pred_help > 0) & (point_ins_gt_help == 0))
        #false_negative_inds = np.where((point_ins_pred_help == 0) & (point_ins_gt_help > 0))
        false_points[false_positive_inds] = 2
        #false_points[false_negative_inds] = 3
        false_points_colors = color_palette_help[false_points]
        v.add_points('False', point_positions[false_points != 0], false_points_colors[false_points != 0], point_size=point_size)

    if use_center_points:
        center_points = np.load(path + '_cp.npy')
        #center_points_ids = np.where(center_points == 1)[0]
        #center_points_colors = np.ones(center_points_ids.shape[0], int)
        #center_points_colors = color_palette_help[center_points_colors]
        #v.add_points('CenterPoints', point_positions[center_points_ids], center_points_colors, point_size=100) 
        center_points = center_points.squeeze()
        center_points = center_points - mean
        center_points_colors = np.ones(center_points.shape[0], int)
        center_points_colors = color_palette_help[center_points_colors]
        v.add_points('CenterPoints', center_points, center_points_colors, point_size=100) 

    if use_lines:
        #instance_ids = np.unique(point_ins_gt_help)
        instance_ids = np.unique(point_ins_pred_help)
        for id in instance_ids:
            if id == 0:
                continue
            idx = np.where(point_ins_pred_help == id)
            instance_points = point_positions[idx]
            x_min = np.min(instance_points[:, 0])
            x_max = np.max(instance_points[:, 0])
            y_min = np.min(instance_points[:, 1])
            y_max = np.max(instance_points[:, 1])
            z_min = np.min(instance_points[:, 2])
            z_max = np.max(instance_points[:, 2])

            bb_center_x = (x_min + x_max) / 2
            bb_center_y = (y_min + y_max) / 2
            bb_center_z = (z_min + z_max) / 2
            bb_center = np.asarray([bb_center_x, bb_center_y, bb_center_z])
            bb_centers = np.zeros(instance_points.shape)
            bb_centers = bb_centers + bb_center

            size = instance_points.shape[0]
            percent = 0.3
            chosen_ids = np.random.randint(low=0, high=size, size=int(size * percent))

            v.add_lines('lines' + str(id), instance_points[chosen_ids], bb_centers[chosen_ids])
    

    # When we added everything we need to the visualizer, we save it.
    if panoptic_visualization:
        name = 'visualizations/vis_4d_' + seq_number + '_' + frame_number + '_pan_' + approach
    elif semantic_visualization:
        name = 'visualizations/vis_4d_' + seq_number + '_' + frame_number + '_sem_' + approach
    elif instance_visualizations:
        name = 'visualizations/vis_4d_' + seq_number + '_' + frame_number + '_ins_' + approach
    elif instance_one_color_visualizations:
        name = 'visualizations/vis_4d_' + seq_number + '_' + frame_number + '_ins1_' + approach
    elif raw_data:
        name = 'visualizations/vis_4d_' + seq_number + '_' + frame_number + '_raw_' + approach
    
    v.save(name)


if __name__ == '__main__':
    main()