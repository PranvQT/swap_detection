import numpy as np


def reset_swap_flags(self):

    print("Swap flags reset")
    self.swap_dict = {
        "flag_inswap": [],
        "flag_normal": [],
        "temp_fr": [],
        "cent_cor_x": [],
        "cent_cor_y": [],
        "past_dist": [],
        "pot_swap": [],
        "green_id": [],
        "red_id": [],
        "oc_frame": [],
        "in_pitch_box": []
    }
    self.rem_list = []


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


# Not used anywhere
# def iou_batch_cupy(bb_test, bb_gt):
#     """
#     From SORT: Computes IOU between two boxes in the form [x1,y1,x2,y2]
#     """
#     bb_gt = np.expand_dims(bb_gt, 0)
#     bb_gt = cp.array(bb_gt)
#     bb_test = np.expand_dims(bb_test, 1)
#     bb_gt = cp.array(bb_test)
#     # bb_test = cp.expand_dims(bb_test, 1)

#     xx1 = cp.maximum(bb_test[...,0], bb_gt[..., 0])
#     yy1 = cp.maximum(bb_test[..., 1], bb_gt[..., 1])
#     xx2 = cp.minimum(bb_test[..., 2], bb_gt[..., 2])
#     yy2 = cp.minimum(bb_test[..., 3], bb_gt[..., 3])
#     w = cp.maximum(0., xx2 - xx1)
#     h = cp.maximum(0., yy2 - yy1)
#     wh = w * h
#     o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
#     + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
#     return(o)
