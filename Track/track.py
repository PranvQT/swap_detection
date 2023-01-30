import numpy as np
from filterpy.kalman import KalmanFilter
from Utils.sort_track_utils import convert_x_to_bbox


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, count_id):
        """
        Initialize a tracker using initial bounding box

        Parameters: 'bbox' must have 'detected class' int number at the -1 position

        Returns: None, as this initialises the class
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.5
        self.kf.Q[4:, 4:] *= 0.5

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)  # STATE VECTOR
        self.time_since_update = 0
        self.id = count_id
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # keep yolov5 detected class information
        self.detclass = bbox[5]

        self.player_type = 3
        self.highlight = 0
        self.highlight_streak = 0
        self.direction = -1
        self.tracklets = []
        self.close_ids = []
        self.is_merged = False
        self.temp_close_ids = []

    def convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and
        returns z in the form [x,y,s,r] where x,y is the center
        of the box and s is the scale/area and r is the aspect ratio

        Parameters: bbox: bounding box

        Returns: Converted bbox according to the shape
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h  # scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def update(self, bbox):
        """
        Updates the state vector with observed bbox

        Parameters: bbox: bounding box

        Returns: None, as the class variables are updated
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate and updates the class variables

        Parameters: None

        Returns: None
        """
        if (self.kf.x[6]+self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate and updates the class variables

        Parameters: None

        Returns: None
        """
        arr_detclass = np.expand_dims(np.array([self.detclass]), 0)
        arr_u_dot = np.expand_dims(self.kf.x[4], 0)
        arr_v_dot = np.expand_dims(self.kf.x[5], 0)
        arr_s_dot = np.expand_dims(self.kf.x[6], 0)
        return np.concatenate((convert_x_to_bbox(self.kf.x), arr_detclass, arr_u_dot, arr_v_dot, arr_s_dot), axis=1), self.player_type, self.highlight
