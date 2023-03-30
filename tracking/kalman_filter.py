from filterpy.kalman import KalmanFilter
import numpy as np


class KalmanBoxTracker:

    count = 0
    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
    #define constant velocity model
    # define a dimension 7 state vector and a dimension 4 input vector
    # [xc, yc, scale, aspect_ratio, xc', yc', scale']
    # Input is [xc, yc, scale, aspect_ratio]
    # F: State Transistion Matrix
    # H: Measurement Matrix
    # P: Current State Covariance Matrix
    # Q: Process Noise Matrix
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],  
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10. # High variance for more spread out of noise (deviations)
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = KalmanBoxTracker.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
            [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
            the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    #scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(bbox, score=None):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
            [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(bbox[2] * bbox[3])
        h = bbox[2] / w
        if(score==None):
            return np.array([bbox[0]-w/2.,bbox[1]-h/2.,bbox[0]+w/2.,bbox[1]+h/2.]).reshape((1,4))
        else:
            return np.array([bbox[0]-w/2.,bbox[1]-h/2.,bbox[0]+w/2.,bbox[1]+h/2.,score]).reshape((1,5))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # if velocity of scale and scale is both smaller than zero
        if((self.kf.x[6]+self.kf.x[2])<=0):
            # consider scale velocity to be zero
            self.kf.x[6] *= 0.0
        # could be SIMDable
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(KalmanBoxTracker.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        # NSA Kalman Filter arXiv:2202.11983
        self.update_R_SG(bbox[-1])

        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(KalmanBoxTracker.convert_bbox_to_z(bbox))

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return KalmanBoxTracker.convert_x_to_bbox(self.kf.x)

    def update_R_NSA(self, c: float):
        self.kf.R = np.eye(self.kf.dim_z) * (1 - c)

    def update_R_SG(self, c: float, sigma: float = 3.):
        # https://doi.org/10.1007/s40747-022-00946-9
        # We precompute 1/sqrt(2*pi)
        self.kf.R = np.eye(self.kf.dim_z) * 0.3989422804 / sigma * np.exp(-0.5 * (c - 1) ** 2)

class KalmanBoxTracker_OH(object):
    """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
    count = 0

    def __init__(self, bbox, init_mode, bbox_before):
        """
    Initialises a tracker using initial bounding box.
    """
        # Unmatched Detections is init_mode = 0
        # define constant velocity model
        # (u, v, s, r, u_dot, v_dot, s_dot) -> (u,v): location center, s: area, r: aspect ratio
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        if init_mode == 0:
            self.kf.R[2:, 2:] *= 1.   # 10.
            self.kf.P[4:, 4:] *= 10.  # 1000. # give high uncertainty to the unobservable initial velocities
            self.kf.P *= 10.
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[4:, 4:] *= 0.01
            self.kf.x[:4] = KalmanBoxTracker_OH.convert_bbox_to_z(bbox)

        elif init_mode == 1:
            self.kf.R[2:, 2:] *= 1.
            self.kf.P[4:, 4:] *= 10.  # give high uncertainty to the unobservable initial velocities
            # self.kf.P *= 10.
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[4:, 4:] *= 0.01
            state_before = KalmanBoxTracker_OH.convert_bbox_to_z(bbox_before)
            state = KalmanBoxTracker_OH.convert_bbox_to_z(bbox)
            self.kf.x[:4] = state
            self.kf.x[4:] = state[0:3] - state_before[0:3]

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.age = 0
        # self.oc_number = 0  # Number of time an object is occluded
        self.time_since_observed = 0    # The period that an object is detected as occluded
        self.confidence = 0.5

    def update(self, bbox, isz):
        """
    Updates the state vector with observed bbox.
    """
        self.update_R_SG(bbox[-1])
        self.time_since_update = 0
        if isz == 0:
            # decrease area change ratio
            self.kf.x[6] /= 2
            self.kf.update(None)
        elif isz == 1:
            self.kf.update(KalmanBoxTracker_OH.convert_bbox_to_z(bbox))
            self.time_since_observed = 0

    def predict(self):
        """
    Advances the state vector and returns the predicted bounding box estimate.
    """
        # to prevent area become negative after prediction, make zero the rate of area change
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return KalmanBoxTracker_OH.convert_x_to_bbox(self.kf.x)

    def get_state(self):
        """
    Returns the current bounding box estimate.
    """
        return KalmanBoxTracker_OH.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
            [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
            the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    #scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(bbox, score=None):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
            [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(bbox[2] * bbox[3])
        h = bbox[2] / w
        if(score==None):
            return np.array([bbox[0]-w/2.,bbox[1]-h/2.,bbox[0]+w/2.,bbox[1]+h/2.]).reshape((1,4))
        else:
            return np.array([bbox[0]-w/2.,bbox[1]-h/2.,bbox[0]+w/2.,bbox[1]+h/2.,score]).reshape((1,5))

    def update_R_SG(self, c: float, sigma: float = 3.):
        # https://doi.org/10.1007/s40747-022-00946-9
        # We precompute 1/sqrt(2*pi)
        self.kf.R = np.eye(self.kf.dim_z) * 0.3989422804 / sigma * np.exp(-0.5 * (c - 1) ** 2)