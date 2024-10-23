import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

class EKF:

    def __init__(self, system, init):
        # EKF Construct an instance of this class
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.Gfun = init.Gfun  # Jocabian of motion model
        self.Vfun = init.Vfun  # Jocabian of motion model
        self.Hfun = init.Hfun  # Jocabian of measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance

        self.state_ = RobotState()

        # init state
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)


    ## Do prediction and set state in RobotState()
    def prediction(self, u):

        # prior belief
        X = self.state_.getState()
        P = self.state_.getCovariance()

        ###############################################################################
        # TODO: Implement the prediction step for EKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################
        
        X_pred = self.gfun(X, u)
        G = self.Gfun(X, u)
        P_pred = G @ P @ G.T + self.M(u)
        

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)


    def correction(self, z, landmarks):
        # EKF correction step
        #
        # Inputs:
        #   z:  measurement
        X_predict = self.state_.getState()
        P_predict = self.state_.getCovariance()
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for EKF                                 #
        # Hint: save your corrected state and cov as X and P                          #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################

        z = np.hstack((z[:2], z[3:5]))
        z1_hat = self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], X_predict)
        z2_hat = self.hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], X_predict)
        z_hat = np.hstack((z1_hat, z2_hat))

        H1 = self.Hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], X_predict, z1_hat)
        H2 = self.Hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], X_predict, z2_hat)
        H = np.vstack((H1, H2))
        Q = np.zeros((self.Q.shape[0]*2, self.Q.shape[1]*2))
        Q[:2, :2] = self.Q
        Q[2:, 2:] = self.Q
        K = P_predict @ H.T @ np.linalg.inv(H @ P_predict @ H.T + Q)
        X = X_predict + K @ (z - z_hat)
        P = (np.eye(P_predict.shape[0]) - K @ H) @ P_predict
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)


    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state