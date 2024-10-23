
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi


class UKF:
    # UKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance
        
        self.kappa_g = init.kappa_g
        
        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)

    
    def prediction(self, u):
        # prior belief
        X = self.state_.getState()
        P = self.state_.getCovariance()

        ###############################################################################
        # TODO: Implement the prediction step for UKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################
        
        # propagate the input Gaussian using an unscented transform
        self.sigma_point(X, P, self.kappa_g)
        X_sigma_points = []
        X_sigma_point_mean = 0
        for i in range(2*self.n + 1):
            X_sigma_point = self.gfun(self.X[:, i], u)
            X_sigma_points.append(X_sigma_point)
            X_sigma_point_mean += self.w[i] * X_sigma_point

        X_sigma_points = (np.array(X_sigma_points).reshape([2*self.n+1, -1])).T
        temp = X_sigma_points - X_sigma_point_mean.reshape(self.n, 1)
        X_Cov = np.dot(np.dot(temp, np.diag(self.w)), temp.T)

        X_pred = X_sigma_point_mean
        P_pred = X_Cov

   
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################


        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)

    def correction(self, z, landmarks):

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
        Q = np.zeros((self.Q.shape[0]*2, self.Q.shape[1]*2))
        Q[:2, :2] = self.Q
        Q[2:, 2:] = self.Q


        self.sigma_point(X_predict, P_predict, self.kappa_g)
        Z_hats = []
        Z_hat_mean = 0
        for i in range(2*self.n + 1):
            Z1_hat = self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], self.X[:, i])
            Z2_hat = self.hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], self.X[:, i])
            Z_hat = np.hstack((Z1_hat, Z2_hat))
            Z_hats.append(Z_hat)
            Z_hat_mean += self.w[i] * Z_hat

        Z_hats = (np.array(Z_hats).reshape([2*self.n+1, -1])).T
        temp = Z_hats - np.expand_dims(Z_hat_mean, axis=1)

        # innovation covariance
        S = np.dot(np.dot(temp, np.diag(self.w)), temp.T) + Q

        # compute state-measurement cross covariance
        Cov_xz = np.dot(np.dot(self.X - np.expand_dims(X_predict, axis=1), np.diag(self.w)), (Z_hats - np.expand_dims(Z_hat_mean, axis=1)).T)

        # filter gain
        K = np.dot(Cov_xz, np.linalg.inv(S))  
        
        # correct the predicted state statistics
        X = X_predict + K @ (z - Z_hat_mean)
        P = P_predict - K @ S @ K.T

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)

    def sigma_point(self, mean, cov, kappa):
        
        self.n = len(mean) # dim of state
        mean = mean.reshape((self.n, 1))
        L = np.sqrt(self.n + kappa) * np.linalg.cholesky(cov)
        Y = mean.repeat(len(mean), axis=-1)
        self.X = np.hstack((mean, Y+L, Y-L))
        self.w = np.zeros([2 * self.n + 1, 1])
        self.w[0] = kappa / (self.n + kappa)
        self.w[1:] = 1 / (2 * (self.n + kappa))
        self.w = self.w.reshape(-1)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state