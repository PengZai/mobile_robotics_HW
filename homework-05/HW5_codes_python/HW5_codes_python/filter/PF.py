
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

from scipy.stats import multivariate_normal
from numpy.random import default_rng
rng = default_rng()

class PF:
    # PF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):
        np.random.seed(2)
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance
        
        # PF parameters
        self.n = init.n
        self.Sigma = init.Sigma
        self.particles = init.particles
        self.particle_weight = init.particle_weight
        self.Neff = 1/np.sum(np.power(self.particle_weight, 2))
        
        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)

    
    def prediction(self, u):
        ###############################################################################
        # TODO: Implement the prediction step for PF, remove pass                     #
        # Hint: Propagate your particles. Particles are saved in self.particles       #
        # Hint: Use rng.standard_normal instead of np.random.randn.                   #
        #       It is statistically more random.                                      #
        ###############################################################################
        
        # sample noise
        LM = np.linalg.cholesky(self.M(u)) # cholesky on a diagonal matrix just sqrt its diagonal elements
        particles = np.zeros(self.particles.T.shape)
        for i in range(self.n):
            wi = np.dot(LM, rng.standard_normal(3)) # noise sample
            particles[i] = self.gfun(self.particles.T[i], u) + wi

        self.particles = particles.T
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################


    def correction(self, z, landmarks):
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))
        
        ###############################################################################
        # TODO: Implement the correction step for PF                                  #
        # Hint: self.mean_variance() will update the mean and covariance              #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################

        z = np.hstack((z[:2], z[3:5]))
        w = np.zeros(self.n) # important weight
        Q = np.zeros((self.Q.shape[0]*2, self.Q.shape[1]*2))
        Q[:2, :2] = self.Q
        Q[2:, 2:] = self.Q

        for i in range(self.n):

            z1_hat = self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], self.particles.T[i])
            z2_hat = self.hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], self.particles.T[i])
            z_hat = np.hstack((z1_hat, z2_hat))
            w[i] = multivariate_normal.pdf(z, z_hat, Q)

        
        self.particle_weight = np.multiply(self.particle_weight, w)
        self.particle_weight = self.particle_weight / np.sum(self.particle_weight)
        self.Neff = 1/np.sum(np.power(self.particle_weight, 2))

        if self.Neff < self.n / 5:
            self.resample()
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.mean_variance()


    def resample(self):
        new_samples = np.zeros_like(self.particles)
        new_weight = np.zeros_like(self.particle_weight)
        W = np.cumsum(self.particle_weight)
        r = np.random.rand(1) / self.n
        count = 0
        for j in range(self.n):
            u = r + j/self.n
            while u > W[count]:
                count += 1
            new_samples[:,j] = self.particles[:,count]
            new_weight[j] = 1 / self.n
        self.particles = new_samples
        self.particle_weight = new_weight
    

    def mean_variance(self):
        X = np.mean(self.particles, axis=1)
        sinSum = 0
        cosSum = 0
        for s in range(self.n):
            cosSum += np.cos(self.particles[2,s])
            sinSum += np.sin(self.particles[2,s])
        X[2] = np.arctan2(sinSum, cosSum)
        zero_mean = np.zeros_like(self.particles)
        for s in range(self.n):
            zero_mean[:,s] = self.particles[:,s] - X
            zero_mean[2,s] = wrap2Pi(zero_mean[2,s])
        P = zero_mean @ zero_mean.T / self.n
        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)
    
    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state

