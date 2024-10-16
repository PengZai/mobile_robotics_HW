import numpy as np
import pandas as pd
import copy
import scipy
import random
import cv2
from numpy.random import randn, rand


if __name__ == "__main__":

    # none information prior
    mu = np.expand_dims(np.random.rand(3), axis= 1)
    copy_mu = copy.deepcopy(mu)
    Sigma = np.array([
        [1000, 0, 0],
        [0, 1000, 0],
        [0, 0, 1000]
    ])
    copy_Sigma = copy.deepcopy(Sigma)

    Kf_1 = pd.read_csv("./data_csv/Kf_1.csv", index_col=False, header=None).to_numpy()
    Kf_2 = pd.read_csv("./data_csv/Kf_2.csv", index_col=False, header=None).to_numpy()

    c_1 = pd.read_csv("./data_csv/C_1.csv", index_col=False, header=None).to_numpy()
    c_2 = pd.read_csv("./data_csv/C_2.csv", index_col=False, header=None).to_numpy()

    R = pd.read_csv("./data_csv/R.csv", index_col=False, header=None).to_numpy()
    t = pd.read_csv("./data_csv/t.csv", index_col=False, header=None).to_numpy()

    z_1_set = pd.read_csv("./data_csv/z_1.csv", index_col=False, header=None).to_numpy()
    z_2_set = pd.read_csv("./data_csv/z_2.csv", index_col=False, header=None).to_numpy()

    z_set = np.hstack((copy.deepcopy(z_1_set), copy.deepcopy(z_2_set)))


    mean_z_1 = np.mean(z_1_set, axis=0)
    mean_z_2 = np.mean(z_2_set, axis=0)
    Sigma_v1 = np.cov(z_1_set.T)
    Sigma_v2 = np.cov(z_2_set.T)
    Sigma_v12 = np.cov(z_set.T)

    # projection matrix
    K1 = np.eye(3)
    K1[:2,:2] = Kf_1
    K1[:2, 2] = c_1[:, 0]
    PM1 = K1 @ np.hstack((np.eye(3), np.zeros(shape = (3,1))))

    K2 = np.eye(3)
    K2[:2,:2] = Kf_2
    K2[:2, 2] = c_2[:, 0]
    PM2 = K2 @ np.hstack((R.T, -R.T@t))


    # triangulation point according to measurement of pixels
    init_p = cv2.triangulatePoints(PM1, PM2, mean_z_1, mean_z_2)
    init_p = init_p/init_p[3]
    init_p = init_p[:3]

    M = 100
    init_p_Sigma = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    samples = np.random.multivariate_normal(init_p[:,0], init_p_Sigma, size = M)
    copy_samples = copy.deepcopy(samples)
    Sigma_w = np.array([
        [0.01, 0, 0],
        [0, 0.01, 0],
        [0, 0, 0.01]
    ])


    def h1(p):
        return Kf_1 @ np.array([[p[0][0]/p[2][0]], [p[1][0]/p[2][0]]]) + c_1

    def h2(p):
        
        p = R.T @ p - R.T @ t
        return Kf_2 @ (np.array([[p[0][0]/p[2][0]], [p[1][0]/p[2][0]]])) + c_2 
    

    def h12(p):

        return np.vstack((h1(p), h2(p)))


    def get_Jacobian_H1(p):
        
        return Kf_1 @ np.array([
                [1/p[2][0], 0, -1* p[0][0]/(p[2][0]**2)],
                [0, 1/p[2][0], -1* p[1][0]/(p[2][0]**2)] 
            ])
    
    def get_Jacobian_H2(p):
        
        q = R.T @ p - R.T @ t
        return Kf_2 @ np.array([
                [1/q[2][0], 0, -1* q[0][0]/(q[2][0]**2)],
                [0, 1/q[2][0], -1* q[1][0]/(q[2][0]**2)] 
            ]) @ R.T
    
    def get_Jacobian_H12(p):

        return np.vstack((get_Jacobian_H1(p), get_Jacobian_H2(p)))

    def EFK(mu, Sigma, z_set, Sigma_z, h, Jacobian_H):
        for z in z_set:
            z = np.expand_dims(z, axis=1)
            Sigma = Sigma + Sigma_w
            H = Jacobian_H(mu)
            K = Sigma @ H.T @ np.linalg.inv(H @ Sigma @ H.T + Sigma_z)
            
            mu = mu + K @ (z - h(mu))
           
            Sigma = (np.eye(3) -  K @ H) @ Sigma

        return mu, Sigma

    def Particle_filter(samples, z_set, Sigma_z, h):

        def resampling(P, W):
            # low variance resampling
            
            resample_P = []

            M = len(W)
            cumW = np.cumsum(W)
            r = rand(1) / M
            j = 1
            for i in range(M):
                u = r + (i - 1) / M
                while u > cumW[j]:
                    j = j + 1
                resample_P.append(P[j])
                W[i] = 1 / M

            return resample_P, W
            
        M = len(samples)
        W = np.ones(M)/M
        
        temp_P = []
        P = []
        for z in z_set:
            observation_probs = np.zeros(M)
            P.clear()
            for i in range(M):
                p = np.random.multivariate_normal(samples[i], Sigma_w, 1).T
                observation_probs[i] = scipy.stats.multivariate_normal(mean = h(p)[:, 0], cov = Sigma_z).pdf(z)
                P.append(p)
            W = np.multiply(observation_probs, W)
            W = W/np.sum(W)
            Neff = 1 / np.sum(np.power(W, 2))
            
            if Neff < M / 5:
                P, W = resampling(P, W)

                
            
            resamples = random.choices(population = P, weights=W, k=M)
            samples = np.array(resamples)[:, :,0]
                
        return samples
    

   

    mu, Sigma = EFK(mu, Sigma, z_1_set, Sigma_v1, h1, get_Jacobian_H1)
    print('EKF with z1, mu=(%f, %f, %f) ' % (mu[0][0], mu[1][0], mu[2][0]))
    mu, Sigma = EFK(mu, Sigma, z_2_set, Sigma_v2, h2, get_Jacobian_H2)
    print('EKF with z2, mu=(%f, %f, %f) ' % (mu[0][0], mu[1][0], mu[2][0]))

    copy_mu, copy_Sigma = EFK(copy_mu, copy_Sigma, z_set, Sigma_v12, h12, get_Jacobian_H12)
    print('EKF with z12, mu=(%f, %f, %f) ' % (copy_mu[0][0], copy_mu[1][0], copy_mu[2][0]))

    samples = Particle_filter(samples, z_1_set, Sigma_v1, h1)
    mu = np.mean(samples, axis = 0)
    print('Particle_filter with z1, mu=(%f, %f, %f) ' % (mu[0], mu[1], mu[2]))
    samples = Particle_filter(samples, z_2_set, Sigma_v2, h2)
    mu = np.mean(samples, axis = 0)
    print('Particle_filter with z2, mu=(%f, %f, %f) ' % (mu[0], mu[1], mu[2]))
    copy_samples = Particle_filter(copy_samples, z_set, Sigma_v12, h12)
    mu = np.mean(copy_samples, axis = 0)
    print('Particle_filter with z12, mu=(%f, %f, %f) ' % (mu[0], mu[1], mu[2]))


    print('end')