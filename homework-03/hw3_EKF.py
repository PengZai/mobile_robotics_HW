import numpy as np
import pandas as pd
import copy
import scipy
import random
import cv2
from numpy.random import randn, rand
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # none information prior
    # mu = np.expand_dims(np.random.rand(3) + 0.5, axis= 1)
    init_mu = np.expand_dims(np.array([1.5,1.5,1]), axis=1)
    copy_init_mu = copy.deepcopy(init_mu)
    Sigma = np.array([
        [100, 0, 0],
        [0, 100, 0],
        [0, 0, 100]
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

    # particle number
    M = 100
    init_p_Sigma = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    samples = np.random.multivariate_normal(init_p[:,0], init_p_Sigma, size = M)
    copy_samples = copy.deepcopy(samples)
    Sigma_w = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
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
        
        trajectories = [mu[:, 0]]
        
        for z in z_set:
            z = np.expand_dims(z, axis=1)
            Sigma = Sigma + Sigma_w
            H = Jacobian_H(mu)
            K = Sigma @ H.T @ np.linalg.inv(H @ Sigma @ H.T + Sigma_z)
            
            mu = mu + K @ (z - h(mu))
            Sigma = (np.eye(3) -  K @ H) @ Sigma
            
            trajectories.append(mu[:, 0])

        
        trajectories = np.array(trajectories)
        
        return mu, Sigma, trajectories

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
        trajectories = []
        trajectories.append([np.sum(samples[:, 0]*W), np.sum(samples[:, 1]*W), np.sum(samples[:, 2]*W)])
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

            trajectory = np.array(P)[:, :, 0]
            trajectories.append([np.sum(trajectory[:, 0]*W), np.sum(trajectory[:, 1]*W), np.sum(trajectory[:, 2]*W)])
            samples = np.array(P)[:, :,0]
            
        
        trajectories = np.array(trajectories)
        
        return samples, trajectories
    
    
    def visualization(p, save_name):
        
        
        N = len(p) 
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        t = np.arange(N)
        
        fig = plt.figure()
        
        line1, = plt.plot(t, x, color='blue')
        line2, = plt.plot(t, y, color='red')
        line3, = plt.plot(t, z, color='green')
        plt.legend([line1, line2, line3], [r'x', r'y', r'z'], loc='best')
        plt.grid(True)
        plt.axis('equal')
        plt.title(save_name)
        plt.xlim([0, N])
        plt.ylim([-3, 3])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$altitude$')
        plt.savefig(save_name+".png")
        # plt.show()
    

   

    mu, Sigma, trajectories_with_z1 = EFK(init_mu, Sigma, z_1_set, Sigma_v1, h1, get_Jacobian_H1)
    print('EKF with z1, mu=(%f, %f, %f) ' % (mu[0][0], mu[1][0], mu[2][0]))
    mu, Sigma, trajectories_with_z2 = EFK(mu, Sigma, z_2_set, Sigma_v2, h2, get_Jacobian_H2)
    print('EKF with z2, mu=(%f, %f, %f) ' % (mu[0][0], mu[1][0], mu[2][0]))
    visualization(np.vstack((trajectories_with_z1, trajectories_with_z2)), "EKF with z1 followed by z2 and init mu = %.2f, %.2f, %.2f" %(init_mu[0], init_mu[1], init_mu[2]))
    
    copy_mu, copy_Sigma, trajectories = EFK(copy_init_mu, copy_Sigma, z_set, Sigma_v12, h12, get_Jacobian_H12)
    print('EKF with z12, mu=(%f, %f, %f) ' % (copy_mu[0][0], copy_mu[1][0], copy_mu[2][0]))
    visualization(trajectories, "EKF with z1 stacking with z2 and init mu = %.2f, %.2f, %.2f" % (copy_init_mu[0], copy_init_mu[1], copy_init_mu[2]))

    # samples, trajectories_with_z1 = Particle_filter(samples, z_1_set, Sigma_v1, h1)
    # print('Particle_filter with z1, mu=(%f, %f, %f) ' % (trajectories[-1][0], trajectories[-1][1], trajectories[-1][2]))
    # samples, trajectories_with_z2 = Particle_filter(samples, z_2_set, Sigma_v2, h2)
    # print('Particle_filter with z2, mu=(%f, %f, %f) ' % (trajectories[-1][0], trajectories[-1][1], trajectories[-1][2]))
    # visualization(np.vstack((trajectories_with_z1, trajectories_with_z2)), "PF with z1 followed by z2")    
    
    copy_samples, trajectories = Particle_filter(copy_samples, z_set, Sigma_v12, h12)
    print('Particle_filter with z12, mu=(%f, %f, %f) ' % (trajectories[-1][0], trajectories[-1][1], trajectories[-1][2]))
    visualization(trajectories, "PF with z1 stacking with z2")


    print('end')