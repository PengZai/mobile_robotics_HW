from numpy.linalg import inv

# data from sensor I
z1 = [10.6715, 8.7925, 10.7172, 11.6302, 10.4889, 11.0347, 10.7269, 9.6966, 10.2939, 9.2127];

# data from sensor II
z2 = [10.7107, 9.0823, 9.1449, 9.3524, 10.2602]

# noise variance of sensor I
sigma_z1 = 1

# noise variance of sensor II
sigma_z2 = 0.64

# non-informative prior
mu_1 = 0
sigma_1 = 1000
mu_2 = 0
sigma_2 = 1000

# Kalman Filter Measurement Update
def KF_update(mu, sigma, sigma_Q, z):
#############################################################################
#                    TODO: Implement your code here                         #
#############################################################################

# this system 
# xt = xt-1
# zt = xt + Qt

# so our kalman filter become
# \bar{\mu}_t = \mu_{t-1},
# \bar{\Sigma}_t = \Sigma_{t-1}
# K_t = \Sigma_{t-1}*(\Sigma_{t-1} + Q_t)^(-1)
# \mu_{t} = \mu_{t-1} + K_t * (z_t - \mu_{t-1})
# \Sigma_t = (1-K_t)*\Sigma_{t-1}

  for sample_z in z:
    K = sigma * 1/(sigma + sigma_Q)
    mu = mu + K * (sample_z - mu)
    sigma = (1-K) * sigma


  return mu, sigma

#############################################################################
#                            END OF YOUR CODE                               #
#############################################################################

# recursive inference with data from sensor I and sensor II
#############################################################################
#                    TODO: Implement your code here                         #
#############################################################################
# run inferece using z1
mu_1, sigma_1 = KF_update(mu_1, sigma_1, sigma_z1, z1)
mu_2, sigma_2 = KF_update(mu_2, sigma_2, sigma_z2, z2)

# run inferece using z2


#############################################################################
#                            END OF YOUR CODE                               #
#############################################################################

print("sensor   mean  variance  precision(1/variance)")
print("  I%11.3f%8.3f%10.3f" % (mu_1, sigma_1, 1/sigma_1))
print(" II%11.3f%8.3f%10.3f" % (mu_2, sigma_2, 1/sigma_2))