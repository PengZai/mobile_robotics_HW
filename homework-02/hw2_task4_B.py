import numpy as np
import matplotlib.pyplot as plt


mu_sensor = [10, 0]
sigma_sensor = [0.5, 0.25]

Jacobian = np.zeros((2,2))
cov_cartesian = np.zeros((2,2))
#############################################################################
#                    TODO: Implement your code here                         #
#############################################################################
# Implement the Jacobians

Jacobian = np.array([[np.cos(mu_sensor[1]), -1*mu_sensor[0]*np.sin(mu_sensor[1])], 
                     [np.sin(mu_sensor[1]), mu_sensor[0]*np.cos(mu_sensor[1])]]
                     )

# Implement the linearized covariance in cartesian corridinates
cov = np.array([[sigma_sensor[0]**2, 0],
               [0, sigma_sensor[1]**2]]
               )

cov_cartesian = Jacobian * cov * Jacobian.T

#############################################################################
#                            END OF YOUR CODE                               #
#############################################################################
print('Jacobian:\n', Jacobian)
print('\nSigma_cartesian:\n', cov_cartesian)