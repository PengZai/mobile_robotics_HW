

import numpy as np
import matplotlib.pyplot as plt

# parameter setting
N = 10000
mu_sensor = [10, 0]
sigma_sensor = [0.5, 0.25]

# generate point clouds
r, theta = np.zeros(10), np.zeros(10)
x, y = np.zeros(10), np.zeros(10)
#############################################################################
#                    TODO: Implement your code here                         #
#############################################################################
# i) Sensor (r, theta) frame 

r = sigma_sensor[0] * np.random.randn(N) + mu_sensor[0]
theta = sigma_sensor[1] * np.random.randn(N) + mu_sensor[1]

# ii) Cartesian (x,y) coordinate frame


x = r*np.cos(theta)
y = r*np.sin(theta)


#############################################################################
#                            END OF YOUR CODE                               #
#############################################################################
# i) Observation in the sensor frame
plt.plot(r, theta, '.', markersize=2)
plt.axis('equal')
plt.grid(True)
plt.title('Sensor Frame Point Cloud')
plt.xlabel('Range (m)')
plt.ylabel('Bearing (rad)')
plt.show()

# ii) Observation in the Cartesian frame
plt.plot(x, y, '.', markersize=2)
plt.axis('equal')
plt.grid(True)
plt.title('Cartesian Frame Point Cloud')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()