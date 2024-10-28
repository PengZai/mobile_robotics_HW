import numpy as np
import matplotlib.pyplot as plt

def calculateEllipseXY(mu, Sigma, k2, N=20):
    """
    input:
    mu     is the [2x1] [x;y] mean vector
    Sigma  is the [2x2] covariance matrix
    k2     is the Chi-squared 2 DOF variable
    N      is the number of points to use (optional, default 20)
    
    output:
    x and y coordinates to draw the ellipse
    """
    # set up angles for drawing the ellipse
    angles = np.linspace(0, 2*np.pi, num=N)
    _circle = np.array([np.cos(angles), np.sin(angles)])

    # make sure it is a numpy array
    mu = np.array(mu)
    Sigma = np.array(Sigma)
        
    # cholesky decomposition
    L = np.linalg.cholesky(Sigma) # Cholesky factor of covariance
    
    # apply the transformation and scale of the covariance
    ellipse = np.sqrt(k2) * L @ _circle

    # shift origin to the mean
    x = mu[0] + ellipse[0, :].T
    y = mu[1] + ellipse[1, :].T

    return x, y

def draw_ellipse(mu, Sigma, k2, colorin='red'):
    """   
    input:
    mu       is the [2x1] [x;y] mean vector
    Sigma    is the [2x2] covariance matrix
    k2       is the Chi-squared 2 DOF variable
    Npoints  number of points to draw the ellipse (default 20)
    colorin  color for plotting ellipses, red for analytical contours, blue for sample contours
    
    --- h = draw_ellipse(mu, Sigma, k2)
    Draws an ellipse centered at mu with covariance Sigma and confidence region k2, i.e.,
    K2 = 1; # 1-sigma
    K2 = 4; # 2-sigma
    K2 = 9; # 3-sigma
    K2 = chi2inv(.50, 2); # 50% probability contour
    """
    Npoints = 20
    
    x, y = calculateEllipseXY(mu, Sigma, k2, Npoints)
    
    if k2 == 9:
        if colorin == 'red':
            plt.plot(x, y, linewidth=1.25, color=colorin, label='analytical contours')
        elif colorin == 'blue':
            plt.plot(x, y, linewidth=1.25, color=colorin, label='sample contours')
    else:
        plt.plot(x, y, linewidth=1.25, color=colorin)




# Sensor frame
plt.plot(r, theta, '.', markersize=2, label='data')
plt.axis('equal')
plt.grid(True)
plt.title('Sensor Frame Point Cloud')
plt.xlabel('Range (m)')
plt.ylabel('Bearing (rad)')
plt.legend()

# compute the mean and covariance of samples in the sensor frame 
mu_sensor = np.zeros((2,))
cov_sensor = np.zeros((2,2))
mu_sensor_sample = np.zeros((2,))
cov_sensor_sample = np.zeros((2,2))
#############################################################################
#                    TODO: Implement your code here                         #
#############################################################################


#############################################################################
#                            END OF YOUR CODE                               #
#############################################################################

# plot the ellipses
for i in range(3):
    # analytical contous in red
    draw_ellipse(mu_sensor, cov_sensor, (i+1)**2, 'red')
    # sample contous in blue
    draw_ellipse(mu_sensor_sample, cov_sensor_sample, (i+1)**2, 'blue')

plt.legend()
plt.show()

# Cartesian frame
plt.plot(x, y, '.', markersize=2, label='data')
plt.axis('equal')
plt.grid(True)
plt.title('Cartesian Frame Point Cloud')
plt.xlabel('x (m)')
plt.ylabel('y (m)')

# compute the mean and covariance of samples in the Cartesian frame
mu_cartesian = np.zeros((2,))
cov_cartesian = np.zeros((2,2))
mu_cartesian_sample = np.zeros((2,))
cov_cartesian_sample = np.zeros((2,2))
#############################################################################
#                    TODO: Implement your code here                         #
#############################################################################


#############################################################################
#                            END OF YOUR CODE                               #
#############################################################################

# plot the ellipses
for i in range(3):
    # analytical contous in red
    draw_ellipse(mu_cartesian, cov_cartesian, (i+1)**2, 'red')    
    # sample contous in blue
    draw_ellipse(mu_cartesian_sample, cov_cartesian_sample, (i+1)**2, 'blue')

plt.legend()
plt.show()