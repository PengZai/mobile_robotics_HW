import numpy as np
import matplotlib.pyplot as plt

# colors
green = np.array([0.2980, 0.6, 0])
darkblue = np.array([0, 0.2, 0.4])
VermillionRed = np.array([156, 31, 46]) / 255

def plot_fuction(prior_belief, prediction, posterior_belief):
    """
    plot prior belief, prediction after action, and posterior belief after measurement
    """    
    fig = plt.figure()
    
    # plot prior belief
    ax1 = plt.subplot(311)
    plt.bar(np.arange(0,10), prior_belief.reshape(-1), color=darkblue)
    plt.title(r'Prior Belief')
    plt.ylim(0, 1)
    plt.ylabel(r'$bel(x_{t-1})$')

    # plot likelihood
    ax2 = plt.subplot(312)
    plt.bar(np.arange(0,10), prediction.reshape(-1), color=green)
    plt.title(r'Prediction After Action')
    plt.ylim(0, 1)
    plt.ylabel(r'$\overline{bel(x_t})}$')

    # plot posterior belief
    ax3 = plt.subplot(313)
    plt.bar(np.arange(0,10), posterior_belief.reshape(-1), color=VermillionRed)
    plt.title(r'Posterior Belief After Measurement')
    plt.ylim(0, 1)
    plt.ylabel(r'$bel(x_t})$')

    plt.show()




# Bayes Filter Problem
"""
Follow steps of Bayes filter.  
You can use the plot_fuction() above to help you check the belief in each step.
Please print out the final answer.
"""
belief = 0.1 * np.ones(10)

#############################################################################
#                    TODO: Implement your code here                         #
#############################################################################

# x0->3->x1 ->4-> x2

def cal_cbelief(pbeliefs):

    observation_vector = 0.4 * np.ones(10)
    observation_vector[0]= observation_vector [3] = observation_vector[6] = 0.8

    cbelief = np.multiply(observation_vector, pbeliefs)
    cbelief = cbelief/np.sum(cbelief)

    return cbelief


def cal_pbelief(control, prev_cbeliefs):
    
    # here, we only consider two different condition, either is robot could receive signal, or not
    # for example, in the first step,  p(x1=3|u1=3,x0=3)=0.5, p(x1=3|u1=3,x0=0)=0.5, others = 0 
    def get_motion_model(next_state, control, current_state):
        
        expect_state = current_state+control
        if expect_state > 9:
            expect_state -= 10
        
        if next_state == expect_state:
            return 0.5
        elif next_state == current_state:
            return 0.5
        else:
            return 0
        

    cbeliefs = np.zeros(10)
    for next_idx in range(10):

        for current_idx in range(10):
            motion_model = get_motion_model(next_idx, control, current_idx)
            cbeliefs[next_idx] += motion_model * prev_cbeliefs[current_idx]

    return cbeliefs



# prior
belief = cal_cbelief(belief)
print(f"prior {belief}")
controls = [3, 4]
for c in controls:
    pbelief = cal_pbelief(c, belief)

    belief = cal_cbelief(pbelief)
   

    print("belief state c=%s    pbelief           belief" % (c) )
    for i in range(10):
        print("%6d %18.3f %18.3f\n" % (i, pbelief[i], belief[i]))

#############################################################################
#                            END OF YOUR CODE                               #
#############################################################################
# plt.bar(np.arange(0,10), belief)

# print("belief state     probability")
# for i in range(10):
#     print("%6d %18.3f\n" % (i, belief[i]))