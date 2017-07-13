import numpy as np
import mdptoolbox, mdptoolbox.example

# #3 States, 2 Actions
# transitions = np.array([[[1/3,1/3,1/3],
#                          [1/3,1/3,1/3],
#                          [1/3,1/3,1/3]],
#                         [[1/3,1/3,1/3],
#                          [1/3,1/3,1/3],
#                          [1/3,1/3,1/3]]])
#
# # Reward matrices or vectors. Like the transition matrices, these can also be
# # defined in a variety of ways. Again the simplest is a numpy array
# rewards = np.array([[-1,1],
#                     [-1,1],
#                     [-1,1]])
#
# # Discount factor. The per time-step discount factor on future rewards.
# # Valid values are greater than 0 upto and including 1. If the discount factor is 1,
# # then convergence is cannot be assumed and a warning will be displayed.
# discount = 0.9
#
# # Stopping criterion. The maximum change in the value function at each iteration
# # is compared against epsilon. Once the change falls below this value, then the
# # value function is considered to have converged to the optimal value function.
# epsilon = 0.9
#
# # maximum iterations
# max_iter = 1000
#
# # By default we run a check on the transitions and rewards arguments to make sure
# # they describe a valid MDP. You can set this argument to True in order to skip this check.
# skip_check = False
#
# print('Transition Matrix: \n',transitions)
# print('\nReward Matrix: \n',rewards)
# print('\nDiscount: \n',discount)
# print('\nEpsilon: \n',epsilon)
# print('\nMaximum Iterations: \n',max_iter)
# print('\nSkip Check: \n',skip_check)
#
# env = mdptoolbox.mdp.MDP(transitions=transitions, reward=rewards, discount=discount,epsilon=epsilon, max_iter=max_iter, skip_check=skip_check)


print("\n\n-----------QLearning MDP----------")

#3 States, 2 Actions
transitions = np.array([
                        [[1/4,1/4,1/4,1/4],
                        [1/4,1/4,1/4,1/4],
                        [1/4,1/4,1/4,1/4],
                        [1/4,1/4,1/4,1/4]],

                        [[1/4,1/4,1/4,1/4],
                        [1/4,1/4,1/4,1/4],
                        [1/4,1/4,1/4,1/4],
                        [1/4,1/4,1/4,1/4]],

                        [[1/4,1/4,1/4,1/4],
                        [1/4,1/4,1/4,1/4],
                        [1/4,1/4,1/4,1/4],
                        [1/4,1/4,1/4,1/4]],

                        [[1/4,1/4,1/4,1/4],
                        [1/4,1/4,1/4,1/4],
                        [1/4,1/4,1/4,1/4],
                        [1/4,1/4,1/4,1/4]],

                        [[0,0,0,1],
                        [0,0,0,1],
                        [0,0,0,1],
                        [0,0,0,1],]])


# Reward matrices or vectors. Like the transition matrices, these can also be
# defined in a variety of ways. Again the simplest is a numpy array
rewards = np.array([[  1, -1, -1, -1, -1],
                    [ -1,  1,  0,  0,  0],
                    [  0, -1,  1,  1,  1],
                    [-10,-10,-10,-10,-10]])

# Discount factor. The per time-step discount factor on future rewards.
# Valid values are greater than 0 upto and including 1. If the discount factor is 1,
# then convergence is cannot be assumed and a warning will be displayed.
discount = 0.9

# Number of iterations to execute.
n_iter = 100000

# By default we run a check on the transitions and rewards arguments to make sure
# they describe a valid MDP. You can set this argument to True in order to skip this check.
skip_check = False

print('Transition Matrix: \n',transitions)
print('\nReward Matrix: \n',rewards)
print('\nDiscount: \n',discount)
print('\nN Iterations: \n',n_iter)
print('\nSkip Check: \n',skip_check)

env = mdptoolbox.mdp.QLearning(transitions=transitions, reward=rewards, discount=discount, n_iter=n_iter)

for i in range(0,10):
    env.run()
    print('Policy',i,':',env.policy)




q_table = env.Q
val_fn = env.V
policy = env.policy


print('\n\nQTable: \n',q_table)
print('\nOptimal Policy:', policy)
print('done')
