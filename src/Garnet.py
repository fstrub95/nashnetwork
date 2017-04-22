__author__ = 'julien-perolat'

from scipy import *
from tools import *
from itertools import product

from collections import namedtuple

try:
    from itertools import izip
except ImportError:  # python3.x
    izip = zip

GarnetSample = namedtuple('GarnetSample', ['state', 'next_state', 'action', 'reward', 'next_player'])

BETA1 = 0.01
BETA2 = 0.99


####################################


class Garnet:
    def __init__(self, Ns, Na, Nb, no_player, sparsity, type_gar):

        # nbr etats
        self.Ns = Ns
        self.Na = Na
        self.no_player = no_player
        self.start = []

        if type_gar == "S_linear_T2":
            self.kernel, reward0, reward1, self.control = garnet_gen_s_linear_old(Ns, Na, Nb, sparsity, Nb)
            self.reward = np.zeros((2, Ns, Na))
            self.reward[0] = reward0
            self.reward[1] = reward1
        if type_gar == "S_linear_T2_zero":
            self.kernel, reward0, reward1, self.control = garnet_gen_s_linear_old(Ns, Na, Nb, sparsity, Nb)
            self.reward = np.zeros((2, Ns, Na))
            self.reward[0] = reward0
            self.reward[1] = -reward0


        elif type_gar == "S_thor":
            self.kernel, self.reward, self.control, self.start = garnet_gen_thor(Ns, Na, sparsity, no_player)
        elif type_gar == "S_thor_zero_sum":
            self.kernel, self.reward, self.control, self.start = garnet_gen_thor(Ns, Na, sparsity, 2)
            self.reward[1] = -self.reward[0]


            # elif type_gar == "S_linear_T1":
            #     kernel, reward0, reward1, control = garnet_gen_s_linear( Ns, Na, Nb, sparsity, Ns)
            # elif type_gar == "S_linear_T2_testJ0":
            #     kernel, reward0, reward1, control = garnet_gen_s_linear_test( Ns, Na, Nb, sparsity, Ns, 0)
            # elif type_gar == "S_linear_T2_testJ1":
            #     kernel, reward0, reward1, control = garnet_gen_s_linear_test( Ns, Na, Nb, sparsity, Ns, 1)
            # if type_gar == "SA_T2":
            #     kernel, reward0, reward1, control = garnet_gen_sa( Ns, Na, Nb, sparsity, Nb)
            # elif type_gar == "SA_T1":
            #     kernel, reward0, reward1, control = garnet_gen_sa( Ns, Na, Nb, sparsity, Ns)
            # elif type_gar == "S_T2":
            #     kernel, reward0, reward1, control = garnet_gen_s( Ns, Na, Nb, sparsity, Nb)
            # elif type_gar == "S_T1":
            #     kernel, reward0, reward1, control = garnet_gen_s( Ns, Na, Nb, sparsity, Ns)

    def compute_sample(self, state, action, policy=None):

        # Compute uniform transition probability given a state and the action
        transition_probability = self.kernel[:, state, action]

        next_state = np.random.choice(self.Ns, 1, p=transition_probability)[0]
        next_player = self.control[next_state]
        reward = self.reward[:, state, action]

        one_sample = GarnetSample(
            state=state,
            next_state=next_state,
            action=action,
            reward=reward,
            next_player=next_player
        )

        return one_sample

    def uniform_batch_data(self, no_samples):

        batch = []

        for j in range(no_samples):
            state = rd.randint(0, self.Ns - 1)
            action = rd.randint(0, self.Na - 1)

            one_sample = self.compute_sample(state, action)

            batch.append(one_sample)  # uniform sampling

        return batch

    def eval_batch_data(self):

        batch = [GarnetSample(
            state=state,
            next_state=state,
            action=0,
            reward=[0.] * self.no_player,
            next_player=self.control[state])

                 for state in range(self.Ns)
                 ]

        return batch

    def eval_batch_data_for_Q(self):

        batch = [GarnetSample(
            state=state,
            next_state=state,
            action=action,
            reward=[0.] * self.no_player,
            next_player=self.control[state])

                 for action in range(self.Na)
                 for state in range(self.Ns)
                 ]

        return batch

    # TEST OK !!!
    def policy_evaluation_exact_v(self, policy_list, gamma):

        reward_average_policy = np.zeros((self.no_player, self.Ns))
        req_non_stat = np.zeros((self.no_player, self.Ns))
        v_non_stat = np.zeros((self.no_player, self.Ns))

        ker_pi = np.zeros((self.Ns, self.Ns))

        ker_non_stat = np.identity(self.Ns)
        Id = np.identity(self.Ns)

        for policy in policy_list:

            # compute kernel
            for i, j in np.ndindex((self.Ns, self.Ns)):
                ker_pi[j, i] = np.vdot(self.kernel[i, j, :], policy[j, :])

            ker_non_stat = gamma * ker_pi.dot(ker_non_stat)

            for player in range(self.no_player):
                for state in range(self.Ns):
                    reward_average_policy[player][state] = np.vdot(self.reward[player][state, :], policy[state, :])

                req_non_stat[player] = reward_average_policy[player] + gamma * ker_pi.dot(req_non_stat[player])

        for player, req_non_stat_player in enumerate(req_non_stat):
            v_non_stat[player] = np.linalg.solve(Id - ker_non_stat, req_non_stat_player)

        return v_non_stat

    # TEST OK !!!
    def policy_evaluation_exact_Q(self, policy_list, gamma):

        Q_non_stat = np.zeros((self.no_player, self.Ns, self.Na))
        v_non_stat = self.policy_evaluation_exact_v(policy_list, gamma)

        for player in range(self.no_player):
            for state, action in np.ndindex((self.Ns, self.Na)):
                Q_non_stat[player][state, action] = self.reward[player][state, action] + gamma * np.vdot(v_non_stat[player], self.kernel[:, state, action])

        return Q_non_stat

    # TEST OK !!!
    def greedy_policy(self, Q_function):

        policy = np.zeros((self.Ns, self.Na))

        for player in range(self.no_player):
            for state in range(self.Ns):
                next_player = self.control[state]
                next_action = np.argmax(Q_function[next_player, state, :], axis=0)  # optimal action regarding the Qfuntion of the next_player (greedy)

                policy[state, next_action] = 1

        return policy

    # TEST OK !!!
    def greedy_best_response(self, policy_list, Q_function, gamma):

        # Defensive programming
        Q_function = np.copy(Q_function)

        # Initialize output with empty list
        res_policy_list = []
        for _ in range(self.no_player):
            res_policy_list.append([])

        greedy_policy = np.zeros(Q_function.shape)

        for policy in policy_list:

            # compute greedy policies
            for one_greedy_policy in greedy_policy:
                one_greedy_policy[:] = self.greedy_policy(Q_function)

            # Select the policies of the controller/next_player
            for state in range(self.Ns):

                next_player = self.control[state]
                for player, one_greedy_policy in enumerate(greedy_policy):
                    if player != next_player:
                        one_greedy_policy[state, :] = policy[state, :]

            for one_greedy_policy, one_res_policy_list in izip(greedy_policy, res_policy_list):
                one_res_policy_list += [one_greedy_policy]

            # Compute next Q_function
            V_function = []
            for one_Q, one_policy in izip(Q_function, greedy_policy):
                one_v = [np.vdot(one_Q[state, :], one_policy[state, :]) for state in range(self.Ns)]
                V_function.append(one_v)
            V_function = np.array(V_function)

            for Q, reward, V in izip(Q_function, self.reward, V_function):
                for state, action in np.ndindex((self.Ns, self.Na)):
                    Q[state, action] = reward[state, action] + gamma * np.vdot(V, self.kernel[:, state, action])

        return res_policy_list

    # TEST OK !!!
    def policy_best_response(self, policy_list, gamma):

        Q_non_stat = self.policy_evaluation_exact_Q(policy_list, gamma)  # evaluation of the policy of policy_list__0
        best_response_policy = self.greedy_best_response(policy_list, Q_non_stat, gamma)

        prev_Q_non_stat = np.copy(Q_non_stat)
        for player in range(self.no_player):
            Q_out = self.policy_evaluation_exact_Q(best_response_policy[player], gamma)
            Q_non_stat[player] = Q_out[player]

        i = 0
        is_stationary = [False]
        while not all(is_stationary) and np.linalg.norm(prev_Q_non_stat - Q_non_stat) > 10e-12:

            # Store previous values
            prev_Q_non_stat = np.copy(Q_non_stat)
            prev_best_response_policy = np.copy(best_response_policy)

            is_stationary = []

            for player in range(self.no_player):
                # compute the policy best response
                out_policies = self.greedy_best_response(best_response_policy[player], prev_Q_non_stat, gamma)
                best_response_policy[player] = out_policies[player]

                # update Q stationary
                Q_out = self.policy_evaluation_exact_Q(best_response_policy[player], gamma)
                Q_non_stat[player] = Q_out[player]

                # check whether the policy was updated
                same_policy = [np.array_equal(prev_policy, new_policy) for prev_policy, new_policy in zip(prev_best_response_policy[player], best_response_policy[player])]
                is_stationary += [all(same_policy)]

        return best_response_policy

    # TEST OK !!!
    def exact_best_response_v(self, policy_list, gamma):
        best_response_policies = self.policy_best_response(policy_list, gamma)

        final = []
        for player, one_best_policy in enumerate(best_response_policies):
            value = self.policy_evaluation_exact_v(one_best_policy, gamma)

            # discard Q/V of other players
            final.append(value[player])

        return final

    # TEST OK !!!
    def exact_best_response_Q(self, policy_list, gamma):
        best_response_policies = self.policy_best_response(policy_list, gamma)

        final = []
        for player, one_best_policy in enumerate(best_response_policies):
            value = self.policy_evaluation_exact_Q(one_best_policy, gamma)

            # discard Q/V of other players
            final.append(value[player])

        return final

    # NOT TESTED
    def Apply_bellman(self, policy_list, Q_function, gamma):

        Q_function = np.copy(Q_function)

        for policy in policy_list:
            for player in range(self.no_player):
                v = np.asarray([np.vdot(Q_function[player][state, :], policy[state, :]) for state in range(self.Ns)])

                for state, action in np.ndindex((self.Ns, self.Na)):
                    Q_function[player][state, action] = \
                        self.reward[player][state, action] + gamma * np.vdot(v, self.kernel[:, state, action])

        return Q_function

    def l2(self, estimate, target):
        return np.linalg.norm(estimate - target) / np.linalg.norm(target)

    def l2errorDiffQstarQpi(self, policy, gamma):
        Qstar = self.exact_best_response_Q([policy], gamma)
        Qpi = self.policy_evaluation_exact_Q([policy], gamma)

        res = [self.l2(one_Qpi, one_Qstar) for one_Qpi, one_Qstar in izip(Qpi, Qstar)]

        return res

    # NOT TESTED
    def merge_policy(self, pi0, pi1):
        policy = np.zeros((self.Ns, self.Na))
        for i in range(self.Ns):
            if self.control[i] == 1:
                policy[i, :] = pi1[i, :]
            elif self.control[i] == 0:
                policy[i, :] = pi0[i, :]
        return policy


####################################################################################################


def garnet_gen_sa(Ns, Na, Nb, sparsity, neighbor):
    ### generating the Kernel
    kernel = np.zeros((Ns, Ns, Na))  # p(s'|s,a)
    for i, j in np.ndindex((Ns, Na)):
        echantillon = rd.sample(list(set(range(Ns)).intersection(range(i - neighbor, i + neighbor))), Nb)
        cumulative = np.concatenate(([0], sort([rd.random() for k in range(Nb - 1)]), [1]), axis=0)
        for k in range(Nb):
            kernel[echantillon[k], i, j] = cumulative[k + 1] - cumulative[k]

    ### generating rewards at random
    reward0 = np.random.randn(Ns, Na)
    reward1 = np.random.randn(Ns, Na)

    masque_reward = np.zeros((Ns, Na))
    N_sparsity = int(Ns * sparsity)
    i = 0
    while i < N_sparsity:
        i_ = rd.randint(0, Ns - 1)
        if masque_reward[i_, 0] == 0:
            masque_reward[i_, :] = 1
            i += 1
    reward0 = reward0 * masque_reward
    reward1 = reward1 * masque_reward
    control = np.random.randint(2, size=Ns)

    return kernel, reward0, reward1, control


def garnet_gen_s(Ns, Na, Nb, sparsity, neighbor):
    ### generating the Kernel
    kernel = np.zeros((Ns, Ns, Na))  # p(s'|s,a)
    for i, j in np.ndindex((Ns, Na)):
        echantillon = rd.sample(list(set(range(Ns)).intersection(range(i - neighbor, i + neighbor))), Nb)
        cumulative = np.concatenate(([0], sort([rd.random() for k in range(Nb - 1)]), [1]), axis=0)
        for k in range(Nb):
            kernel[echantillon[k], i, j] = cumulative[k + 1] - cumulative[k]

    ### generating rewards at random
    reward0 = np.zeros((Ns, Na))
    reward1 = np.zeros((Ns, Na))

    biais0 = np.random.randn(Ns)
    biais1 = np.random.randn(Ns)
    for i, j in np.ndindex((Ns, Na)):
        reward0[i, j] = biais0[i]
        reward1[i, j] = biais1[i]

    masque_reward = np.zeros((Ns, Na))
    N_sparsity = int(Ns * sparsity)
    i = 0
    while i < N_sparsity:
        i_ = rd.randint(0, Ns - 1)
        if masque_reward[i_, 0] == 0:
            masque_reward[i_, :] = 1
            i += 1
    reward0 = reward0 * masque_reward
    reward1 = reward1 * masque_reward
    control = np.random.randint(2, size=Ns)

    return kernel, reward0, reward1, control


def garnet_gen_thor(Ns, Na, sparsity, no_player, nn_ratio=0.10):
    print("neighbor 1 and noise in reward")

    ### generating determinist Kernel
    kernel = np.zeros((Ns, Ns, Na))  # p(s'|s,a)

    neighbor = 1  # int(max(math.floor(nn_ratio*Ns/2), 1))

    # kernel -> state space is a thor!
    for state in range(Ns):
        action_next_state = np.random.normal(state, neighbor, Na)  # pick a random state in state surrounding
        action_next_state = np.round(action_next_state).astype(int)
        action_next_state = np.fmod(action_next_state, Ns)  # use modulo to enforce index > Ns (negative indices are already handled by numpy)
        for action, next_state in enumerate(action_next_state):
            kernel[next_state, state, action] = 1  # determinist kernel

    # the reward linearly decrease b its distance to some starting point (warning : thor)
    def get_reward_function(start, Ns):
        return lambda state: (min(math.fabs(state - start), Ns - math.fabs(state - start))) * 2.0 / Ns

    # compute reward
    reward = np.zeros((no_player, Ns, Na))

    start_points = np.random.randint(Ns, size=no_player)

    for player, reward_player in enumerate(reward):

        reward_fct = get_reward_function(start=start_points[player], Ns=Ns)

        for state, reward_state in enumerate(reward_player):
            # reward_state[:] = reward_fct(state)
            reward_state[:] = np.random.normal(reward_fct(state), 0.005, Na)

    # Apply sparsity
    mask = np.random.binomial(1, 1 - sparsity, reward.shape)
    reward *= mask

    # compute control
    control = np.random.randint(no_player, size=Ns)

    return kernel, reward, control, start_points


def garnet_gen_s_linear_old(Ns, Na, Nb, sparsity, neighbor):
    ### generating the Kernel
    kernel = np.zeros((Ns, Ns, Na))  # p(s'|s,a)
    for i, j in np.ndindex((Ns, Na)):
        echantillon = rd.sample(list(set(range(Ns)).intersection(range(i - neighbor, i + neighbor))), Nb)
        cumulative = np.concatenate(([0], sort([rd.random() for k in range(Nb - 1)]), [1]), axis=0)
        for k in range(Nb):
            kernel[echantillon[k], i, j] = cumulative[k + 1] - cumulative[k]

    ### generating rewards at random
    reward0 = np.zeros((Ns, Na))
    reward1 = np.zeros((Ns, Na))

    biais0 = (np.arange(Ns) / (1.0 * (Ns - 1)))
    biais1 = 1 - (np.arange(Ns) / (1.0 * (Ns - 1)))

    for i, j in np.ndindex((Ns, Na)):
        reward0[i, j] = biais0[i]
        reward1[i, j] = biais1[i]

    masque_reward = np.zeros((Ns, Na))
    N_sparsity = int(Ns * sparsity)
    i = 0
    while i < N_sparsity:
        i_ = rd.randint(0, Ns - 1)
        if masque_reward[i_, 0] == 0:
            masque_reward[i_, :] = 1
            i += 1
    reward0 = reward0 * masque_reward
    reward1 = reward1 * masque_reward
    control = np.random.randint(2, size=Ns)

    return kernel, reward0, reward1, control

# Ns = 5
# Na = 3
# Nb = 1
# gamma = 0.9
#
# garnet = Garnet(Ns, Na, Nb, 0.5, "S_linear_T2")
#
# policy_random = (1.0 * np.ones((garnet.Ns, garnet.Na))) / (1.0 * garnet.Na)




# print garnet.policy_evaluation_exact_v_old([policy_random], gamma)
# print garnet.policy_evaluation_exact_v([policy_random], gamma)
#
# print "Next"
#
# print garnet.policy_evaluation_exact_Q_old([policy_random], gamma)
# print garnet.policy_evaluation_exact_Q([policy_random], gamma)





# Qfunction1 = np.array([[ -3.70074342e-16,  -3.70074342e-16],
#        [ -3.70074342e-16,  -3.70074342e-16],
#        [  5.00000000e-01,   1.31818182e+00],
#        [  1.56818182e+00,   2.64669421e+00],
#        [  1.89669421e+00,   1.89669421e+00]])
# Qfunction2 = np.array([[ -1.23358114e-16,  -1.23358114e-16],
#        [ -1.23358114e-16,  -1.23358114e-16],
#        [  5.00000000e-01,   1.31818182e+00],
#        [  1.06818182e+00,   1.32851240e+00],
#        [  1.07851240e+00,   1.07851240e+00]])
#
# Qfunction = np.array([Qfunction1.tolist(), Qfunction2.tolist()])

# print garnet.greedy_policy_old(Qfunction1, Qfunction2)
# print garnet.greedy_policy(Qfunction)


# print garnet.greedy_best_response_old(policy_list, Qfunction1, Qfunction2, gamma)
# print garnet.greedy_best_response( policy_list, Qfunction, gamma)




# print garnet.policy_best_response_old([policy_random], gamma)
# print garnet.policy_best_response([policy_random], gamma)


# print ("Old")
# print (garnet.exact_best_response_Q_old([policy_random], gamma))
#
# print ("New")
# print (garnet.exact_best_response_Q([policy_random], gamma))

# print ("Old")
# print (garnet.exact_best_response_v_old([policy_random], gamma))
#
# print ("New")
# print (garnet.exact_best_response_v([policy_random], gamma))





#
# print garnet.l2errorDiffQstarQpi(policy_random, gamma)
# print garnet.l2errorDiffQstarQpi_old(policy_random, gamma)

# print ("Error random policy Bellman")
# print ([err0_random, err1_random])
# print ("")













# def garnet_gen_s_linear_test( Ns, Na, Nb, sparsity, neighbor, joueur):
#
#
#
#     ### generating the Kernel
#     kernel = np.zeros((Ns, Ns, Na))  # p(s'|s,a)
#     for i, j in np.ndindex((Ns, Na)):
#         echantillon = rd.sample(list(set(range(Ns)).intersection(range(i-neighbor,i+neighbor))), Nb)
#         cumulative = np.concatenate(([0], sort([rd.random() for k in range(Nb - 1)]), [1]), axis=0)
#         for k in range(Nb):
#             kernel[echantillon[k], i, j] = cumulative[k + 1] - cumulative[k]
#
#     ### generating rewards at random
#     reward0 = np.zeros((Ns, Na))
#     reward1 = np.zeros((Ns, Na))
#
#     biais0 = (np.arange(Ns)/(1.0*(Ns-1)))
#     biais1 = 1-(np.arange(Ns)/(1.0*(Ns-1)))
#
#     for i, j in np.ndindex((Ns, Na)):
#         reward0[i,j] = biais0[i]
#         reward1[i,j] = biais1[i]
#
#     masque_reward = np.zeros((Ns, Na))
#     N_sparsity = int(Ns * sparsity)
#     i = 0
#     while i < N_sparsity:
#         i_ = rd.randint(0, Ns - 1)
#         if masque_reward[i_, 0] == 0:
#             masque_reward[i_, :] = 1
#             i += 1
#     reward0 = reward0 * masque_reward
#     reward1 = reward1 * masque_reward
#     control = joueur*np.ones(Ns)
#
#     return Ns, Na, kernel, reward0, reward1, control
