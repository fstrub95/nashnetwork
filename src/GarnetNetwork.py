__author__ = 'florian-strub'
import tensorflow as tf
import numpy as np
from collections import defaultdict
from Layer import Layer


def computeNetwork(input, layers, dropout):
    y = input
    for layer in layers[:-1]:
        y = tf.matmul(y, layer.w) + layer.b
        y = tf.nn.tanh(y)
        y = tf.nn.dropout(y, dropout)

    y = tf.matmul(y, layers[-1].w) + layers[-1].b
    y = tf.nn.softmax(y)

    return y


class GarnetNetwork(object):
    def buildQNetwork(self, state, next_state, qLayerSize, indexPlayer):

        with tf.name_scope('QNetwork_' + str(indexPlayer)):
            qLayers = [Layer(qLayerSize[i], qLayerSize[i + 1], "layer" + str(i)) for i in range(len(qLayerSize) - 1)]

            y_Qa, y_Qb = state, next_state
            for layer in qLayers[:-1]:
                y_Qa, y_Qb = tf.matmul(y_Qa, layer.w) + layer.b, \
                             tf.matmul(y_Qb, layer.w) + layer.b

                y_Qa, y_Qb = tf.nn.elu(y_Qa) \
                    , tf.nn.elu(y_Qb)

                y_Qa, y_Qb = tf.nn.dropout(y_Qa, self.dropout), \
                             tf.nn.dropout(y_Qb, self.dropout)

            y_Qa, y_Qb = tf.matmul(y_Qa, qLayers[-1].w) + qLayers[-1].b \
                , tf.matmul(y_Qb, qLayers[-1].w) + qLayers[-1].b

        with tf.name_scope('Qa_' + str(indexPlayer)):
            y_Qa = tf.reduce_sum(tf.multiply(y_Qa, self.action_mask), reduction_indices=1, keep_dims=False)

        return y_Qa, y_Qb

    def flow_gradient_for_network(self, x, Q=False, Pi=False):

        assert (Q or Pi)

        def use_gradient():
            return x

        def stop_gradient():
            return tf.stop_gradient(x)

        if Q:
            keep_network = self.keep_network[0]
        else:
            keep_network = self.keep_network[1]

        x_ = tf.cond(tf.equal(keep_network, 1), use_gradient, stop_gradient)

        return x_

    def buildPolicyNetwork(self, input, controller, policyLayerSize):

        pi_networks = []

        for player in range(self.noPlayer):
            with tf.name_scope('policy' + str(player)):
                policyLayers = [Layer(policyLayerSize[i], policyLayerSize[i + 1], "layer" + str(i)) for i in range(len(policyLayerSize) - 1)]
                pi = computeNetwork(input, policyLayers, self.dropout)
                pi_networks.append(pi)

        policies = pi_networks[0]
        for player in range(1, self.noPlayer):
            player_vector = tf.multiply(controller, 0) + player
            use_policy_of_player = tf.equal(player_vector, controller, name="policy_picker")

            policies = tf.where(use_policy_of_player, pi_networks[player], policies)

        return policies

    def __del__(self):
        tf.reset_default_graph()

    def __init__(self, qLayerSize, policyLayerSize, params):

        self.noPlayer = params.noPlayer

        self.Ns = params.Ns
        self.Na = params.Na

        self.err = {}

        # check that the provide network dimension fit regarding the input/output
        assert (qLayerSize[0] == self.Ns and qLayerSize[-1] == self.Na)
        assert (policyLayerSize[0] == self.Ns and policyLayerSize[-1] == self.Na)

        # constant
        self.gamma = tf.constant(params.gamma)
        self.Q_reg_constant = tf.to_float(tf.constant(params.Qregu))
        self.Pi_reg_constant = tf.to_float(tf.constant(params.Pregu))

        self.Q_lrt = tf.to_float(tf.constant(params.Q_lrt))
        self.Pi_lrt = tf.to_float(tf.constant(params.Pi_lrt))

        # input
        self.state = tf.placeholder(tf.float32, shape=[None, self.Ns], name="state")
        self.next_state = tf.placeholder(tf.float32, shape=[None, self.Ns], name="next_state")
        self.controller = tf.placeholder(tf.int32, shape=[None], name="controller")
        self.action_mask = tf.placeholder(tf.float32, shape=[None, self.Na], name="action_mask")
        self.dropout = tf.placeholder(tf.float32, name="dropout")

        self.keep_network = tf.placeholder_with_default(tf.ones(shape=[2], dtype=tf.int32), shape=[2], name="keep_network")

        # target
        with tf.name_scope('Reward'):
            self.target = tf.placeholder(tf.float32, shape=[None, self.noPlayer], name="target")
            reward = tf.split(self.target, self.noPlayer, axis=1, name="target2reward")  # split target by players)
            reward = [tf.reshape(r, [-1]) for r in reward]

        ###########################
        # Build Networks
        ###########################

        # build QNetwork
        self.networksOfPlayers = defaultdict(list)
        with tf.name_scope('QNetworks'):
            for player in range(self.noPlayer):
                Qa, Qb = self.buildQNetwork(self.state, self.next_state, qLayerSize, player)

                Qa = self.flow_gradient_for_network(Qa, Q=True)
                Qb = self.flow_gradient_for_network(Qb, Q=True)

                self.networksOfPlayers["Qa"].append(Qa)
                self.networksOfPlayers["Qb"].append(Qb)

        # build policyNetwork
        with tf.name_scope('PolicyNetworks') as scope_name:

            policies = self.buildPolicyNetwork(self.next_state, self.controller, policyLayerSize)
            self.policies = self.flow_gradient_for_network(policies, Pi=True)

        ###########################
        # Compute data flows
        ###########################

        # Comute the expected Q_value
        QbMean = []
        with tf.name_scope('PickQmean'):
            for player, Qb in enumerate(self.networksOfPlayers["Qb"]):
                with tf.name_scope(name='Qmean_' + str(player)) as scope_name:
                    oneQbMean = tf.reduce_sum(tf.multiply(Qb, self.policies), reduction_indices=1, keep_dims=False)
                    QbMean.append(oneQbMean)

        # for each player pick an action
        QbMax = []
        with tf.name_scope('PickQmax'):
            for player, oneQb in enumerate(self.networksOfPlayers["Qb"]):
                with tf.name_scope('PickQmax_' + str(player)) as scope_name:
                    oneQmax = tf.reduce_max(oneQb, reduction_indices=1, keep_dims=False, name='Qmax_' + str(player))
                    oneQmean = QbMean[player]

                    player_vector = tf.multiply(self.controller, 0) + player
                    is_same_player = tf.equal(player_vector, self.controller, name="is_same_player")
                    oneQbMax = tf.where(is_same_player, oneQmax, oneQmean)

                    QbMax.append(oneQbMax)

        ###########################
        # Compute Loss/residuals
        ###########################

        # compute the residual for every players
        residuals = []
        with tf.name_scope('Residual'):
            for player, oneQa, oneQbMean, oneQbMax, one_reward in zip(range(len(reward)), self.networksOfPlayers["Qa"], QbMean, QbMax, reward):
                with tf.name_scope('residual_Qoptimal_player' + str(player)) as scope_name:
                    residual_Qoptimal = tf.square(oneQa - one_reward - self.gamma * oneQbMax)

                with tf.name_scope('residual_Qmean_player' + str(player)) as scope_name:
                    residualQmean = tf.square(oneQa - one_reward - self.gamma * oneQbMean)

                residuals.append(residual_Qoptimal + residualQmean)

            # Pack the residuals for player in a single tensor
            residuals = tf.stack(residuals)
            residuals = tf.reduce_sum(residuals, reduction_indices=0, keep_dims=False)

        # Optimize the (normalized) residual
        self.loss = tf.reduce_mean(residuals, reduction_indices=0, keep_dims=False)

        # compute L2 reg
        with tf.name_scope('QOptim'):

            with tf.name_scope('Regularization'):
                variables = [v for v in tf.trainable_variables() if v.name.startswith('QNetworks')]
                l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in variables]) * self.Q_reg_constant

            # self.optimizer_Q = tf.train.RMSPropOptimizer(learning_rate=self.Q_lrt, momentum=0.8, decay=0.5).minimize(self.loss + l2_reg, var_list=variables)
            self.optimizer_Q = tf.train.AdamOptimizer(learning_rate=self.Q_lrt).minimize(self.loss + l2_reg, var_list=variables)

        with tf.name_scope('PiOptim'):

            with tf.name_scope('Regularization'):
                variables = [v for v in tf.trainable_variables() if v.name.startswith('PolicyNetworks')]
                l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in variables]) * self.Pi_reg_constant * l2_reg

            # self.optimizer_Pi = tf.train.RMSPropOptimizer(learning_rate=self.Pi_lrt, momentum=0.8, decay=0.5).minimize(self.loss + l2_reg, var_list=variables)
            self.optimizer_Pi = tf.train.AdamOptimizer(learning_rate=self.Pi_lrt).minimize(self.loss + l2_reg, var_list=variables)

        ###########################
        # Configure training
        ###########################

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.23)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

    def execute(self, dataset, nEpoch, nMiniBatch, garnet=None, gamma=None):

        self.err["errPolicy"] = [[] for _ in range(self.noPlayer)]
        self.err["errResidual"] = {"train": [], "test": []}
        self.err["errBellmanExact"] = []
        self.err["histoAction"] = []

        print ("############ Intial state:")
        self.eval(dataset=dataset, garnet=garnet, gamma=gamma, step=0)

        print ("############ Start training:")
        for t in range(nEpoch):

            self.train(train_dataset=dataset.train, nMiniBatch=nMiniBatch)

            if t % 50 == 0 and t > 0:
                print ("")
                print ("epoch : " + str(t))
                self.eval(dataset=dataset, garnet=garnet, gamma=gamma, step=t)

        print ("############ Final State:")
        self.eval(dataset=dataset, garnet=garnet, gamma=gamma, step=nEpoch)

        return self.err

    def eval(self, dataset, garnet, gamma, step):

        self.err["errResidual"]["train"].append(self.eval_residual(dataset=dataset.train))
        self.err["errResidual"]["test"].append(self.eval_residual(dataset=dataset.test))

        print ("Train residual: {0}   \t   Test residual: {1} ".format(self.err["errResidual"]["train"][-1],
                                                                       self.err["errResidual"]["test"][-1]))
        self.eval_policy(eval_dataset=dataset.eval, garnet=garnet, gamma=gamma)

    def eval_policy(self, eval_dataset, garnet, gamma):

        policies = self.sess.run(self.policies, feed_dict={
            self.state: eval_dataset.State(),
            self.next_state: eval_dataset.NextState(),
            self.action_mask: eval_dataset.Action(),
            self.controller: eval_dataset.NextPlayer(),
            self.dropout: 1})

        # Compute err
        err = garnet.l2errorDiffQstarQpi(policies, gamma)

        histoAction = np.histogram(policies.argmax(axis=1), bins=np.linspace(-0.5, self.Na - 0.5, num=self.Na + 1))[0]
        histoAction = 1.0 * histoAction / policies.shape[0]

        # Stock err
        for player, one_err in enumerate(err):
            self.err["errPolicy"][player].append(one_err)

        self.err["histoAction"].append(histoAction)

        # Print err
        print("Error Greedy policy Bellman")
        print (err)
        print("Actions: ")
        print (histoAction)

        return policies

    def eval_residual(self, dataset):

        err = self.sess.run(self.loss, feed_dict={
            self.state: dataset.State(),
            self.next_state: dataset.NextState(),
            self.action_mask: dataset.Action(),
            self.controller: dataset.NextPlayer(),
            self.target: dataset.Reward(),
            self.dropout: 1})

        return err

    def train(self, train_dataset, nMiniBatch):

        n_iter = int(train_dataset.NoSamples() / nMiniBatch) + 1

        for i in range(n_iter):
            # Creating the mini-batch
            batch = train_dataset.NextBatch(nMiniBatch)

            # running one step of the optimization method on the mini-batch
            self.sess.run((self.optimizer_Pi, self.optimizer_Q), feed_dict={
                self.state: batch.state,
                self.next_state: batch.next_state,
                self.action_mask: batch.action,
                self.controller: batch.next_player,
                self.target: batch.reward,
                self.dropout: 1})
