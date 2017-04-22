import numpy as np

try:
    from itertools import izip
except ImportError:  #python3.x
    izip = zip

from collections import namedtuple



GarnetData = namedtuple('GarnetData', ['state', 'next_state', 'action', 'reward', 'next_player'])



class GarnetDataset(object):
    def __init__(self, samples, Ns, Na ):

        self.samples = samples

        self.Ns = Ns
        self.Na = Na

        self.no_player = len(samples[0].reward)
        self.no_samples = len(self.samples)

        self.state=np.zeros((self.no_samples, self.Ns))
        self.next_state=np.zeros((self.no_samples, self.Ns))
        self.action=np.zeros((self.no_samples, self.Na))
        self.reward=np.zeros((self.no_samples, self.no_player))
        self.next_player=np.zeros(self.no_samples, dtype=int)

        for i, sample in enumerate(self.samples):
            self.state[i, sample.state] = 1
            self.next_state[i, sample.next_state] = 1
            self.action[i, sample.action] = 1
            self.reward[i] = sample.reward
            self.next_player[i] = sample.next_player


        self.epoch_completed = 0
        self.index_epoch_completed = 0


    def State(self):
        return self.state

    def Action(self):
        return self.action

    def NextState(self):
        return self.next_state

    def Reward(self):
        return self.reward

    def NextPlayer(self):
        return self.next_player

    def EpochCompleted(self):
        return self.epoch_completed

    def IndexEpochCompleted(self):
        return self.index_epoch_completed

    def NoSamples(self):
        return self.no_samples



    def NextBatch(self,size_batch, shuffle = True):

        # return a minibatch of size sizeBatch
        start = self.index_epoch_completed
        self.index_epoch_completed = self.index_epoch_completed + size_batch

        #when all the samples are used, restart and shuffle
        if self.index_epoch_completed > self.no_samples:

            self.epoch_completed +=1

            #reset indices
            start = 0
            self.index_epoch_completed = size_batch
            assert size_batch <= self.no_samples

            if shuffle:

                #inplace permutation
                permute = np.arange(self.no_samples)
                np.random.shuffle(permute)

                #shuffle data
                self.state  = self.state[permute]
                self.action  = self.action[permute]
                self.next_state = self.next_state[permute]
                self.reward = self.reward[permute]
                self.next_player = self.next_player[permute]


        end = self.index_epoch_completed

        return GarnetData(
            state=self.state[start:end],
            next_state=self.next_state[start:end],
            action=self.action[start:end],
            reward=self.reward[start:end],
            next_player=self.next_player[start:end])




