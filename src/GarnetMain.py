__author__ = 'florian-strub'
import os
import pickle
import collections
import tensorflow as tf

from GarnetNetwork import GarnetNetwork
from GarnetDataset import GarnetDataset
from Garnet import Garnet
from tools import *
from argparse import Namespace

import matplotlib.pyplot as plt

Dataset = collections.namedtuple('Dataset', ['train', 'test', 'eval'])

params = Namespace(

    Ns=100,
    Na=5,
    Nb=1,

    noPlayer=2,

    gamma=0.9,

    garnet="S_linear_T2_zero",
    sparsity=0.5,

    coefSample=5,

    Qlayers=[80],
    Pilayers=[80],

    Qregu=1e-7,
    Pregu=1e-7,
    dropout=1,

    Q_lrt=0.001,
    Pi_lrt=5e-05,

    epoch=5000,
    batchSize=20,

    gpu=1,
    outDir=".",
    seed=-1,
)


def start(params, garnet=None, dataset=None):
    print (params)

    no_samples = params.Ns * params.Na * params.coefSample

    if params.seed > -1:
        np.random.seed(params.seed)
        rd.seed(params.seed)
        tf.set_random_seed(params.seed)

    ##############################################
    # Build training dataset



    if garnet is None:
        garnet = Garnet(params.Ns, params.Na, params.Nb, params.noPlayer, params.sparsity, params.garnet)

        if dataset is None:
            dataset = Dataset(
                train=GarnetDataset(Ns=params.Ns, Na=params.Na, samples=garnet.uniform_batch_data(no_samples)),
                test=GarnetDataset(Ns=params.Ns, Na=params.Na, samples=garnet.uniform_batch_data(params.Ns * params.Na)),
                eval=GarnetDataset(Ns=params.Ns, Na=params.Na, samples=garnet.eval_batch_data()),
            )

        print ("Starting points")
        print (garnet.start)

    #############################################
    # Compute random policy

    policy_random = (1.0 * np.ones((garnet.Ns, garnet.Na))) / (1.0 * garnet.Na)
    err_random = garnet.l2errorDiffQstarQpi(policy_random, params.gamma)
    print ("Error random policy Bellman")
    print (err_random)
    print ("")

    #############################################
    # build network

    Qlayers = [params.Ns] + params.Qlayers + [params.Na]
    Pilayers = [params.Ns] + params.Pilayers + [params.Na]

    print (["Qlayers  : ", Qlayers])
    print (["Pilayers : ", Pilayers])


    fApp = GarnetNetwork(qLayerSize=Qlayers, policyLayerSize=Pilayers, params=params)
    plots = fApp.execute(dataset=dataset, nEpoch=params.epoch, nMiniBatch=params.batchSize, garnet=garnet, gamma=params.gamma)

    #############################################
    # plot results

    print("Final Error random policy Bellman")
    print (err_random)

    res = {}
    res["garnet"] = garnet
    res["params"] = params
    res["plots"] = plots

    with open(os.path.join(params.outDir, "data.p"), "wb") as file:
        pickle.dump(res, file=file)

    # plot with various axes scales

    x = range(0, len(plots["errResidual"]["train"]))

    # Two subplots, the axes array is 1-d
    fig, axarr = plt.subplots(3, sharex=True)

    axarr[0].set_title(str(params))

    axarr[0].plot(x, np.log10(plots["errResidual"]["train"]), color='b', linestyle='-')
    axarr[0].plot(x, np.log10(plots["errResidual"]["test"]), color='r', linestyle='-')
    axarr[0].legend(["Train residual Error", "Test residual Error"])

    colors = ["b", "k", "r", "g", "y", "m"]
    legends = []
    for player in range(params.noPlayer):
        axarr[1].plot(x, plots["errPolicy"][player], color=colors[player], linestyle='-')
        axarr[1].plot(x, [err_random[player]] * len(x), color=colors[player], linestyle='--')

        legends += ["NN policy " + str(player), "RND policy " + str(player)]

    axarr[1].legend(legends)
    axarr[1].set_ylim([0.0, 1.0])

    actions = np.array(plots["histoAction"])
    legend = []
    for i in range(params.Na):
        axarr[2].plot(x, actions[:, i])
        legend.append('action ' + str(i))
    axarr[2].legend(legend)
    axarr[2].set_ylim([0.0, 1.0])

    fig.savefig(params.outDir + "/training.eps")
    plt.close(fig)

    return plots

if __name__ == '__main__':
    start(params)
