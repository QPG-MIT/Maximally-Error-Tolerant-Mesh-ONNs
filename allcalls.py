# imports #####################################################################

import matplotlib.pyplot as plt
import copy
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Lambda, Dense
from tensorflow.python.keras import backend as K
from tensorflow.keras.datasets import mnist, fashion_mnist
from extra_keras_datasets import kmnist
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib 

import cupy as cp
import helperfunctions as h
import time
import meshes as ms
import meshes.gpu as mg
import math
import numpy as np
import scipy as sp
import seaborn as sns
from collections import namedtuple

import errcorr3mzi_riemann as ec

import neurophox
from neurophox.tensorflow import RM, SVD
from neurophox.ml.nonlinearities import cnormsq
from neurophox.tensorflow.generic import MeshPhasesTensorflow

import mainfunctions as m

# extracting test accuracy from a model parameter file ########################################################################

windowhalfwidth = 8
inputsize = (2*windowhalfwidth)**2
numlayers = 2
gamma_pos = 'out'
splitting_error = 0.23
err = int(splitting_error*100)
runno = 1
dataset = 'mnist'

foldername = 'models/'+dataset+f'/size{inputsize}/err{err}/run{runno}'
filename = 'idealthetas.pickle'

with open(foldername+'/'+filename, 'rb') as inputfile:
    thetas, phis, gammas = pickle.load(inputfile)

mnist_dp = h.MNISTDataProcessor(dataset=dataset)
data = mnist_dp.fourier(windowhalfwidth)
num_test = len(data.y_test_ind)

bs_sigma = 0.5 * np.arcsin(2 * splitting_error)
alphainput = np.ones((numlayers, inputsize, inputsize // 2)) * 2 * bs_sigma
alphainput[:, 1::2, -1] = 0
betainput = np.zeros((numlayers, inputsize, inputsize // 2))

bs_error = [tuple([betainput[i], alphainput[i]]) for i in np.arange(numlayers)]

model = h.const_onn_EO(inputsize, theta_init=thetas, phi_init=phis, gamma_init=gammas, bs_error=bs_error, gamma_pos=gamma_pos)

netout = model.predict(data.x_test)
predoutputs = np.argmax(netout, axis=1)
accuracy = 100*(1 - np.count_nonzero(predoutputs-data.y_test_ind)/num_test)

print(f'The test accuracy is: {accuracy}')

# transferring parameters from the above maximally error tolerant mesh to a different faulty mesh #

thetasideal, phisideal, gammasideal = h.noisytoideal(thetas, phis, gammas, alphainput, betainput, numlayers, inputsize, gamma_pos='out')

alphatest = (2*np.random.rand(numlayers, inputsize, inputsize // 2)-1) * bs_sigma
alphatest[:, 1::2, -1] = 0
betatest = (2*np.random.rand(numlayers, inputsize, inputsize // 2)-1) * bs_sigma
betatest[:, 1::2, -1] = 0
bs_errortest = [tuple([betatest[i], alphatest[i]]) for i in np.arange(numlayers)]

thetastransferred, phistransferred, gammastransferred = h.error_correction(thetasideal, phisideal, gammasideal, alphatest, betatest, 
                                                                           numlayers, inputsize, gamma_pos='out')

modeltransferred = h.const_onn_EO(inputsize, theta_init=thetastransferred, phi_init=phistransferred, gamma_init=gammastransferred,
                                  L=numlayers, bs_error=bs_errortest, gamma_pos='out')

predoutputsmod = np.argmax(modeltransferred.predict(data.x_test), axis=1)
print(f'ported acc is {100*(1 - np.count_nonzero(predoutputsmod-data.y_test_ind)/num_test)}')

# direct training ##############################################################################################################

numlayers = 2
windowhalfwidths = [8, 10]
datasets = ['mnist', 'fashion_mnist', 'kmnist']

for dataset in datasets:
    for windowhalfwidth in windowhalfwidths:
        for splitting_error in np.arange(36):
            m.directtraining(numlayers, windowhalfwidth, dataset, epochs=50, batch_size=512, N_classes=10, nummodels=1, 
                gamma_pos='out', saveflag=True, saveflagcheckpoints=False, splitting_errors=splitting_error, 
                selectedloss=tf.keras.losses.CategoricalCrossentropy(), numruns=5)

# transfer training #########################################################################################################

numlayers = 2
windowhalfwidths = [8, 10]
datasets = ['mnist', 'fashion_mnist', 'kmnist']

for dataset in datasets:
    for windowhalfwidth in windowhalfwidths:
        for runno in np.arange(1, 6):
            m.transfertraining(numlayers, windowhalfwidth, dataset, runno=runno, epochsperstep=2, batch_size=512, 
                N_classes=10, nummodels=1, gamma_pos='out', trainlayervector=None, useerrcorr=False, useimmprevoptim=False, 
                selectedloss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam')

# transfer training KMNIST halfwidth = 8 with 5 epochs per percent instead of 2 ############################

numlayers = 2
windowhalfwidth = 8
dataset = 'kmnist'

for runno in np.arange(1, 6):
    m.transfertraining(numlayers, windowhalfwidth, dataset, runno=runno, epochsperstep=5, batch_size=512, 
        N_classes=10, nummodels=1, gamma_pos='out', trainlayervector=None, useerrcorr=False, useimmprevoptim=False, 
        selectedloss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam')

# generating the accuracies of uncorrected and error-corrected ideal models that are programmed into a faulty mesh ######

numlayers = 2
windowhalfwidths = [8, 10]
datasets = ['mnist', 'fashion_mnist', 'kmnist']

for dataset in datasets:
    for windowhalfwidth in windowhalfwidths:
        m.uncorrcorr3mzi(dataset, windowhalfwidth=windowhalfwidth, numruns=5, numlayers=2)

# measuring accuracy of trained models on lossy meshes ################################################################

numlayers = 2
windowhalfwidth = 8
dataset = 'kmnist'

splitting_errors = [0, 0.1]
runnos = [4, 1]
multiplierarray = np.array([0.25, 0.5, 0.75, 1]) 
heaterlengthfracarray = np.array([0.5]) 

for i in np.arange(2):
    for j in np.arange(np.size(multiplierarray)):
        for k in np.arange(np.size(heaterlengthfracarray)):
            lossymeshaccuracies(numlayers, windowhalfwidth, dataset,
                                splitting_error=splitting_errors[i], 
                                runno=runnos[i], numtrials=10, 
                                heaterlengthfrac=heaterlengthfracarray[k],
                                multiplier=multiplierarray[j],
                                gamma_pos='out', trainlayervector=None)
            if i==1:
                lossymeshaccuracies(numlayers, windowhalfwidth, dataset,
                                    splitting_error=splitting_errors[i], 
                                    runno=runnos[i], numtrials=10, 
                                    heaterlengthfrac=heaterlengthfracarray[k],
                                    multiplier=multiplierarray[j],
                                    randomalphabetas=True,
                                    gamma_pos='out', trainlayervector=None)

# plotting ###################################################################################################

# Fig 3 ######################################################################################################

bottom = {'mnist':90, 'kmnist':78, 'fashion':80} # yaxis limits
top = {'mnist':100, 'kmnist':92, 'fashion':90}

numinputs = [256, 400] # number of input features
dataname = ['mnist', 'fashion', 'kmnist']

lasterror = 1/(2*np.sqrt(2))
percentjump = 1 
splitting_errors = np.array([i/100 for i in np.arange(1, 36, percentjump)]+[lasterror])

xaxis = np.array([0]+(100*splitting_errors).tolist())
xaxisdir = np.array([0]+(100*splitting_errors[:-1]).tolist())
alpha = 0.25

fig, axs = plt.subplots(2, 3, sharex=True)
fig.set_size_inches(15, 8)

takemaxadia = True # if True, for each error level in trasfer training, plot the highest accuracy found at or above that error level 
takemaxdir = False # don't do the same for direct training

onepoint5 = True # flag that plots 3MZI results
adiaplot = True # flag that plots transfer training results

labels = []
labels.append('uncorrected')
labels.append('error-corrected')
if onepoint5: labels.append('3-MZI')
if adiaplot: labels.append('transfer training')
labels.append('direct training')

ncols = len(labels)
fontsize = 17

for i in np.arange(2):
    for j in np.arange(3):
        
        plotvars = m.extractquants(dataname[j], numinputs[i], takemaxadia, takemaxdir)
        if dataname[j] == 'mnist' and numinputs[i] == 256:
            print(plotvars.adiabatic[:, 0], plotvars.adiabatic.shape)
            print(plotvars.uncorr[0, :], plotvars.uncorr.shape)

        plt.rcParams['font.size'] = '13'
        
        count = 0
        axs[i, j].plot(xaxis, plotvars.uncorrmed, color='red', label=labels[count], linewidth=3)
        count += 1
        axs[i, j].fill_between(xaxis, plotvars.uncorrupp, plotvars.uncorrlow, facecolor='red', alpha=alpha,
                               label='_nolegend_')

        axs[i, j].plot(xaxis, plotvars.corrmed, color='green', label=labels[count], linewidth=3)
        count += 1
        axs[i, j].fill_between(xaxis, plotvars.corrupp, plotvars.corrlow, facecolor='green', alpha=alpha,
                               label='_nolegend_')
        
        if onepoint5:
            axs[i, j].plot(xaxis, plotvars.mzi3corrmed, color='orange', label=labels[count], linewidth=3)
            count += 1
            axs[i, j].fill_between(xaxis, plotvars.mzi3corrupp, plotvars.mzi3corrlow, facecolor='orange', alpha=alpha,
                                   label='_nolegend_')
        
        if adiaplot:
            axs[i, j].plot(xaxis, plotvars.adiamed, color='purple', label=labels[count], linewidth=3)
            count += 1
            axs[i, j].fill_between(xaxis, plotvars.adiaupp, plotvars.adialow, facecolor='purple', alpha=alpha,
                                   label='_nolegend_')

        axs[i, j].plot(xaxisdir, plotvars.dirmed, color='blue', label=labels[count], linewidth=3)
        axs[i, j].fill_between(xaxisdir, plotvars.dirupp, plotvars.dirlow, facecolor='blue', alpha=alpha,
                               label='_nolegend_')

        axs[i, j].vlines((np.sin(np.pi/4)/2)*100, bottom[dataname[j]], top[dataname[j]], linestyles='dashed', linewidth=3)
            
        axs[i, j].set_ylim(bottom=bottom[dataname[j]], top=top[dataname[j]])
        if j!=2:
            axs[i, j].set_yticks(np.arange(bottom[dataname[j]], top[dataname[j]]+1, step=1))
        else:
            axs[i, j].set_yticks(np.arange(bottom[dataname[j]], top[dataname[j]]+1, step=2))
        axs[i, j].set_xticks(np.arange(0, 37, step=5))
        axs[i, j].set_xlim(left=0, right=36)
        axs[i, j].tick_params(axis='both', labelsize=fontsize)
        axs[i, j].grid(axis='both')

        axs[i, j].set_title(f'{dataname[j]}, inputsize={numinputs[i]}', fontsize=fontsize)

fig.supxlabel('Beamsplitter error in % (error level percent)', fontsize=fontsize)
fig.supylabel('Test accuracy(%)', fontsize=fontsize)
fig.legend(labels, loc='upper center', bbox_to_anchor=(0.52, 1.06),
           ncol=ncols, prop={'size':fontsize})
plt.tight_layout()
plt.savefig('figures/allaccsforjourno.pdf', bbox_inches='tight')
fig.show()

# Fig 4 ##################################################################################################

bottom = {'mnist':90, 'kmnist':78, 'fashion':80}
top = {'mnist':100, 'kmnist':92, 'fashion':90}

lasterror = 1/(2*np.sqrt(2))
percentjump = 1 
splitting_errors = np.array([i/100 for i in np.arange(1, 36, percentjump)]+[lasterror])

xaxis = np.array([0]+(100*splitting_errors).tolist())
xaxisdir = np.array([0]+(100*splitting_errors[:-1]).tolist())
alpha = 0.25

numinputs = 256
inputsize = numinputs

dataset = 'kmnist'

# two subplots
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(20, 6)

spinelinewidth = 2.5
plotlinewidth = 3.5

for i in range(2):
    for axis in ['top','bottom','left','right']:
        axs[i].spines[axis].set_linewidth(spinelinewidth)

# 5epo vs 2epo
fontsize = 19

plotvars = extractquants(dataset, 256, True, False, get5epo=True)
        
axs[0].plot(xaxis, plotvars.uncorrmed, color='red', label='uncorr.', linewidth=plotlinewidth)
axs[0].fill_between(xaxis, plotvars.uncorrupp, plotvars.uncorrlow, facecolor='red', alpha=alpha,
                       label='_nolegend_')

axs[0].plot(xaxis, plotvars.corrmed, color='green', label='corr.', linewidth=plotlinewidth)
axs[0].fill_between(xaxis, plotvars.corrupp, plotvars.corrlow, facecolor='green', alpha=alpha,
                       label='_nolegend_')

axs[0].plot(xaxis, plotvars.mzi3corrmed, color='orange', label='3-MZI', linewidth=plotlinewidth)
axs[0].fill_between(xaxis, plotvars.mzi3corrupp, plotvars.mzi3corrlow, facecolor='orange', alpha=alpha,
                       label='_nolegend_')


axs[0].plot(xaxis, plotvars.adiamed, color='purple', label='trans. 2 epo', linewidth=plotlinewidth)
axs[0].fill_between(xaxis, plotvars.adiaupp, plotvars.adialow, facecolor='purple', alpha=alpha,
                       label='_nolegend_')

axs[0].plot(xaxisdir, plotvars.dirmed, color='blue', label='direct', linewidth=plotlinewidth)
axs[0].fill_between(xaxisdir, plotvars.dirupp, plotvars.dirlow, facecolor='blue', alpha=alpha,
                       label='_nolegend_')

axs[0].plot(xaxis, plotvars.adia5med, color='magenta', label='trans. 5 epo', linewidth=plotlinewidth)
axs[0].fill_between(xaxis, plotvars.adia5upp, plotvars.adia5low, facecolor='magenta', alpha=alpha,
                       label='_nolegend_')

axs[0].vlines((np.sin(np.pi/4)/2)*100, bottom[dataset], top[dataset], linestyles='dashed', linewidth=plotlinewidth)

axs[0].set_ylim(bottom=bottom[dataset], top=top[dataset])
axs[0].set_yticks(np.arange(bottom[dataset], top[dataset]+1, step=2))
axs[0].set_xticks(np.arange(0, 37, step=5))
axs[0].set_xlim(left=0, right=36)
axs[0].tick_params(axis='both', labelsize=fontsize)
axs[0].set_xlabel('(a) Beamsplitter error in % (error level percent)', fontsize=fontsize)
axs[0].set_ylabel(dataset+' test accuracy(%)', fontsize=fontsize)
axs[0].grid(axis='both', linewidth=spinelinewidth)

axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.32), ncol=2, fontsize=fontsize)

# kmnist 256 ideal and 10% error as a function of unbalanced loss
upquant = 0.75
lowquant = 0.25

# ideal acc no loss
idealacc = plotvars.idealacc[0]

# reading ideal loss files
heaterlengthfracarray = np.array([0.5])
multiplierarray = np.array([0.25, 0.5, 0.75, 1])

splitting_error = 0.0
runno = 4
errindex = int(splitting_error*100)

foldername = 'models/'+dataset+f'/size{inputsize}/err{errindex}/run{runno}'

ideallossaccs = np.zeros((4, 10))
ideallossbars = np.zeros((2, 4))
for i in range(1):
    for j in range(4):
        with open(foldername+
                  f'/lossyacc_heater{heaterlengthfracarray[i]}_multiplier{multiplierarray[j]}.pickle',
                  'rb') as outputfile:
            lossesaccsdump = pickle.load(outputfile)
        ideallossaccs[j, :] = lossesaccsdump['lossyacc']
ideallossmed = np.median(ideallossaccs, axis=1)
ideallossupp = np.quantile(ideallossaccs, upquant, axis=1)
ideallosslow = np.quantile(ideallossaccs, lowquant, axis=1)
ideallossbars = np.vstack((ideallossmed-ideallosslow, ideallossupp-ideallossmed))

# 10% acc no loss
splitting_error = 0.1
runno = 1
errindex = int(splitting_error*100)

foldername = 'models/'+dataset+f'/size{inputsize}/err{errindex}/run{runno}'

with open(foldername+'/idealhistories.pickle', 'rb') as outputfile:
    idealhistories = pickle.load(outputfile)
acc10 = idealhistories[-1] 

# reading 10% error loss files
error10lossaccs = np.zeros((4, 10))
error10lossbars = np.zeros((2, 4))
for i in range(1):
    for j in range(4):
        with open(foldername+
                  f'/lossyacc_heater{heaterlengthfracarray[i]}_multiplier{multiplierarray[j]}.pickle',
                  'rb') as outputfile:
            lossesaccsdump = pickle.load(outputfile)
        error10lossaccs[j, :] = lossesaccsdump['lossyacc']
error10lossmed = np.median(error10lossaccs, axis=1)
error10lossupp = np.quantile(error10lossaccs, upquant, axis=1)
error10losslow = np.quantile(error10lossaccs, lowquant, axis=1)
error10lossbars = np.vstack((error10lossmed-error10losslow, error10lossupp-error10lossmed))

# porting the 10% max faulty matrix into a sample mesh with random faults ###################################

acc10random = np.array([89.18])

# 10% random mesh with loss
foldername = 'models/'+dataset+f'/size{inputsize}/err{errindex}/run{runno}'
error10randomlossaccs = np.zeros((4, 10))
error10randomlossbars = np.zeros((2, 4))
for i in range(1):
    for j in range(4):
        with open(foldername+
                  f'/randomalphabetaslossyacc_heater{heaterlengthfracarray[i]}_multiplier{multiplierarray[j]}.pickle',
                  'rb') as outputfile:
            lossesaccsdump = pickle.load(outputfile)
        error10randomlossaccs[j, :] = lossesaccsdump['lossyacc']
error10randomlossmed = np.median(error10randomlossaccs, axis=1)
error10randomlossupp = np.quantile(error10randomlossaccs, upquant, axis=1)
error10randomlosslow = np.quantile(error10randomlossaccs, lowquant, axis=1)
error10randomlossbars = np.vstack((error10randomlossmed-error10randomlosslow, error10randomlossupp-error10randomlossmed))

# plotting ####################################################################################################################
thirdmarker = 'D'
heaterbasemeanloss = 0.084
splitterbasemeanloss = 0.021

multiplierarray = np.array([[0, 0.25, 0.5, 0.75, 1]])
heaterlengthfracarray = np.array([[0.5]])
lossaxis = 2*multiplierarray*(heaterbasemeanloss*heaterlengthfracarray + splitterbasemeanloss)

axs[1].scatter(np.array([0]), np.array(idealacc[3]), c='green', s=170, marker='o')
axs[1].scatter(np.array([0]), acc10, c='blue', s=170, marker='s')
axs[1].scatter(np.array([0]), acc10random, c='red', s=170, marker=thirdmarker)

axs[1].errorbar(lossaxis[0, 1:], ideallossmed, yerr=ideallossbars, ecolor='green',
                linewidth=plotlinewidth,
                ls='none', marker='o', markersize=14, color='green', capsize=7, label='ideal')
axs[1].errorbar(lossaxis[0, 1:], error10lossmed, yerr=error10lossbars, ecolor='blue',
                linewidth=plotlinewidth,
                ls='none', marker='s', markersize=14, color='blue', capsize=7,
                label='10% max error-tolerant MZI mesh')
axs[1].errorbar(lossaxis[0, 1:], error10randomlossmed, yerr=error10randomlossbars,
                ecolor='red', linewidth=plotlinewidth,
                ls='none', marker=thirdmarker, markersize=14, color='red', capsize=7,
                label='10% random error MZI mesh')

axs[1].tick_params(axis='both', labelsize=fontsize)

axs[1].set_xlabel('(b) Mean MZI loss in dB', fontsize=fontsize)
axs[1].set_ylabel('KMNIST test accuracy(%)', fontsize=fontsize)
axs[1].set_ylim(bottom=89.05, top=89.3)
axs[1].grid(axis='both', linewidth=spinelinewidth)

axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.32), ncol=1, fontsize=fontsize)

plt.savefig('figures/trans5loss.pdf', bbox_inches='tight')
fig.show()

## comparison of linear classification on raw and fourier-transformed features ###################################################
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, InputLayer, Flatten, Dropout, Softmax

numlayers = 2
windowhalfwidth = 8
inputsize = (2*windowhalfwidth)**2
units = inputsize; N = inputsize; num_layers_withinlayer = inputsize
newinputsize = inputsize

epochs = 50
batch_size = 512
N_classes = 10
nummodels = 1

mnist_dp = h.MNISTDataProcessor(dataset='fashion_mnist')
gamma_pos = 'out'

xtrain = np.reshape(mnist_dp.x_train_raw, (mnist_dp.x_train_raw.shape[0],
                                           mnist_dp.x_train_raw.shape[1]*mnist_dp.x_train_raw.shape[2]), order='C')
ytrain = mnist_dp.y_train
ytrain_onehot = np.eye(10)[ytrain]

xtest = np.reshape(mnist_dp.x_test_raw, (mnist_dp.x_test_raw.shape[0],
                                         mnist_dp.x_test_raw.shape[1]*mnist_dp.x_test_raw.shape[2]), order='C')
ytest = mnist_dp.y_test
ytest_onehot = np.eye(10)[ytest]

# Fourier transformed data
data = mnist_dp.fourier(windowhalfwidth)
num_test = len(data.y_test_ind)
newdata = data

## one-layer network on raw data
Lin = Sequential([
    InputLayer(input_shape = xtrain[0].shape),
    Dense(10, activation = 'softmax')
    ])
Lin.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
                      metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])
Lin.fit(xtrain, ytrain, batch_size = 600, 
            epochs = 50, validation_data = (xtest, ytest))

predlabels = np.argmax(Lin.predict(xtest), axis=1)
print(f"acc = ({np.mean(predlabels==ytest)})")
Lin.save(f"models/FMNISTLinClass")

## one-layer network on Fourier transformed data
realizedxtrain = np.concatenate((np.real(data.x_train), np.imag(data.x_train)), axis=1)
realizedxtest = np.concatenate((np.real(data.x_test), np.imag(data.x_test)), axis=1)
FourierLin = Sequential([
    InputLayer(input_shape = (2*inputsize,)),
    Dense(10, activation = 'softmax')
    ])
FourierLin.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
                      metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])
FourierLin.fit(realizedxtrain, data.y_train_ind, batch_size = 600, 
            epochs = 50, validation_data = (realizedxtest, data.y_test_ind))

predlabels = np.argmax(FourierLin.predict(realizedxtest), axis=1)
print(f"acc = ({np.mean(predlabels==data.y_test_ind)})")
FourierLin.save(f"models/FMNISTFourierLinClass")


