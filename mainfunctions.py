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

######################################################################################################################################################################

def directtraining(numlayers, windowhalfwidth, dataset, epochs=50, batch_size=512, N_classes=10, nummodels=1, 
                    gamma_pos='out', saveflag=True, saveflagcheckpoints=False, splitting_errors=35, 
                    selectedloss=tf.keras.losses.CategoricalCrossentropy(), numruns=5):
    '''
    Trains neurophox Clements maximally faulty mesh models for a given error level on the chosen dataset.

            Parameters:
                    numlayers (int): number of layers
                    windowhalfwidth (int): half the side of the square window used to lowpass filter the image FFT 
                    dataset (string): 'mnist' or 'fashion_mnist' or 'kmnist' 
                    epochs (int): number of training epochs, default 50
                    batch_size (int): training batch size, default 512
                    N_classes (int): number of output classes, default 10
                    nummodels (int): redundant variable, should be hardcoded to 1 to ensure smooth execution of other functions
                    gamma_pos (string): position of the Clements mesh phase screen, only 'out' is currently supported
                    saveflag (bool): Flag that, if true, results in the final model being saved 
                    saveflagcheckpoints (bool): Flag that, if true, results in the intermediate checkpoint models being saved
                    splitting_errors: beamsplitter error level in percent
                    selectedloss: loss function for training
                    numruns: number of independent models to be trained 

            Saves model for the i-th run in the folder dataset+f"{inputsize}"+f'/err{splitting_errors}/run{i+1}'. Returns nothing.
    '''

    # params ########################################################################

    inputsize = (2*windowhalfwidth)**2
    units = inputsize
    num_layers_withinlayer = inputsize
    N = inputsize
    newinputsize = inputsize

    # data initialization ###########################################################

    # use dataset = 'mnist', 'fashion_mnist', or 'kmnist' to access the corresponding datasets
    mnist_dp = h.MNISTDataProcessor(dataset=dataset)
    data = mnist_dp.fourier(windowhalfwidth)
    num_test = len(data.y_test_ind)

    if dataset == 'fashion_mnist':
        dataset = 'fashion'
    
    bs_sigma = 0.5 * np.arcsin(2 * splitting_errors)

    # setting mesh errors
    alphainput = np.ones((numlayers, newinputsize, newinputsize // 2)) * 2 * bs_sigma
    alphainput[:, 1::2, -1] = 0
    betainput = np.zeros((numlayers, newinputsize, newinputsize // 2))

    bs_error = [tuple([betainput[i], alphainput[i]]) for i in np.arange(numlayers)]

    for i in np.arange(numruns):
        foldername = "models/"+dataset+f"/size{inputsize}"+f'/err{splitting_errors}/run{i+1}'
        initaccuracies, idealaccuracies = h.train_network(inputsize, 
                                                     numlayers, 
                                                     gamma_pos,
                                                     nummodels, 
                                                     data, 
                                                     epochs, 
                                                     batch_size, 
                                                     foldername, 
                                                     saveflag,
                                                     saveflagcheckpoints,
                                                     selectedloss=selectedloss,
                                                     bs_error=bs_error)

#################################################################################################################################################################

def transfertraining(numlayers, windowhalfwidth, dataset, runno=1, epochsperstep=2, batch_size=512, N_classes=10, nummodels=1, 
                    gamma_pos='out', trainlayervector=None, useerrcorr=False, useimmprevoptim=False, 
                    selectedloss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam'):
    '''
    Trains neurophox Clements maximally faulty mesh models for a given error level on the chosen dataset.

            Parameters:
                    numlayers (int): number of layers
                    windowhalfwidth (int): half the side of the square window used to lowpass filter the image FFT 
                    dataset (string): 'mnist' or 'fashion_mnist' or 'kmnist' 
                    runno: run of the ideal mesh model that should be used as the starting point for transfer training
                    epochsperstep (int): number of retraining epochs when weights are transferred from one error level to the next, default 2
                    batch_size (int): training batch size, default 512
                    N_classes (int): number of output classes, default 10
                    nummodels (int): redundant variable, should be hardcoded to 1 to ensure smooth execution of other functions
                    gamma_pos (string): position of the Clements mesh phase screen, only 'out' is currently supported
                    trainlayervector: List of bools that indicates whether the corresponding layer should be trained. None by default. 
                    useerrcorr (bool): Flag that, if true, uses Bandyopadhyay-type error correction when weights are transferred from one error level to the next.
                                        False by default.
                    useimmprevoptim (bool): Flag that, if true, uses the optimizer (along with its state) of the preceding error error level for the next level.
                                        False by default. 
                    selectedloss: loss function for training
                    optimizer: optimizer to be used, 'adam' by default

            Saves models periodically in the folder dataset+f"{inputsize}"+f'/err{errinit}/run{runno}'+f'/adiabaticnoisymodel{splitting_errors[j]}'. 
            Saves accuracies in dataset+f"{inputsize}"+f'/err{errinit}/run{runno}'+'/adiabatichistories.pickle'
            Returns nothing.
    '''

    inputsize = (2*windowhalfwidth)**2
    units = inputsize
    num_layers_withinlayer = inputsize
    N = inputsize
    newinputsize = inputsize

    # data initialization ###########################################################

    # use dataset = 'mnist', 'fashion_mnist', or 'kmnist' to access the corresponding datasets
    mnist_dp = h.MNISTDataProcessor(dataset=dataset)
    data = mnist_dp.fourier(windowhalfwidth)
    num_test = len(data.y_test_ind)

    if dataset == 'fashion_mnist':
        dataset = 'fashion'

    # model parameter loading ########################################################
    errinit = 0
    foldername = "models/"+dataset+f"/size{inputsize}"+f'/err{errinit}/run{runno}'

    with open(foldername+f'/idealthetas.pickle', 'rb') as outputfile:
        thetasorig, phisorig, gammasorig = pickle.load(outputfile)

    thetas = copy.deepcopy(thetasorig)
    phis = copy.deepcopy(phisorig)
    gammas = copy.deepcopy(gammasorig)

    # max faulty mesh error levels from 1% to 35.36% ###################################### 
    lasterror = 1/(2*np.sqrt(2))
    percentjump = 1 
    splitting_errors = np.array([i/100 for i in np.arange(1, 36, percentjump)]+[lasterror])
    bs_sigma = 0.5 * np.arcsin(2 * splitting_errors)
    numerrors = len(splitting_errors)

    # training and results #################################################################

    imperfect_accuracies = np.zeros((numerrors,))
    modelhistories = []
    modelhistories.append(splitting_errors)

    newdata = data

    for j in np.arange(numerrors):
        print(j)

        # setting mesh errors
        alphainput = np.ones((numlayers, newinputsize, newinputsize // 2)) * 2 * bs_sigma[j]
        alphainput[:, 1::2, -1] = 0
        betainput = np.zeros((numlayers, newinputsize, newinputsize // 2))

        if useerrcorr:
            thetas, phis, gammas = h.error_correction(thetas, phis, gammas, alphainput, betainput, 
                                                      numlayers, inputsize, gamma_pos=gamma_pos)

        bs_error = [tuple([betainput[i], alphainput[i]]) for i in np.arange(numlayers)]
        
        # initialize faulty mesh with params thetas, phis, gammas
        noisymodel = h.const_onn_EO(newinputsize,
                                    theta_init=thetas, 
                                    phi_init=phis, 
                                    gamma_init=gammas,
                                    L=numlayers, 
                                    bs_error=bs_error, 
                                    gamma_pos=gamma_pos, 
                                    trainlayervectorin=trainlayervector)
        
        noisymodel.compile(optimizer=optimizer,
                             loss=selectedloss,
                             metrics=['accuracy'])

        # train for epochsperstep number of epochs
        history = noisymodel.fit(newdata.x_train,
                                 newdata.y_train,
                                 epochs=epochsperstep,
                                 batch_size=batch_size,
                                 validation_data=(newdata.x_test, newdata.y_test),
                                 verbose=2)

        # record test accuracy
        noisypredoutputs = np.argmax(noisymodel.predict(newdata.x_test), axis=1)
        imperfect_accuracies[j] = 100*(1 - np.count_nonzero(noisypredoutputs-newdata.y_test_ind)/num_test)

        # extract parameters of this slightly retrained model
        thetas, phis, gammas = h.param_extraction(noisymodel)

        if useerrcorr:
            thetas, phis, gammas = h.noisytoideal(thetas, phis, gammas, alphainput, betainput, 
                                                  numlayers, inputsize, gamma_pos=gamma_pos)

        if j%10==0:
            noisymodel.save(foldername+f'/adiabaticnoisymodel{splitting_errors[j]}')
        if useimmprevoptim:
            optimizer = noisymodel.optimizer

    # save training results
    modelhistories.append({'useerrcorr':useerrcorr, 'useimmprevoptim':useimmprevoptim})
    modelhistories.append(imperfect_accuracies)
    with open(foldername+'/adiabatichistories.pickle', 'wb') as outputfile:
        pickle.dump(modelhistories, outputfile)

#########################################################################################################################################################################

def extractthetas(foldername, filename, inputsize, numlayers=2, gamma_pos='out', matrices=False, inputismodel=False):
    if inputismodel:  
        model = tf.keras.models.load_model(foldername+'/'+filename)
    else:
        with open(foldername+'/'+filename, 'rb') as inputfile:
            thetas, phis, gammas = pickle.load(inputfile)
        model = h.const_onn_EO(inputsize,
                               theta_init=thetas, 
                               phi_init=phis, 
                               gamma_init=gammas,
                               L=numlayers,  
                               gamma_pos=gamma_pos)
    extparams = h.param_extraction(model, matrices=matrices)
    
    return extparams

def uncorrcorr3mzi(dataset, windowhalfwidth=8, numruns=5, numlayers=2):

    '''
    Plugs weights of ideal meshes into faulty meshes without and with the error-correction of Bandyopadhyay et al and measures the resultant test accuracies.
    Also programs the ideal matrices into faulty 3MZI meshes, with error-correction, and records the resultant test accuracy. 

            Parameters:
                    numlayers (int): number of layers, default 2
                    windowhalfwidth (int): half the side of the square window used to lowpass filter the image FFT, default 8 
                    dataset (string): 'mnist' or 'fashion_mnist' or 'kmnist' 
                    numruns: number of random meshes to generate for each error level

            Saves accuracies of uncorrected/corrected/3MZI faulty meshes in dataset+f"{inputsize}"+f'/err0/run{counter}'+'/greenredaccs.pickle'
    '''

    inputsize = (2*windowhalfwidth)**2
    N = inputsize
    start = time.time()

    # use dataset = 'mnist', 'fashion_mnist', or 'kmnist' to access the corresponding datasets
    mnist_dp = h.MNISTDataProcessor(dataset=dataset)
    data = mnist_dp.fourier(windowhalfwidth)

    if dataset == 'fashion_mnist':
        dataset = 'fashion'

    lasterror = 1/(2*np.sqrt(2))
    percentjump = 1 
    splitting_errors = np.array([i/100 for i in np.arange(1, 36, percentjump)]+[lasterror])

    x_nn = 0.5 * np.arcsin(2 * splitting_errors)

    for counter in range(1, 6):
        
        matrices = True
        foldername = "models/"+dataset+f"/size{inputsize}"+f'/err0/run{counter}'
        filename = 'idealthetas.pickle'

        extmatrices = extractthetas(foldername, filename, inputsize, matrices=matrices, inputismodel=False)

        ct = numruns
        x = x_nn
        acc = np.zeros([3, len(x), ct])
        nn = [ms.SymClementsNetwork(extmatrices[l], method='diag') for l in np.arange(numlayers)]

        errors_and_accs = []

        for (i, xi) in enumerate(x):
            for ct_i in range(ct):
                dpList = 2*np.random.rand(2, N*(N-1)//2, 3)-1
        #         errors_and_accs[-1].append(dpList)
                acc[:, i, ct_i] = ec.run_stuff(nn, data.x_test, data.y_test_ind, dpList*xi)
                print (f"N={N}, acc={acc[2,i,ct_i]}, inst={ct_i}, sig={xi:.3f}, time={(time.time()-start)/60}")

        errors_and_accs.append(acc)

        with open(foldername+'/greenredaccs.pickle', 'wb') as outputfile:
            pickle.dump(errors_and_accs, outputfile)

###########################################################################################################################################################################

def lossymeshaccuracies(numlayers, windowhalfwidth, dataset, splitting_error=0.1, runno=1, numtrials=10, 
                    MZIbasemeanloss=0.02, MZIbasestddev=0.0016, multiplier=1, heaterfrac=0.8, 
                    gamma_pos='out', trainlayervector=None):

    '''
    Plugs trained neurophox Clements lossless maximally faulty mesh phases into lossy meshes and measures the test accuracy.

            Parameters:
                    numlayers (int): number of layers
                    windowhalfwidth (int): half the side of the square window used to lowpass filter the image FFT 
                    dataset (string): 'mnist' or 'fashion_mnist' or 'kmnist' 
                    splitting_error: the error of the lossless mesh model that is being used
                    runno: run of the lossless mesh model that should be plugged into the lossy mesh
                    numtrials: number of random lossy meshes over which the accuracy will be averaged
                    MZIbasemeanloss: minimum mean loss per MZI in dB
                    MZIbasestddev: minimum loss standard deviation per MZI in dB
                    multiplier (int): multiple of the minimum loss that should be used in the instantiated mesh,
                                      mean loss per MZI will be multiplier * MZIbasemeanloss, stddev will be sqrt(multiplier) * MZIbasestddev  
                    heaterfrac: fraction of the loss that occurs in the center part of the MZI
                    gamma_pos (string): position of the Clements mesh phase screen, only 'out' is currently supported
                    trainlayervector: List of bools that indicates whether the corresponding layer should be trained. None by default. 

            Saves accuracies in dataset+f"{inputsize}"+f'/err{errindex}/run{runno}'+f'/lossyacc{meanlossperMZI}.pickle' where errindex = int(splitting_error*100)
            Returns nothing.
    '''

    inputsize = (2*windowhalfwidth)**2
    units = inputsize
    num_layers_withinlayer = inputsize
    N = inputsize
    newinputsize = inputsize

    mnist_dp = h.MNISTDataProcessor(dataset=dataset)
    data = mnist_dp.fourier(windowhalfwidth)
    num_test = len(data.y_test_ind)

    if dataset == 'fashion_mnist':
        dataset = 'fashion'

    ###########################################################################################################

    # for splitting_error = 0.0, use runno = 4
    # for splitting_error = 0.1, use runno = 1

    errindex = int(splitting_error*100)

    foldername = 'models/'+dataset+f"/size{inputsize}"+f'/err{errindex}/run{runno}'
    filename = 'idealthetas.pickle'

    sourcemodelpath = foldername+'/'+filename

    with open(sourcemodelpath, 'rb') as outputfile:
        thetasorig, phisorig, gammasorig = pickle.load(outputfile)

    thetas = copy.deepcopy(thetasorig)
    phis = copy.deepcopy(phisorig)
    gammas = copy.deepcopy(gammasorig)

    ## input loss in dB
    meanlossperMZI = MZIbasemeanloss*multiplier
    standevperMZI = MZIbasestddev*np.sqrt(multiplier)

    ## splitter error ##########################################################################

    bs_sigma = 0.5 * np.arcsin(2 * splitting_error)
    alphainput = np.ones((numlayers, newinputsize, newinputsize // 2)) * 2 * bs_sigma
    alphainput[:, 1::2, -1] = 0
    betainput = np.zeros((numlayers, newinputsize, newinputsize // 2))

    bs_error = [tuple([betainput[i], alphainput[i]]) for i in np.arange(numlayers)]

    ###########################################################################################################
    heatermean = heaterfrac*meanlossperMZI
    heaterstandev = np.sqrt(heaterfrac)*standevperMZI
    periphermean = (1-heaterfrac)*meanlossperMZI/2
    peripherstandev = np.sqrt((1-heaterfrac)/2)*standevperMZI

    lossesaccsdump = {}
    lossesaccsdump['sourcemodelpath'] = sourcemodelpath
    lossesaccsdump['numtrials'] = numtrials 
    lossesaccsdump['N'] = N
    lossesaccsdump['meanlossperMZI'] = meanlossperMZI
    lossesaccsdump['standevperMZI'] = standevperMZI
    lossesaccsdump['heaterfrac'] = heaterfrac
    lossesaccsdump['heatermean'] = heatermean
    lossesaccsdump['heaterstandev'] = heaterstandev
    lossesaccsdump['periphermean'] = periphermean
    lossesaccsdump['peripherstandev'] = peripherstandev

    lossyacc = np.zeros(numtrials)

    start = time.time()

    for itr in np.arange(numtrials):
        targetloss = np.zeros((numlayers, N, N//2, 3, 2))

        targetloss[:, :, :, 0, :] = np.random.normal(loc=periphermean, scale=peripherstandev,
                                                     size=(numlayers, N, N//2, 2))
        while np.any(targetloss[:, :, :, 0, :]<0):
            (targetloss[:, :, :, 0, :])[targetloss[:, :, :, 0, :]<0] =\
            np.random.normal(loc=periphermean, scale=peripherstandev,
                                                         size=(numlayers, N, N//2, 2))[targetloss[:, :, :, 0, :]<0]
            
        targetloss[:, :, :, 2, :] = np.random.normal(loc=periphermean, scale=peripherstandev,
                                                     size=(numlayers, N, N//2, 2))
        while np.any(targetloss[:, :, :, 2, :]<0):
            (targetloss[:, :, :, 2, :])[targetloss[:, :, :, 2, :]<0] =\
            np.random.normal(loc=periphermean, scale=peripherstandev,
                                                         size=(numlayers, N, N//2, 2))[targetloss[:, :, :, 2, :]<0]
            
        targetloss[:, :, :, 1, :] = np.random.normal(loc=heatermean, scale=heaterstandev,
                                                     size=(numlayers, N, N//2, 2)) 
        while np.any(targetloss[:, :, :, 1, :]<0):
            (targetloss[:, :, :, 1, :])[targetloss[:, :, :, 1, :]<0] =\
            np.random.normal(loc=heatermean, scale=heaterstandev,
                                                         size=(numlayers, N, N//2, 2))[targetloss[:, :, :, 1, :]<0]
        
        lossesaccsdump[f'targetloss{itr}'] = targetloss

        newdata = data
        wvgloss = targetloss

        noisymodel = h.const_onn_EO(newinputsize,
                                    theta_init=thetas, 
                                    phi_init=phis, 
                                    gamma_init=gammas,
                                    L=numlayers, 
                                    wvgloss=wvgloss,
                                    bs_error=bs_error,
                                    gamma_pos=gamma_pos, 
                                    trainlayervectorin=trainlayervector)

        netout = noisymodel.predict(newdata.x_test)
        noisypredoutputs = np.argmax(netout, axis=1)
        lossyacc[itr] = 100*(1 - np.count_nonzero(noisypredoutputs-newdata.y_test_ind)/num_test)
        print(f'iter {itr}, acc is {lossyacc[itr]}, time = {(time.time()-start)/60}')
            
    lossesaccsdump['lossyacc'] = lossyacc

    with open(foldername+f'/lossyacc{meanlossperMZI}.pickle', 'wb') as outputfile:
        pickle.dump(lossesaccsdump, outputfile)

###########################################################################################################################################################################

# This class has as attributes all the quantities that need to be plotted 
class plotquants:
    def __init__(self, idealacc, uncorr, uncorrmed, uncorrupp, uncorrlow, 
                 corr, corrmed, corrupp, corrlow,
                 mzi3corr, mzi3corrmed, mzi3corrupp, mzi3corrlow, 
                 adiabatic, adiamed, adiaupp, adialow, 
                 diracc, dirmed, dirupp, dirlow,
                 get5epo=False, adiabatic5=0.0, adia5med=0.0, adia5upp=0.0, adia5low=0.0):
        self.idealacc = idealacc
        self.uncorr = uncorr
        self.uncorrmed = uncorrmed
        self.uncorrupp = uncorrupp
        self.uncorrlow = uncorrlow
        self.corr = corr
        self.corrmed = corrmed
        self.corrupp = corrupp
        self.corrlow = corrlow
        self.mzi3corr = mzi3corr
        self.mzi3corrmed = mzi3corrmed
        self.mzi3corrupp = mzi3corrupp
        self.mzi3corrlow = mzi3corrlow
        self.adiabatic = adiabatic
        self.adiamed = adiamed
        self.adiaupp = adiaupp
        self.adialow = adialow
        self.diracc = diracc
        self.dirmed = dirmed
        self.dirupp = dirupp
        self.dirlow = dirlow
        self.get5epo = get5epo
        if get5epo:
            self.adiabatic5 = adiabatic5
            self.adia5med = adia5med
            self.adia5upp = adia5upp
            self.adia5low = adia5low

# extracting quantities to be plotted from pickle files ########################################################
def extractquants(dataname, numinputs, takemaxadia=True, takemaxdir=False, get5epo=False):
    '''
    Extracts test accuracies from model history pickle files for various model training runs 
    on a given dataset and returns a plotquants object that contains median and quantiles of the accuracies.  

            Parameters:
                    dataname (str): name of the dataset
                    numinputs: number of input features
                    takemaxadia (bool): flag that sets the test accuracy of a faulty transfer-trained model 
                                        with a given beamsplitter error to that of the best model 
                                        with at least that much beamsplitter error in that run.
                                        True by default.   
                    takemaxdir (bool): flag that sets the test accuracy of a faulty direct-trained model 
                                       with a given beamsplitter error to that of the best model 
                                       with at least that much beamsplitter error in that run.
                                       False by default.
                    get5epo (bool): if true, the function also extracts the performance of 
                                    5 epochs per percent transfer-trained models. False by default.

            Returns plotquants object.
    '''

    upquant = 0.75; lowquant = 0.25

    idealacc = []
    acc = []

    errinit = 0 

    for counter in range(1, 6):

        foldername = 'models/'+dataname+f'/size{inputsize}/err{errinit}/run{counter}'

        with open(foldername+'/greenredaccs.pickle', 'rb') as outputfile:
            greenredaccs = pickle.load(outputfile)
        acc.append(greenredaccs[-1])

        with open(foldername+'/idealhistories.pickle', 'rb') as outputfile:
            idealhistories = pickle.load(outputfile)
        idealacc.append(idealhistories[-1])

    idealaccrep = [np.tile(idealacc[i], (1, 5)) for i in range(5)]
    idealacc = (np.array(idealacc)).T
    
    uncorr = [np.concatenate((idealaccrep[i], acc[i][0]*100), axis=0) for i in range(5)]
    uncorr = np.concatenate(uncorr, axis=1)
    
    corr = [np.concatenate((idealaccrep[i], acc[i][1]*100), axis=0) for i in range(5)]
    corr = np.concatenate(corr, axis=1)
    
    mzi3corr = [np.concatenate((idealaccrep[i], acc[i][2]*100), axis=0) for i in range(5)]
    mzi3corr = np.concatenate(mzi3corr, axis=1)

    uncorrmed = np.median(uncorr, axis=1)
    uncorrupp = np.quantile(uncorr, upquant, axis=1)
    uncorrlow = np.quantile(uncorr, lowquant, axis=1)

    corrmed = np.median(corr, axis=1)
    corrupp = np.quantile(corr, upquant, axis=1)
    corrlow = np.quantile(corr, lowquant, axis=1)
    
    mzi3corrmed = np.median(mzi3corr, axis=1)
    mzi3corrupp = np.quantile(mzi3corr, upquant, axis=1)
    mzi3corrlow = np.quantile(mzi3corr, lowquant, axis=1)
    
    adiabatic = np.zeros((5, 37))

    for i in np.arange(5):
        foldername = 'models/'+dataname+f'/size{inputsize}/err{errinit}/run{i+1}'
        with open(foldername+'/adiabatichistories.pickle', 'rb') as outputfile:
            exthistories = pickle.load(outputfile)

        adiabatic[i, :] = np.concatenate((np.array([idealacc[0, i]]), exthistories[-1]))

        if takemaxadia:
            adiabatic[i, :] = np.array([np.max(adiabatic[i, j:]) for j in np.arange(adiabatic[i, :].size)])

    adiamed = np.median(adiabatic, axis=0)
    adiaupp = np.quantile(adiabatic, upquant, axis=0)
    adialow = np.quantile(adiabatic, lowquant, axis=0)
    
    diracc = np.zeros((5, 36))
    diracc[:, 0] = idealacc[0]

    for i in np.arange(5):
        for j in np.arange(35):
            foldername = 'models/'+dataname+f'/size{inputsize}/err{j+1}/run{i+1}'
            with open(foldername+'/idealhistories.pickle', 'rb') as outputfile:
                exthistories = pickle.load(outputfile)

            diracc[i, j+1] = exthistories[-1][0]

        if takemaxdir:
            diracc[i, :] = np.array([np.max(diracc[i, k:]) for k in np.arange(diracc[i, :].size)])

    dirmed = np.median(diracc, axis=0)
    dirupp = np.quantile(diracc, upquant, axis=0)
    dirlow = np.quantile(diracc, lowquant, axis=0)
    
    if get5epo:
        adiabatic5 = np.zeros((5, 37))

        for i in np.arange(5):
            foldername = 'models/'+dataname+f'/size{inputsize}/err{errinit}/run{i+1}'
            with open(foldername+'/adiabatichistories5epo.pickle', 'rb') as outputfile:
                exthistories = pickle.load(outputfile)

            adiabatic5[i, :] = np.concatenate((np.array([idealacc[0, i]]), exthistories[-1]))

            if takemaxadia:
                adiabatic5[i, :] = np.array([np.max(adiabatic5[i, j:]) for j in np.arange(adiabatic5[i, :].size)])

        adia5med = np.median(adiabatic5, axis=0)
        adia5upp = np.quantile(adiabatic5, upquant, axis=0)
        adia5low = np.quantile(adiabatic5, lowquant, axis=0)
        
    if get5epo:
        plotvars = plotquants(idealacc, uncorr, uncorrmed, uncorrupp, uncorrlow, 
                              corr, corrmed, corrupp, corrlow,
                              mzi3corr, mzi3corrmed, mzi3corrupp, mzi3corrlow,  
                              adiabatic, adiamed, adiaupp, adialow, 
                              diracc, dirmed, dirupp, dirlow,
                              get5epo, adiabatic5, adia5med, adia5upp, adia5low)
    else:
        plotvars = plotquants(idealacc, uncorr, uncorrmed, uncorrupp, uncorrlow, 
                              corr, corrmed, corrupp, corrlow,
                              mzi3corr, mzi3corrmed, mzi3corrupp, mzi3corrlow,  
                              adiabatic, adiamed, adiaupp, adialow, 
                              diracc, dirmed, dirupp, dirlow)
    
    return plotvars

