import numpy as np
import scipy as sp
# import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Lambda, Dense
from tensorflow.python.keras import backend as K
from tensorflow.keras.datasets import mnist, fashion_mnist
from extra_keras_datasets import kmnist
from tensorflow.keras.optimizers import Adam

import neurophox
from neurophox.tensorflow import RM, SVD, BM
from neurophox.ml.nonlinearities import cnormsq

#import seaborn as sns
from collections import namedtuple
import time
import pickle

#import cupy as cp

from neurophox.tensorflow.generic import MeshPhasesTensorflow
import matplotlib.pyplot as plt
import copy
#import jax.numpy as jnp

##########################################################################################################################

class EOIntensityModulation(tf.keras.layers.Layer):
    def __init__(self,
                 N,
                 alpha=0.1,
                 g=np.pi,
                 phi_b=np.pi,
                 train_alpha=False,
                 train_g=False,
                 train_phi_b=False,
                 single_param_per_layer=True):
        super(EOIntensityModulation, self).__init__()
        
        if single_param_per_layer:
            var_shape = [1]
        else:
            var_shape = [N]
        
        self.g     = self.add_variable(shape=var_shape,
                                       name="g",
                                       initializer=tf.constant_initializer(g),
                                       trainable=train_g,
                                       constraint=lambda x: tf.clip_by_value(x, 1e-3, 1.5*np.pi))
        self.phi_b = self.add_variable(shape=var_shape,
                                       name="phi_b",
                                       initializer=tf.constant_initializer(phi_b),
                                       trainable=train_phi_b,
                                       constraint=lambda x: tf.clip_by_value(x, -np.pi, +np.pi))
        self.alpha = self.add_variable(shape=var_shape,
                                       name="alpha",
                                       initializer=tf.constant_initializer(alpha),
                                       trainable=train_alpha,
                                       constraint=lambda x: tf.clip_by_value(x, 0.01, 0.99))
    
    def call(self, inputs):
        alpha, g, phi_b = tf.complex(self.alpha, 0.0), tf.complex(self.g, 0.0), tf.complex(self.phi_b, 0.0)
        Z = inputs
        return 1j * tf.sqrt(1-alpha) * tf.exp(-1j*0.5*g*tf.math.conj(Z)*Z - 1j*0.5*phi_b) *\
    tf.cos(0.5*g*tf.math.conj(Z)*Z + 0.5*phi_b) * Z
    
#def nonlin(Z, alpha=0.1, g=0.05*np.pi, phi_b=1*np.pi):
#    alpha, g, phi_b = complex(alpha, 0.0), complex(g, 0.0), complex(phi_b, 0.0)
#    return 1j * cp.sqrt(1-alpha) * cp.exp(-1j*0.5*g*cp.conj(Z)*Z - 1j*0.5*phi_b) *\
#           cp.cos(0.5*g*cp.conj(Z)*Z + 0.5*phi_b) * Z
    
def nonlin(Z, alpha=0.1, g=0.05*np.pi, phi_b=1*np.pi):
    alpha, g, phi_b = complex(alpha, 0.0), complex(g, 0.0), complex(phi_b, 0.0)
    return 1j * np.sqrt(1-alpha) * np.exp(-1j*0.5*g*np.conj(Z)*Z - 1j*0.5*phi_b) *\
           np.cos(0.5*g*np.conj(Z)*Z + 0.5*phi_b) * Z
    
#def nonlinjax(Z, alpha=0.1, g=0.05*jnp.pi, phi_b=1*jnp.pi):
#    alpha, g, phi_b = complex(alpha, 0.0), complex(g, 0.0), complex(phi_b, 0.0)
#    return 1j * jnp.sqrt(1-alpha) * jnp.exp(-1j*0.5*g*jnp.conj(Z)*Z - 1j*0.5*phi_b) *\
#           jnp.cos(0.5*g*jnp.conj(Z)*Z + 0.5*phi_b) * Z
    
#################################################################################################################################    
    
def const_onn_EO(N,
                 N_classes=10,
                 L=2,
                 butterclements = 'clements',
                 train_alpha=False,
                 train_g=False,
                 train_phi_b=False,
                 single_param_per_layer=True,
                 theta_init='haar_rect',
                 phi_init='random_phi',
                 gamma_init='random_gamma',
                 gamma_pos=None,
                 trainlayervectorin=None,
                 bs_error=0.0,
                 wvgloss=0.0,
                 e_l=0.0,
                 e_r=0.0,
                 alpha=0.1,
                 g=0.05*np.pi,
                 phi_b=1*np.pi, 
                 singlelayerlin: bool=False,
                 disptrainflag: bool=False):
    alpha = np.asarray(alpha)
    g     = np.asarray(g)
    phi_b = np.asarray(phi_b)
    
    if trainlayervectorin == None:
        trainlayervector = [True for i in np.arange(L)]
    else:
        trainlayervector = copy.deepcopy(trainlayervectorin)
    
    
    if alpha.size == 1:
        alpha = np.tile(alpha, L)
    else:
        assert alpha.size == L, 'alpha has a size which is inconsistent with L'
    
    if g.size == 1:
        g = np.tile(g, L)
    else:
        assert g.size == L, 'g has a size which is inconsistent with L'
    
    if phi_b.size == 1:
        phi_b = np.tile(phi_b, L)
    else:
        assert phi_b.size == L, 'phi_b has a size which is inconsistent with L'
    
    layers=[]
    #, trainable=trainlayervector[i]
    for i in range(L):
        
        if disptrainflag:
            print(f'{i}th layer flag is {trainlayervector[i]}')
        
        if isinstance(wvgloss, np.ndarray):
            wvglossin = wvgloss[i]
        else:
            wvglossin = wvgloss
            
        if butterclements == 'clements':
        
            if isinstance(theta_init, np.ndarray) and\
            isinstance(phi_init, np.ndarray) and isinstance(gamma_init, np.ndarray):
                if isinstance(e_l, np.ndarray) and isinstance(e_r, np.ndarray):
                    layers.append(neurophox.tensorflow.RM(N, theta_init=theta_init[i], phi_init=phi_init[i],
                                     gamma_init=gamma_init[i], gamma_pos=gamma_pos, e_l=e_l[i], e_r=e_r[i], basis='sm', trainable=trainlayervector[i]))
                elif isinstance(bs_error, list):
                    layers.append(RM(N, theta_init=theta_init[i], phi_init=phi_init[i],
                                     gamma_init=gamma_init[i], gamma_pos=gamma_pos,
                                     wvgloss=wvglossin, bs_error=bs_error[i], basis='sm', trainable=trainlayervector[i]))
                else:
                    layers.append(RM(N, theta_init=theta_init[i], phi_init=phi_init[i],
                                     gamma_init=gamma_init[i], gamma_pos=gamma_pos,
                                     wvgloss=wvglossin, bs_error=bs_error, basis='sm', trainable=trainlayervector[i]))
            else:
                if isinstance(bs_error, list):
                    layers.append(RM(N, theta_init=theta_init, phi_init=phi_init,
                                     gamma_init=gamma_init, gamma_pos=gamma_pos,
                                     wvgloss=wvglossin, bs_error=bs_error[i], basis='sm', trainable=trainlayervector[i]))
                else:
                    layers.append(RM(N, theta_init=theta_init, phi_init=phi_init,
                                     gamma_init=gamma_init, gamma_pos=gamma_pos,
                                     wvgloss=wvglossin, bs_error=bs_error, basis='sm', trainable=trainlayervector[i]))
                    
        elif butterclements == 'butterfly':
            layers.append(BM(N, theta_init=theta_init[i], phi_init=phi_init[i],
                         basis='sm'))
            
        else:
            raise ValueError('Wrong value for butterclements!')
                
        if not singlelayerlin:
            layers.append(EOIntensityModulation(N,
                                                alpha[i],
                                                g[i],
                                                phi_b[i],
                                                train_alpha=train_alpha,
                                                train_g=train_g,
                                                train_phi_b=train_phi_b,
                                                single_param_per_layer=single_param_per_layer))
    
    if not singlelayerlin:
        layers.append(Activation(cnormsq))
        layers.append(Lambda(lambda x: tf.math.real(x[:, :N_classes])))
        layers.append(Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)))
    
    return tf.keras.models.Sequential(layers)

###############################################################################################################################
    
def norm_inputs(inputs, feature_axis=1):
    # if feature_axis == 1:
    #     n_features, n_examples = inputs.shape
    # elif feature_axis == 0:
    #     n_examples, n_features = inputs.shape
    # for i in range(n_features):
    #     l1_norm = np.mean(np.abs(inputs[i, :]))
    #     inputs[i, :] /= l1_norm

    inputs = inputs / np.sqrt(np.sum(np.abs(inputs)**2, axis=1))[:, None] * np.sqrt(80)
    return inputs

##################################################################################################################################
    
ONNData = namedtuple('ONNData', ['x_train', 'y_train', 'y_train_ind', 'x_test',\
                                 'y_test', 'y_test_ind', 'units', 'num_classes'])

class MNISTDataProcessor:
    def __init__(self, dataset: str='', whichpoints: str='', numpoints: int=0, inputdata=None):
        #match dataset:
        if dataset == 'mnist':
            if inputdata is None:
                (self.x_train_raw, self.y_train), (self.x_test_raw, self.y_test) = mnist.load_data()
            else:
                (self.x_train_raw, self.y_train) = inputdata
                _, (self.x_test_raw, self.y_test) = mnist.load_data()
        elif dataset == 'fashion_mnist':
            if inputdata is None:
                (self.x_train_raw, self.y_train), (self.x_test_raw, self.y_test) = fashion_mnist.load_data()
            else:
                (self.x_train_raw, self.y_train) = inputdata
                _, (self.x_test_raw, self.y_test) = fashion_mnist.load_data()
        elif dataset == 'kmnist':
            if inputdata is None:
                (self.x_train_raw, self.y_train), (self.x_test_raw, self.y_test) = kmnist.load_data()
            else:
                (self.x_train_raw, self.y_train) = inputdata
                _, (self.x_test_raw, self.y_test) = kmnist.load_data()
        else:
            raise ValueError('Invalid assignment to dataset')
        #(self.x_train_raw, self.y_train), (self.x_test_raw, self.y_test) =\
        #fashion_mnist.load_data() if fashion else mnist.load_data()
        print(f'xtrainshape={self.x_train_raw.shape}, xtestshape={self.x_test_raw.shape}, ytrainshape={self.y_train.shape}, ytestshape={self.y_test.shape}')
        self.num_train = self.x_train_raw.shape[0]
        self.num_test = self.x_test_raw.shape[0]
        self.whichpoints = whichpoints
        self.numpoints = numpoints
        
        if whichpoints == '': 
            self.x_train_ft = np.fft.fftshift(np.fft.fft2(self.x_train_raw), axes=(1, 2))
            self.x_test_ft = np.fft.fftshift(np.fft.fft2(self.x_test_raw), axes=(1, 2))
        else:
            self.meandigits = np.zeros((10, 28, 28))
            self.chosen_x_train = np.zeros((numpoints*10, 28, 28))
            self.chosen_y_train = np.zeros((numpoints*10,), dtype=np.uint8)
            for i in np.arange(10):
                indices = np.squeeze(np.argwhere(self.y_train==i))
                self.meandigits[i] = np.mean(self.x_train_raw[indices, :, :], axis=0)
                frodist = np.linalg.norm((self.x_train_raw[indices, :, :]-self.meandigits[i]), ord='fro', axis=(1, 2))
                #frodist = [np.linalg.norm((self.x_train_raw[indices, :, :]-self.meandigits[i])[j], ord='fro') 
                #           for j in np.arange(len(indices))]
                #pick out numpoints smallest distances, place them in the chosen_x_train array
                if whichpoints == 'nearest':
                    sortedindices = np.argsort(frodist)[:numpoints]
                elif whichpoints == 'farthest':
                    sortedindices = np.argsort(frodist)[-numpoints:]
                else:
                    raise ValueError('Invalid value for whichpoints')
                self.chosen_x_train[(i*numpoints):((i+1)*numpoints), :, :] = self.x_train_raw[indices[sortedindices], :, :]
                self.chosen_y_train[(i*numpoints):((i+1)*numpoints)] = self.y_train[indices[sortedindices]]
                #print(f'finished loop {i}')
            #shuffle the training set and labels
            self.permutindices = np.random.permutation(np.arange(numpoints*10))
            self.chosen_x_train[:] = self.chosen_x_train[self.permutindices, :, :] 
            self.chosen_y_train[:] = self.chosen_y_train[self.permutindices]
            
            self.num_train = int(10*numpoints)
            
            self.x_train_ft = np.fft.fftshift(np.fft.fft2(self.chosen_x_train), axes=(1, 2))
            self.x_test_ft = np.fft.fftshift(np.fft.fft2(self.x_test_raw), axes=(1, 2))
            
    def fourier(self, freq_radius):
        dimsize = self.x_train_raw.shape[1]
        
        if freq_radius < dimsize//2:
            min_r, max_r = dimsize//2 - freq_radius, dimsize//2 + freq_radius
            x_train_ft = self.x_train_ft[:, min_r:max_r, min_r:max_r]
            x_test_ft = self.x_test_ft[:, min_r:max_r, min_r:max_r]
        else:
            x_train_ft = self.x_train_ft
            x_test_ft = self.x_test_ft
        
        if self.whichpoints == '':
            return ONNData(
                x_train=norm_inputs(x_train_ft.reshape((self.num_train, -1))).astype(np.complex64),
                y_train=np.eye(10)[self.y_train],
                y_train_ind=self.y_train,
                x_test=norm_inputs(x_test_ft.reshape((self.num_test, -1))).astype(np.complex64),
                y_test=np.eye(10)[self.y_test],
                y_test_ind=self.y_test,
                units=(2 * freq_radius)**2,
                num_classes=10
            )
        else:
            return ONNData(
                x_train=norm_inputs(x_train_ft.reshape((self.num_train, -1))).astype(np.complex64),
                y_train=np.eye(10)[self.chosen_y_train],
                y_train_ind=self.chosen_y_train,
                x_test=norm_inputs(x_test_ft.reshape((self.num_test, -1))).astype(np.complex64),
                y_test=np.eye(10)[self.y_test],
                y_test_ind=self.y_test,
                units=(2 * freq_radius)**2,
                num_classes=10
            )    
    
    def resample(self, p, b=0):
        dimsize = self.x_train_raw.shape[1]
        m = dimsize - b * 2
        min_r, max_r = b, dimsize - b
        x_train_ft = sp.ndimage.zoom(self.x_train_raw[:, min_r:max_r, min_r:max_r], (1, p / m, p / m))
        x_test_ft = sp.ndimage.zoom(self.x_test_raw[:, min_r:max_r, min_r:max_r], (1, p / m, p / m))
        return ONNData(
            x_train=norm_inputs(x_train_ft.reshape((self.num_train, -1)).astype(np.complex64)),
            y_train=np.eye(10)[self.y_train],
            x_test=norm_inputs(x_test_ft.reshape((self.num_test, -1)).astype(np.complex64)),
            y_test=np.eye(10)[self.y_test],
            units=p ** 2,
            num_classes=10
        )
    
###########################################################################################################################
        
def param_extraction(model, matrices: bool=False):
    if not matrices:
        thetas_extracted = [v for v in model.variables if v.name == "theta:0"]
        thetas = np.array([K.eval(thetas_extracted[i]) for i in np.arange(len(thetas_extracted))])

        phis_extracted = [v for v in model.variables if v.name == "phi:0"]
        phis = np.array([K.eval(phis_extracted[i]) for i in np.arange(len(phis_extracted))])
        phis[:, 1::2, -1] = 0

        gammas_extracted = [v for v in model.variables if v.name == "gamma:0"]
        gammas = np.array([K.eval(gammas_extracted[i]) for i in np.arange(len(gammas_extracted))])

        return thetas, phis, gammas
    
    else:
        layernames = [layer.name for layer in model.layers]
        thetas_extracted = [v for v in model.variables if v.name == "theta:0"]
        thetas = K.eval(thetas_extracted[0])
        N = thetas.shape[0]
        inputmat = np.eye(N, dtype=np.complex64)
        numlayers = len(thetas_extracted)
        extmatrices = [model.get_layer(layernames[j]).transform(inputmat).numpy().T for j in np.arange(0, 2*numlayers, 2)]
        return extmatrices

#def param_extraction(model):
#    thetas_extracted = [v for v in model.variables if v.name == "theta:0"]
#    thetas = np.array([K.eval(thetas_extracted[0]), K.eval(thetas_extracted[1])])

#    phis_extracted = [v for v in model.variables if v.name == "phi:0"]
#    phis = np.array([K.eval(phis_extracted[0]), K.eval(phis_extracted[1])])
#    phis[:, 1::2, -1] = 0

#    gammas_extracted = [v for v in model.variables if v.name == "gamma:0"]
#    gammas = np.array([K.eval(gammas_extracted[0]), K.eval(gammas_extracted[1])])
    
#    return thetas, phis, gammas

##############################################################################################################################
    
def clements_evenroll(X, numlayers):
    return np.array([[np.roll(row, 0) if row_index%2==0 else np.roll(row, 1)\
                           for row,row_index in zip(X[layer], range(X[layer].shape[0]))]\
                           for layer in range(numlayers)])

# does more than inverting the above function; reduces size to fit the Neurophox convention  
def undo_clements_evenroll(X, numlayers):
    X = np.array([[np.roll(row, 0) if row_index%2==0 else np.roll(row, -1)\
                           for row,row_index in zip(X[layer], range(X[layer].shape[0]))]\
                           for layer in range(numlayers)])
    return X[:, :, ::2]
    
def error_correction(thetas, phis, gammas, alphas, betas, numlayers, inputsize, gamma_pos='in'):

    thetaprimes = np.zeros_like(thetas, dtype=np.float32)
    phiprimes = np.zeros_like(phis, dtype=np.float32)
    gammaprimes = np.zeros_like(gammas, dtype=np.float32)

    # How thetas are organized: inner dimension is phase shifts in a column of the hardware array
    #                           outer dimension selects column of hardware array
    # How e_l/e_rs are organized: inner dimension is phase shifts in a column of the hardware array
    #                             outer dimension selects column of hardware array
    # Correcting thetas
    
    # same theta error correction for both input and output phase screens
    arccosarg = (np.cos(thetas) + np.sin(2*alphas) * np.sin(2*betas)) / (np.cos(2*alphas) * np.cos(2*betas))
    arccosarg[arccosarg>=1] = 1; arccosarg[arccosarg<=-1] = -1; 
    thetaprimes = np.arccos(arccosarg)
    thetaprimes[:, 1::2, -1] = 0

    # Correcting phis (including the backshifting of the phases to the beginning)
    # backshifting of phases is hardcoded for clements

    zetabraw = np.angle( np.cos(alphas+betas) * np.cos(thetaprimes/2) +\
                     1j * np.sin(alphas-betas) * np.sin(thetaprimes/2) )
    zetacraw = np.angle( np.cos(alphas+betas) * np.cos(thetaprimes/2) -\
                     1j * np.sin(alphas-betas) * np.sin(thetaprimes/2) )
    zetadraw = np.angle( np.cos(alphas-betas) * np.sin(thetaprimes/2) -\
                     1j * np.sin(alphas+betas) * np.cos(thetaprimes/2) )
    zetabbase = np.angle(np.cos(thetas/2))
    zetacbase = np.angle(np.cos(thetas/2))
    zetadbase = np.angle(np.sin(thetas/2))

    zetab = zetabraw-zetabbase; zetac = zetacraw-zetacbase; zetad = zetadraw-zetadbase
    
    if gamma_pos == 'in':
        input_phases_upper = -zetac + (thetas-thetaprimes)/2
        input_phases_lower = -zetad + (thetas-thetaprimes)/2
    elif gamma_pos == 'out':
        input_phases_upper = -zetab + (thetas-thetaprimes)/2
        input_phases_lower = -zetad + (thetas-thetaprimes)/2
    else: raise ValueError('invalid value for gamma_pos')
        
    input_phases_upper[:, 1::2, -1] = 0; input_phases_lower[:, 1::2, -1] = 0;

    input_phases_layerwise_expanded = np.zeros((numlayers, inputsize, inputsize))
    input_phases_layerwise_expanded[:, :, :-1:2] = input_phases_upper
    input_phases_layerwise_expanded[:, :, 1::2] = input_phases_lower
    input_phases_layerwise_expanded = clements_evenroll(input_phases_layerwise_expanded, numlayers)

    input_phases_sentback = np.zeros_like(input_phases_layerwise_expanded)
    input_phases_sentback[:] = input_phases_layerwise_expanded
    # input_phases_sentback[:, 1::2, 0] = 0; input_phases_sentback[:, 1::2, -1] = 0;
    phimods = np.zeros_like(input_phases_layerwise_expanded)

    # defining meshlayout for even inputsize Clements 
    # meshlayout has 0 for no mzi, 1 for mzi first port, and 2 for the 2nd port
    meshlayout = np.array([k%2+1 for k in range(inputsize)])
    meshlayout = clements_evenroll(np.tile(meshlayout, (numlayers, inputsize, 1)), numlayers)
    meshlayout[:, 1::2, 0] = 0; meshlayout[:, 1::2, -1] = 0; 
    
    if gamma_pos == 'in':
        for i in range(numlayers):
            for j in range(inputsize-2, -1, -1):
                indices = np.arange(inputsize) + meshlayout[i, j, :]%2
                input_phases_sentback[i, j, :] += input_phases_sentback[i, j+1, indices] 
                phimods[i, j, :] = input_phases_sentback[i, j+1, :] - input_phases_sentback[i, j+1, indices]
    else:
        for i in range(numlayers):
            for j in range(1, inputsize):
                indices = np.arange(inputsize) + meshlayout[i, j, :]%2
                input_phases_sentback[i, j, :] += input_phases_sentback[i, j-1, indices] 
                phimods[i, j, :] = input_phases_sentback[i, j-1, :] - input_phases_sentback[i, j-1, indices]

    if gamma_pos == 'in':
        phiprimes = phis-zetab+zetad 
        gammaprimes = gammas+input_phases_sentback[:, :1, :]
    else:
        phiprimes = phis-zetac+zetad
        gammaprimes = gammas+input_phases_sentback[:, -1:, :]
    
    phiprimes += undo_clements_evenroll(phimods, numlayers)
    phiprimes[:, 1::2, -1] = 0
    
    return thetaprimes, phiprimes, gammaprimes
    
def noisytoideal(thetaprimes, phiprimes, gammaprimes, alphas, betas, numlayers, inputsize, gamma_pos='in'):
    
    thetas = np.zeros_like(thetaprimes, dtype=np.float32)
    phis = np.zeros_like(phiprimes, dtype=np.float32)
    gammas = np.zeros_like(gammaprimes, dtype=np.float32)
    
    arccosarg = np.cos(thetaprimes/2)**2 * np.cos(2*(alphas+betas)) -\
                np.sin(thetaprimes/2)**2 * np.cos(2*(alphas-betas))
    arccosarg[arccosarg>=1] = 1; arccosarg[arccosarg<=-1] = -1;
    thetas = np.arccos(arccosarg)
    thetas[:, 1::2, -1] = 0
    
    zetabraw = np.angle( np.cos(alphas+betas) * np.cos(thetaprimes/2) +\
                     1j * np.sin(alphas-betas) * np.sin(thetaprimes/2) ) 
    zetacraw = np.angle( np.cos(alphas+betas) * np.cos(thetaprimes/2) -\
                     1j * np.sin(alphas-betas) * np.sin(thetaprimes/2) )
    zetadraw = np.angle( np.cos(alphas-betas) * np.sin(thetaprimes/2) -\
                     1j * np.sin(alphas+betas) * np.cos(thetaprimes/2) )
    zetabbase = np.angle(np.cos(thetas/2))
    zetacbase = np.angle(np.cos(thetas/2))
    zetadbase = np.angle(np.sin(thetas/2))

    zetab = zetabraw-zetabbase; zetac = zetacraw-zetacbase; zetad = zetadraw-zetadbase

    #input_phases_upper = zetac - (thetas-thetaprimes)/2
    #input_phases_lower = zetad - (thetas-thetaprimes)/2
    
    if gamma_pos == 'in':
        input_phases_upper = zetac - (thetas-thetaprimes)/2
        input_phases_lower = zetad - (thetas-thetaprimes)/2
    elif gamma_pos == 'out':
        input_phases_upper = zetab - (thetas-thetaprimes)/2
        input_phases_lower = zetad - (thetas-thetaprimes)/2
    else: raise ValueError('invalid value for gamma_pos')
    
    input_phases_upper[:, 1::2, -1] = 0; input_phases_lower[:, 1::2, -1] = 0;
    
    input_phases_layerwise_expanded = np.zeros((numlayers, inputsize, inputsize))
    input_phases_layerwise_expanded[:, :, :-1:2] = input_phases_upper
    input_phases_layerwise_expanded[:, :, 1::2] = input_phases_lower
    input_phases_layerwise_expanded = clements_evenroll(input_phases_layerwise_expanded, numlayers)

    input_phases_sentback = np.zeros_like(input_phases_layerwise_expanded)
    input_phases_sentback[:] = input_phases_layerwise_expanded
    # input_phases_sentback[:, 1::2, 0] = 0; input_phases_sentback[:, 1::2, -1] = 0;
    phimods = np.zeros_like(input_phases_layerwise_expanded)
    
    # defining meshlayout for even inputsize Clements 
    # meshlayout has 0 for no mzi, 1 for mzi first port, and 2 for the 2nd port
    meshlayout = np.array([k%2+1 for k in range(inputsize)])
    meshlayout = clements_evenroll(np.tile(meshlayout, (numlayers, inputsize, 1)), numlayers)
    meshlayout[:, 1::2, 0] = 0; meshlayout[:, 1::2, -1] = 0;
    
    #for i in range(numlayers):
    #    for j in range(inputsize-2, -1, -1):
    #        indices = np.arange(inputsize) + meshlayout[i, j, :]%2
    #        input_phases_sentback[i, j, :] += input_phases_sentback[i, j+1, indices] 
    #        phimods[i, j, :] = input_phases_sentback[i, j+1, :] - input_phases_sentback[i, j+1, indices]
            
    if gamma_pos == 'in':
        for i in range(numlayers):
            for j in range(inputsize-2, -1, -1):
                indices = np.arange(inputsize) + meshlayout[i, j, :]%2
                input_phases_sentback[i, j, :] += input_phases_sentback[i, j+1, indices] 
                phimods[i, j, :] = input_phases_sentback[i, j+1, :] - input_phases_sentback[i, j+1, indices]
    else:
        for i in range(numlayers):
            for j in range(1, inputsize):
                indices = np.arange(inputsize) + meshlayout[i, j, :]%2
                input_phases_sentback[i, j, :] += input_phases_sentback[i, j-1, indices] 
                phimods[i, j, :] = input_phases_sentback[i, j-1, :] - input_phases_sentback[i, j-1, indices]
            
    #phis = phiprimes+zetab-zetad; phis += undo_clements_evenroll(phimods, numlayers)
    #phis[:, 1::2, -1] = 0
    #gammas = gammaprimes+input_phases_sentback[:, :1, :]
    
    if gamma_pos == 'in':
        phis = phiprimes+zetab-zetad 
        gammas = gammaprimes+input_phases_sentback[:, :1, :]
    else:
        phis = phiprimes+zetac-zetad
        gammas = gammaprimes+input_phases_sentback[:, -1:, :]
        
    phis += undo_clements_evenroll(phimods, numlayers); phis[:, 1::2, -1] = 0 
    
    return thetas, phis, gammas

########################################################################################################################
    
def train_network(inputsize, numlayers, gamma_pos, 
               nummodels, data, epochs, batch_size, foldername, saveflag, saveflagcheckpoints,
               selectedloss=tf.keras.losses.CategoricalCrossentropy(),
               trainlayervector=None, 
               bs_error=0.0):
    start = time.time()
    
    idealmodels = []
    initaccuracies = np.zeros((nummodels,))
    num_test = len(data.y_test_ind)
#     selectedloss = 'mse'

    for i in np.arange(nummodels):
        model = const_onn_EO(inputsize, L=numlayers, 
                               gamma_pos='out',
                               bs_error=bs_error,
                               trainlayervectorin=trainlayervector)
        model.compile(optimizer='adam',
                             loss=selectedloss,
                             metrics=['accuracy'])
        predoutputs = np.argmax(model.predict(data.x_test), axis=1)
        initaccuracies[i] = 100*(1 - np.count_nonzero(predoutputs-data.y_test_ind)/num_test)
        idealmodels.append(model)
        
    modelhistories = []
    modelhistories.append(bs_error)
    idealaccuracies = np.zeros((nummodels,))
    
    filepath = foldername+"/saved-model-{epoch:02d}-{val_accuracy:.2f}"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, 
                                                                   monitor='val_loss', verbose=0, 
                                                                   save_best_only=False, 
                                                                   save_weights_only=False, 
                                                                   mode='auto', period=1)

    for i in np.arange(nummodels):
        if saveflagcheckpoints:
            history = idealmodels[i].fit(data.x_train,
                                              data.y_train,
                                              epochs=epochs,
                                              batch_size=batch_size,
                                              validation_data=(data.x_test, data.y_test),
                                              verbose=2, 
                                              callbacks=[model_checkpoint_callback])
        else:
            history = idealmodels[i].fit(data.x_train,
                                              data.y_train,
                                              epochs=epochs,
                                              batch_size=batch_size,
                                              validation_data=(data.x_test, data.y_test),
                                              verbose=2)
            
        predoutputs = np.argmax(idealmodels[i].predict(data.x_test), axis=1)
        idealaccuracies[i] = 100*(1 - np.count_nonzero(predoutputs-data.y_test_ind)/num_test)
        modelhistories.append(history.history)
        # save trained models
        if saveflag:
            idealmodels[i].save(foldername+f'/idealmodels{i}')

            thetas, phis, gammas = param_extraction(idealmodels[i])
            with open(foldername+'/idealthetas.pickle', 'wb') as outputfile:
                pickle.dump((thetas, phis, gammas), outputfile)
    
    modelhistories.append(idealaccuracies)
    if saveflag:
        with open(foldername+'/idealhistories.pickle', 'wb') as outputfile:
            pickle.dump(modelhistories, outputfile)
        
    print(f'Done, time={(time.time()-start)/60}')
#     del idealmodels
    return initaccuracies, idealaccuracies
