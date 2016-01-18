
"""
Project Gondor MLMVN network class, CPU and GPU
"""

import sys
import importlib
import re

import numpy as np
from scipy.io import loadmat, savemat
import timeit
from os import path

# Placeholder for gpu math modules
gpuarray = None
culinalg = None
cumath = None
misc = None


class network(object):
    """
    Main mlmvn simulator class. Initializes a multi-layer, multi-valued neural
    network with specified learning samples and applies various learning
    algorithms to it.
    """

    def init_host(self):
        """
        Initialize the network objects for use on the host machine. Required
        for both host and device implementations
        """

        # Import initial input samples (and desired outputs) from text file
        if type(self.inputFile) is str:
            self.initialInputs_h = np.loadtxt(self.inputFile)

        elif type(self.inputFile) is np.ndarray:
            self.initialInputs_h = self.inputFile

        else:
            raise IOError('Input learning samples must be either a text file \
                          or a numpy array')

        # Get the dimensions of the input sample array
        self.inputDim = self.initialInputs_h.shape

        # Get number of layers (input, hidden and output) as well as
        # the number of output neurons
        self.numLayers = len(self.networkSize_h)
        self.numLayers_1 = self.numLayers - 1
        self.numOutputs_h = self.networkSize_h[-1]

        if self.stopKey == 'error':
            if self.numOutputs_h == 1:
                self.errorRate_axis = None
            else:
                self.errorRate_axis = 0

        self.numSamples_h = self.inputDim[0]
        self.inputsPerSample_h = self.inputDim[1] - self.numOutputs_h

        # Parse the desired outputs from the input file
        self.desiredOutputs_h = (np.copy(
            self.initialInputs_h[0: self.inputDim[0],
                                 self.inputsPerSample_h: self.inputDim[1]],
            order='C'))

        self.outputsPerSample = self.desiredOutputs_h.shape[1]

        # pi * 2, needed a lot
        self.twoPi_h = np.float32(np.pi * 2)

        self.halfSectors_h = np.floor(self.numSectors_h / 2)

        # Get angular size of each sector
        self.sectorSize_h = self.twoPi_h / self.numSectors_h

        if self.discreteOutput is True:
            # Initialize container for complex-conjugated desired outputs
            self.complex_desiredOutputs_h = (
                np.exp((self.desiredOutputs_h + 0.5) *
                       1j * self.sectorSize_h).astype(np.complex64))
            # Desired discrete outputs equal a root of unity corresponding to
            # the bisector of a desired output
            if self.stopKey is 'armse':
                self.angular_desiredOutputs_h = np.mod(
                    np.angle(self.complex_desiredOutputs_h), self.twoPi_h)
        else:
            self.complex_desiredOutputs_h = np.exp(self.desiredOutputs_h * 1j)

            # if self.stopKey == 'armse':
            self.angular_desiredOutputs_h = np.copy(self.desiredOutputs_h)

        # Parse the initial input samples from the input file
        self.initialInputs_h = np.copy(self.initialInputs_h[
            0: self.inputDim[0], 0: self.inputsPerSample_h], order='C')

        # Initialize container for the network outputs
        self.networkOutputs_h = np.zeros(
            (self.numSamples_h, self.numOutputs_h), order='C',
            dtype=np.complex64)

        # Output errors container will hold errors of output layer neurons
        # for all samples before compensating for errors that are greater
        # than half the number of sectors
        self.outputErrors_h = np.empty_like(self.networkOutputs_h)

        # Initialize containers that indicates which output values need to
        # be adjusted and which should be saved
        self.badOutput_h = np.empty_like(self.networkOutputs_h, dtype=bool)
        self.goodOutput_h = np.empty_like(self.networkOutputs_h, dtype=bool)

        # Initialize container for mask used to fix border cases
        self.outputMask_h = np.zeros_like(self.networkOutputs_h,
                                          dtype=np.float32)
        # If using soft margins, initialize relevant containers
        if self.softMargins is True and self.stopKey is 'armse':
            # argOutputs will hold the argument of the weighted sums for
            # the output neurons
            self.argOutputs_h = np.empty_like(self.networkOutputs_h)

        # Initialize container for the network errors over all samples
        self.networkErrors_h = np.zeros((self.numSamples_h, 1),
                                        order='C', dtype=np.complex64)

        # Initialize global and local error values to 0, important for
        # max error algorithm
        self.globalError_h = 0
        self.localError_h = 0

        # Initialize complex unit circle
        self.sectors_h = np.zeros((self.numSectors_h, 1), dtype=np.complex64)

        # Calculate values for the unit circle
        for sector in xrange(self.numSectors_h):
            angSector = (self.twoPi_h * (sector - 1)) / self.numSectors_h
            self.sectors_h[sector] = np.exp(angSector * 1j)

        # Initialize the network weights
        self.__getWeights__()

        # Initialize array to hold layer outputs
        self.neuronOutputs_h = np.empty(self.numLayers, order='C', dtype=object)
        self.neuronErrors_h = np.empty_like(self.neuronOutputs_h)
        self.weightedSum_h = np.empty(self.numLayers, dtype=object)
        self.learningRates_h = np.empty(self.numLayers, dtype=object)

        # Iterate over each layer and allocate initialize a subarray
        for layer in xrange(0, self.numLayers):
            # Create a temporary array for the output vector of each layer
            temp_outputs = np.zeros((self.networkSize_h[layer], 1),
                                    order='C', dtype=np.complex64)

            self.neuronOutputs_h[layer] = np.copy(temp_outputs)
            self.neuronErrors_h[layer] = np.copy(temp_outputs)
            self.weightedSum_h[layer] = np.copy(temp_outputs)
            self.learningRates_h[layer] = np.copy(temp_outputs)

        del temp_outputs

        # Initialize container for holding input values, converted with roots
        # of complex unities
        if self.discreteInput:
            # Convert sector values into complex numbers on the unit circle
            theta = self.twoPi_h * (self.initialInputs_h / self.numSectors_h)
            rho = np.ones((self.initialInputs_h.shape[0],
                           self.initialInputs_h.shape[1]))

            # Convert discrete input values (integers) into real and imaginary
            # parts
            self.inputs_h = np.asfortranarray(self.pol2cart(rho, theta))

            del theta, rho

        else:
            temp = np.zeros(self.initialInputs_h.shape,
                            order='C', dtype=np.complex64)
            temp.imag = self.initialInputs_h
            self.inputs_h = np.exp(temp)

            del temp


        self.iterations = 0

    def init_device(self):
        """
        Initialize network objects for use on the GPU. Required for device
        implementation
        """

        self.initCuda()

        # Send desired outputs to the device
        self.desiredOutputs_d = gpuarray.to_gpu_async(
            self.desiredOutputs_h.astype(np.complex64))

        # Build container for holding network size (inelegantly)
        # This is necessary due to a limitation of the pycuda gpuarray object,
        # which sometimes treats an array of a single element as a
        # dimensionless array.
        self.networkSize_d = np.empty(self.numLayers, dtype=object)

        for layer in xrange(self.numLayers):
            if layer < self.numLayers_1 - 1:
                temp = (np.expand_dims(np.repeat(
                    self.networkSize_h[layer],
                    self.networkSize_h[layer + 2]), axis=1))

            elif layer == self.numLayers_1 - 1:
                temp = (np.expand_dims(np.repeat(
                    self.networkSize_h[layer],
                    self.networkSize_h[layer + 1]), axis=1))

            else:
                temp = (np.expand_dims(np.repeat(
                    self.networkSize_h[layer],
                    self.networkSize_h[layer]), axis=1))

            self.networkSize_d[layer] = gpuarray.to_gpu_async(
                np.copy(temp).astype(np.complex64))

        # Send complex desired outputs to the device

        self.complex_desiredOutputs_d = np.empty(
            self.numSamples_h, dtype=object)

        for sample in xrange(self.numSamples_h):
            self.complex_desiredOutputs_d[sample] = gpuarray.to_gpu_async(
                np.copy(np.expand_dims(
                    self.complex_desiredOutputs_h[sample], axis=1), order='C'))

        #self.complex_desiredOutputs_d = gpuarray.to_gpu_async(
            #np.copy(np.expand_dims(
                #self.complex_desiredOutputs_h, axis=2), order='C'))

        # Send initial input samples to the device
        self.initialInputs_d = gpuarray.to_gpu_async(self.initialInputs_h)

        temp = np.asfortranarray(np.expand_dims(np.repeat(
            self.numOutputs_h, self.numSamples_h), axis=1))

        #temp = np.asfortranarray(np.full((self.numSamples_h,
                                          #self.numOutputs_h),
                                         #self.numOutputs_h))

        self.numOutputs_d = gpuarray.to_gpu_async(
            temp.astype(np.float32))

        self.numSamples_d = gpuarray.to_gpu_async(
            np.array([self.numSamples_h]).astype(np.float32).reshape(1, 1))

        # Send container for network outputs to the device
        self.networkOutputs_d = gpuarray.to_gpu_async(self.networkOutputs_h)

        # Workaround for setting elements in a gpuarray by their index
        # Calculates the column-wise indices of the outputs for each sample,
        # which will be used when determining the outputs of the network.
        self.samplesIndex_d = np.empty(
            self.numSamples_h, order='C', dtype=object)

        #for sample in xrange(self.numSamples_h):
            #self.samplesIndex_d[sample] = gpuarray.to_gpu_async(
                #np.asfortranarray(np.arange(
                    #sample, (self.numSamples_h * self.numOutputs_h),
                    #self.numSamples_h)))

        for sample in xrange(self.numSamples_h):
            self.samplesIndex_d[sample] = gpuarray.to_gpu_async(
                np.asfortranarray(np.arange(
                    sample * self.numOutputs_h,
                    (sample * self.numOutputs_h) +
                    self.numOutputs_h)))

        # Send containers for soft margin errors to the device
        self.outputErrors_d = gpuarray.to_gpu_async(self.outputErrors_h)
        self.badOutput_d = gpuarray.to_gpu_async(
            self.badOutput_h.astype(np.float32))

        self.goodOutput_d = gpuarray.to_gpu_async(
            self.goodOutput_h.astype(np.float32))
        self.outputMask_d = gpuarray.to_gpu_async(self.outputMask_h)

        # Initialize device container for network errors over all samples
        self.networkErrors_d = gpuarray.to_gpu_async(self.networkErrors_h)

        # Initialize containers for the global and local error values
        self.globalError_d = gpuarray.zeros((1, 1), dtype=np.complex64)
        self.localError_d = gpuarray.zeros((1, 1), dtype=np.complex64)

        # Send the local and global threshold values to the gpu
        self.globalThreshold_d = gpuarray.to_gpu_async(
            np.array(self.globalThreshold_h).astype(np.float32).reshape(1, 1))

        self.localThreshold_d = gpuarray.to_gpu_async(
            np.array(self.localThreshold_h).astype(np.float32).reshape(1, 1))

        temp = np.asfortranarray(np.expand_dims(np.repeat(
            np.pi, self.numSamples_h), axis=1))

        self.pi_d = gpuarray.to_gpu(temp.astype(np.float32))

        temp = np.asfortranarray(np.expand_dims(np.repeat(
            self.twoPi_h, self.outputsPerSample), axis=1))

        self.twoPi_d = gpuarray.to_gpu_async(temp)

        temp = np.asfortranarray(np.expand_dims(np.repeat(
            self.twoPi_h, self.numSamples_h), axis=1))

        self.twoPi_2_d = gpuarray.to_gpu_async(temp)

        # Workaround for sending the number of sectors to the gpu.
        # Elementwise division won't work otherwise
        #temp = np.asfortranarray(np.expand_dims(np.repeat(
            #self.numSectors_h, self.numSamples_h), axis=1))

        temp = np.asfortranarray(np.full((self.numSamples_h,
                                          self.outputsPerSample),
                                         self.numSectors_h))

        self.numSectors_d = gpuarray.to_gpu_async(
            temp.astype(np.float32))

        # Workaround for sending half the number of sectors to the gpu.
        temp = np.asfortranarray(np.full((self.numSamples_h,
                                          self.outputsPerSample),
                                         self.halfSectors_h))

        self.halfSectors_d = gpuarray.to_gpu_async(
            temp.astype(np.float32))

        # Another workaround, same issue
        temp = np.expand_dims(np.repeat(
            self.sectorSize_h, self.numOutputs_h), axis=1)

        self.sectorSize_d = gpuarray.to_gpu_async(
            np.copy(temp).astype(np.complex64))

        # Initialize container to hold the inputsPerSample value
        temp = (np.expand_dims(np.repeat(
            self.inputsPerSample_h, self.networkSize_h[1]), axis=1))

        self.inputsPerSample_d = gpuarray.to_gpu_async(
            temp.astype(np.complex64))

        # Initialize and fill containers for the network weights on the gpu
        # Note that the bias weights for each layer are stored in a separate
        # container due to pycuda limitations
        self.networkWeights_d = np.empty(self.numLayers, dtype=object)
        self.networkWeights_bias_d = np.empty(self.numLayers, dtype=object)

        # Randomize weights with pseudorandom generator:
        # self.__randomizeWeights__()

        # Transfer host weights to the gpu
        self.__transferWeights__()

        # Initialize and fill other containers
        self.neuronOutputs_d = np.empty(self.numLayers, dtype=object)
        self.neuronErrors_d = np.empty(self.numLayers, dtype=object)
        self.weightedSum_d = np.empty(self.numLayers, dtype=object)
        self.learningRates_d = np.empty(self.numLayers, dtype=object)

        for layer in xrange(self.numLayers):

            self.neuronOutputs_d[layer] = gpuarray.to_gpu_async(
                self.neuronOutputs_h[layer])
            self.neuronErrors_d[layer] = gpuarray.to_gpu_async(
                self.neuronErrors_h[layer])
            self.weightedSum_d[layer] = gpuarray.to_gpu_async(
                self.weightedSum_h[layer])
            self.learningRates_d[layer] = gpuarray.to_gpu_async(
                self.learningRates_h[layer])

        # Export the complex initial inputs to the gpu
        #temp = np.asfortranarray(np.expand_dims(self.inputs_h, axis=2))
        #self.inputs_d = gpuarray.to_gpu(temp)

        self.inputs_d = np.empty(self.numSamples_h, dtype=object)
        for sample in xrange(self.numSamples_h):
            self.inputs_d[sample] = gpuarray.to_gpu_async(
                np.copy(np.expand_dims(self.inputs_h[sample], axis=1),
                        order='C'))

        # Delete temp - it did good, but we don't need it now
        del temp

    def initCuda(self):
        """
        Imports and initializes the libraries necessary to utilize CUDA
        """
        # Try to import the CUDA-related libraries, including pycuda and skcuda
        try:
            # This block of code should work, but it uses global variables
            # as part of an import operation. Although this works in CPython,
            # it's a big no-no according to the Python 2.7 docs

            global autoinit, culinalg, cumath, gpuarray, misc

            autoinit = importlib.import_module('..autoinit', 'pycuda.subpkg')
            gpuarray = importlib.import_module('..gpuarray', 'pycuda.subpkg')
            cumath = importlib.import_module('..cumath', 'pycuda.subpkg')

            culinalg = importlib.import_module('..linalg', 'skcuda.subpkg')
            misc = importlib.import_module('..misc', 'skcuda.subpkg')

        except ImportError:
            raise ImportError('The CUDA-related libraries could not be loaded,\
                              with exception message: ', sys.exc_info()[1])


        # Initialize the skcuda libraries and the pycuda device context
        try:
            # Initialize the cuda linear algebra library
            culinalg.init()
            misc.init()
            autoinit.device

        except AttributeError:
            raise AttributeError('The CUDA-related libraries could not be\
                                 initialized, with exception message: ',
                                 sys.exc_info()[1])

    def __init__(self, inputName='../../DATASETS/MODIFIED/IrisMVN-3_Learn-75.txt',
                 outputName='outputTest', netSize=[2, 1], discInput=False,
                 discOutput=True, globalThresh=0.0, localThresh=0.0,
                 sectors=3, weightKey='random', stopKey='rmse',
                 softMargins=True, cuda=False, iterLimit=None,
                 refreshLimit=1, timeLimit=None):

        # Initialize strings for the input sample and output weights filenames
        self.inputFile = inputName

        if outputName is not None and outputName == '':
            raise TypeError('Weight export filenames must be\
                                 a nonempty string')

        self.outputFile = outputName

        self.weightKey = weightKey

        self.stopKey = stopKey

        # Boolean value indicating whether soft error margins are used
        self.softMargins = softMargins

        # Boolean values indicating whether input and output values are discrete
        # or continuous

        #if type(discInput) != bool or type(discOutput) != bool:
            #raise TypeError('discInput and discOutput should be boolean values')
        self.discreteInput = discInput
        self.discreteOutput = discOutput

        # Values for global and local error thresholds
        #if ((type(globalThresh) != int and float) or
                #(type(localThresh) != int and float)):
            #raise TypeError('local and global thresholds should be integer \
                            #or float values')

        if globalThresh < 0 or localThresh < 0:
            raise ValueError('local and global thresholds should be positive\
                             real numbers')

        if localThresh > globalThresh:
            raise ValueError('local threshold should be less than global\
                             threshold')
        self.globalThreshold_h = globalThresh
        self.localThreshold_h = localThresh

        # Network topology contained in a list - if the topology is undefined,
        # set it to a null value, and it will be checked later
        if netSize is not None:
            # Topology was passed as a string
            size = netSize
            if type(netSize) == str:
                # Strip off brackets and delimiters and convert to integer list
                size = [int(i) for i in re.split('\W+|_', netSize)]

            self.networkSize_h = np.asarray(size, dtype=int)

        else:
            self.networkSize_h = None

        #if ((len(self.networkSize_h.shape) == 1 and
             #self.networkSize_h.shape[0] == 1) or
            #(len(self.networkSize_h.shape) == 2 and
             #self.networkSize_h.shape[0] != 1) or
                #self.networkSize_h.dtype != int):
            #raise ValueError('netSize should be a 1-dimensional array\
                             #of integers with at least 2 elements')

        # Get number of sectors for the unit circle
        #if type(sectors) != int or float:
            #raise TypeError('sectors should be an integer value - a float value\
                            #will be floored to the closest integer value')
        self.numSectors_h = int(sectors)

        # Iteration limit is an integer
        #if ((type(iterLimit) != int and float) and iterLimit is not None):
            #raise TypeError('iterLimit should be an integer value or a null\
                            #value')
        #if iterLimit <= 0:
            #raise ValueError('iterLimit should be a positive integer')
        self.iterLimit = iterLimit
        if not self.iterLimit:
            self.iterLimit = float('inf')

        # refresh limit defines the number of iterations per error report
        # for the learning algorithm - default value is 1, so a report will
        # be made every iteration
        #if type(refreshLimit) != int and float:
            #raise TypeError('refreshLimit should be an integer value - a float\
                            #value will be floored to the closest integer value')
        self.refreshLimit = int(refreshLimit)

        if self.refreshLimit <= 0:
            self.refreshLimit = float('inf')

        self.timing = 0.0

        if timeLimit:
            raise Warning('Simulation time limit is a feature of the GUI',
                          ' and cannot be used in the command line')

        # These flags are used to keep track of the currently-running algorithm
        self.__learning__ = False
        self.__testing__ = False
        self.__randomizing__ = False
        self.terminate = False

        # Flag determines whether or not pycuda will be used
        #if type(cuda) != bool:
            #raise TypeError('Cuda flag should be a boolean value')
        self.useCuda = cuda
        # Initialize network objects for the host
        self.init_host()
        # If pycuda is used, initialize network objects on the device
        if self.useCuda is True:
            self.init_device()

    def printParameters(self):
        print '{:-^40}'.format('Network Parameters') + \
            '\nTopology: ', self.networkSize_h, \
            '\nInput file: ' + self.inputFile + \
            '\nOutput file: ' + (self.outputFile if
                                 type(self.outputFile) == str else 'None') + \
            '\nInput values\n', self.initialInputs_h, \
            '\n\nDesired outputs\n', self.desiredOutputs_h, \
            '\n\nStopping criteria: ', self.stopKey, \
            '\nDiscrete input: ', self.discreteInput, \
            '\nDiscrete output: ', self.discreteOutput, \
            '\nGlobal threshold: ', self.globalThreshold_h, \
            '\nLocal threshold: ', self.localThreshold_h, \
            '\nInitial weights: ', self.weightKey, \
            '\nSectors: ', self.numSectors_h, \
            '\n' + '-' * 40

    def __action_sendMetrics__(self):
        """
        Helper function, sends metrics to the GUI or prints it to the terminal.
        If the simulator is learning, send the current iteration and the current
        global error; if testing, send the absolute number of errors over all
        samples, the overall accuracy of the current weights, and the rmse value
        for those weights
        """

        # Learning case - send a report per number of iterations
        if (self.__learning__ and not self.__testing__ and
                self.iterations % self.refreshLimit == 0):
            # If using CUDA, transfer global error from device to host
            if self.useCuda is True:
                self.globalError_h = self.globalError_d.get()[0][0]

            # Non-GUI case - print the iteration, error type, and current
            # global error
            print 'Current time: {0:.3f}'.format(self.timing)
            print self.iterations, '- Global error', self.stopKey, \
                '= {0:.3f}'.format(self.globalError_h)


        # Testing case
        elif self.__testing__:

            print 'Absolute errors = {0:.0f}'.format(self.AbsErrors), \
                '\nAccuracy = {0:.2f}%'.format(self.Accuracy), \
                '\nRMSE = {0:.3f}'.format(self.rmse)

    def __getOutputs__(self):
        """
        Helper function, returns the current network outputs over all learning
        samples
        """

        if self.useCuda is True:
            for sample in xrange(self.numSamples_h):
                self.networkOutputs_h[sample] = self.networkOutputs_d[sample].get()

        return self.networkOutputs_h

    def __getWeights__(self):
        """
        Helper function, handles importing user-defined weights or randomly
        generating weights
        """

        # String was passed as weights argument
        if type(self.weightKey) == str:
            # If user-provided weights are specified (.mat file)
            if self.weightKey.endswith('.mat'):
                # Load the weight array (inelegantly) to the network
                # weights container if the dimensions match
                self.networkWeights_h = (
                    (loadmat(self.weightKey))[path.basename(
                        self.weightKey)[:-4]][0])

                # Reorder the weights array in case they need to be exported
                # to a GPU simulation
                self.networkWeights_h = np.array(
                    [self.networkWeights_h[layer].copy(order='C') for layer in
                     xrange(self.numLayers)], order='C')

                # Check the input weight dimensions
                assert self.__checkWeights__()

            # If a .mat filename was not passed by the user
            else:
                # Initialize array to hold network weights
                self.networkWeights_h = np.empty(self.numLayers, order='C',
                                                 dtype=object)
                # Use a uniform distribution to generate initial weight values
                if self.weightKey == 'random':
                    self.__randomizeWeights__()

        # An array was passed as a weight vector
        elif type(self.weightKey) == np.ndarray:
            # Assign the passed array to the weights container
            self.networkWeights_h = self.weightKey
            assert self.__checkWeights__()

    def __checkWeights__(self):
        """
        Helper function, checks if the given weights match the topology and
        dimensions of the network and returns an error if they don't
        """
        # Only check if a network topology was passed
        if self.networkSize_h is not None:
            # Check the number of layers in the network v. layers in the weights
            if self.networkSize_h.shape != self.networkWeights_h.shape:
                raise ValueError('The imported weights with ',
                                 self.weightKey.shape[0],
                                 ' layers does not match network with ',
                                 self.networkSize.shape[0], ' layers')

            # Check the number of neurons in each layer
            for layer in xrange(self.numLayers):
                if self.networkSize_h[layer] != self.networkWeights_h[layer].shape[0]:
                    raise ValueError('The imported weights with ',
                                     self.weightKey[layer].shape[0],
                                     ' neurons in layer', layer,
                                     ' does not match ',
                                     self.networkSize_h[layer],
                                     ' neurons in the given network')
                layerShape = ()

                if layer == 0:
                    layerShape += (self.networkSize_h[0],
                                   (self.inputsPerSample_h + 1))

                else:
                    layerShape += (self.networkSize_h[layer],
                                   (self.networkSize_h[layer - 1] + 1))

                # Check the dimensions of the weight array for each layer
                if layerShape != self.networkWeights_h[layer].shape:
                    raise ValueError('The imported weights at layer ',
                                     layer, 'have dimensions ',
                                     self.networkWeights_h[layer].shape,
                                     ' but need dimensions ', layerShape)

        # If a topology was not passed, use the shape of the weights array as
        # the network topology
        elif self.networkSize_h is None:
            shape = []

            for layer in xrange(len(self.networkWeights_h)):
                shape.append(self.networkWeights_h[layer].shape[0])

            self.networkSize_h = np.asarray(shape, dtype=int)

        return True

    def __checkParams__(self):
        """
        Helper function, checks that all parameters passed to the simulator
        are valid
        """

        allStopKeys = ['max', 'error', 'mse', 'rmse', 'armse']
        if self.stopKey not in allStopKeys:
            raise ValueError('Stop key must be one of ', allStopKeys)

        if self.stopKey != 'armse' and self.softMargins is True:
            raise TypeError('Soft margins can only be applied to learning\
                            with armse')

        if type(self.softMargins) != bool:
            raise TypeError('softMargins should be a boolean value')

    def __randomizeWeights__(self):
        """
        Helper function, randomizes weights for the entire network
        """

        if self.__randomizing__ is False:
            self.__randomizing__ = True

        for layer in xrange(0, self.numLayers):
            if layer == 0:
                # Create a temporary array to hold generated weight values
                temp = np.empty((self.networkSize_h[layer],
                                self.inputsPerSample_h + 1),
                                order='C', dtype=np.complex64)

                # Fill the real and imaginary parts with numbers from a
                # uniform distribution and offset by 0.5 + 0.5j
                temp.real = np.random.uniform(size=temp.shape)
                temp.imag = np.random.uniform(size=temp.shape)

                temp -= complex(0.5, 0.5)

                self.networkWeights_h[layer] = temp

            else:
                temp = np.empty((self.networkSize_h[layer],
                                self.networkSize_h[layer - 1] + 1),
                                order='C', dtype=np.complex64)

                temp.real = np.random.uniform(size=temp.shape)
                temp.imag = np.random.uniform(size=temp.shape)

                temp -= complex(0.5, 0.5)

                self.networkWeights_h[layer] = np.copy(temp)

        # Delete this object - we won't need it anymore
        del temp

        self.__doneMessage__()

        return

    def __transferWeights__(self):
        """
        Helper function, transfers current weights from the host to the device
        """
        if self.useCuda is True:
            for layer in xrange(self.numLayers):
                # Transfer input weights
                self.networkWeights_d[layer] = gpuarray.to_gpu_async(
                    np.copy(self.networkWeights_h[layer][:, 1:]))

                # Transfer bias weights
                self.networkWeights_bias_d[layer] = gpuarray.to_gpu_async(
                    np.copy(np.expand_dims(
                        self.networkWeights_h[layer][:, 0], axis=1)))

    def __doneMessage__(self):
        """
        Prints a message telling user that a task is complete
        """

        if self.__learning__:
            print 'Learning process completed'

        elif self.__testing__:
            print 'Testing process completed'

        elif self.__randomizing__:
            print 'Weights randomized'

    def __stop__(self):
        """
        Helper function, forces the stop flag to True
        """

        self.terminate = True

    def __reset__(self):
        """
        Resets all flags, as well as iteration data
        """

        self.__learning__ = False
        self.__testing__ = False
        self.__randomizing__ = False

        self.terminate = False
        self.finishedLearning = False

        self.iterations = 0

    def checkContinue(self):
        """
        Check if the currently running process should continue or not, based
        on the terminate flag and the iteration cap
        """

        if (self.terminate or self.iterLimit <= self.iterations or
                self.finishedLearning):
            return False

        else:
            return True

    def exportWeights(self, outputFile=None):
        """
        Export the current network weights to a file with the designated
        output file name
        """

        # If CUDA was used, we need to transfer the contents of the
        # gpuarray instances to the host container before saving
        if self.useCuda is True:
            for layer in xrange(self.numLayers):
                self.networkWeights_h[layer] = (
                    np.concatenate([self.networkWeights_bias_d[layer].get(),
                                   self.networkWeights_d[layer].get()], axis=1))

        if self.outputFile is not None:
            # Split output filename into two parts: the absolute root path and
            # the filename - head and tail - respectively
            head, tail = path.split(self.outputFile)
            # Filename is the absolute path, dict key for weights is relative
            # filename
            savemat(head + tail, {tail: self.networkWeights_h})

        elif self.outputFile is None and outputFile is not None:
            savemat(outputFile, {outputFile: self.networkWeights_h})

        else:
            raise TypeError('No save file name was provided')

        return

    def pol2cart(self, rho, theta):
        """
        Helper function, converts polar coordinates to cartesian ones
        """
        numRows = theta.shape[0]
        numCols = theta.shape[1]

        cartesianMat = np.zeros((numRows, numCols), dtype=np.complex64)

        cartesianMat.real = rho * np.cos(theta)
        cartesianMat.imag = rho * np.sin(theta)

        return cartesianMat

    def __gpu_mod__(self, x, y):
        """
        Helper function, calculates the modulus of two gpu arrays.
        """

        return x - cumath.floor(x / y) * y

    def correct_networkWeights(self, sample, layer):
        """
        Adjusts the weights for each neuron in the network using a self-adaptive
        learning rate
        """
        if self.useCuda is False:
            # Calculate the learning rate as the reciprocal absolute value of
            # weighted sum
            self.learningRates_h[layer] = np.power(
                np.abs(self.weightedSum_h[layer]), -1)

            # Evaluating the input layer
            if layer == 0:
                # Calculate the weights for each input of each neuron in the
                # current layer
                self.networkWeights_h[layer][:, 1:] += (np.outer(
                    self.learningRates_h[layer] * self.neuronErrors_h[layer],
                    np.conjugate(self.inputs_h[sample, :])))

                # Calculate the bias weight for each input of each neuron
                self.networkWeights_h[layer][:, 0] += (
                    self.learningRates_h[layer] * self.neuronErrors_h[layer])

            # Evaluating the hidden and output layers
            elif 1 <= layer <= self.numLayers_1:
                # Recalculate the weighted sum for the previous layer
                self.calculate_neuronOutputs(sample, layer - 1)

                # Evaluating weights of the hidden layer neurons
                if layer < self.numLayers_1:

                    # Adjust the weights of the current layer
                    self.networkWeights_h[layer][:, 1:] += (np.outer(
                        (self.neuronErrors_h[layer] *
                         self.learningRates_h[layer]),
                        np.conjugate(self.neuronOutputs_h[layer - 1])))

                    # Evaluating bias weight for the hidden layers
                    self.networkWeights_h[layer][:, 0] += (
                        self.learningRates_h[layer] *
                        self.neuronErrors_h[layer])

                # Evaluating weights of the output layer neurons
                else:

                    self.networkWeights_h[layer][:, 1:] += (np.outer(
                        self.neuronErrors_h[layer],
                        np.conjugate(self.neuronOutputs_h[layer - 1])))

                    # Evaluating bias weight for the output layer
                    self.networkWeights_h[layer][:, 0] += (
                        self.neuronErrors_h[layer])

        # GPU implementation
        else:
            self.learningRates_d[layer] = (
                (self.weightedSum_d[layer].__abs__()).__pow__(-1))

            if layer == 0:

                self.networkWeights_d[layer] += (culinalg.dot(
                    (self.learningRates_d[layer] * self.neuronErrors_d[layer]),
                    (culinalg.hermitian(self.inputs_d[sample]))))

                self.networkWeights_bias_d[layer] += (
                    self.learningRates_d[layer] * self.neuronErrors_d[layer])

            elif 1 <= layer <= self.numLayers_1:

                self.calculate_neuronOutputs(sample, layer - 1)

                if layer < self.numLayers_1:
                    self.networkWeights_d[layer] += (culinalg.dot(
                        (self.neuronErrors_d[layer] *
                         self.learningRates_d[layer]),
                        culinalg.hermitian(self.neuronOutputs_d[layer - 1])))

                    self.networkWeights_bias_d[layer] += (
                        self.learningRates_d[layer] *
                        self.neuronErrors_d[layer])

                else:
                    self.networkWeights_d[layer] += (culinalg.dot(
                        self.neuronErrors_d[layer],
                        culinalg.hermitian(self.neuronOutputs_d[layer - 1])))

                    self.networkWeights_bias_d[layer] += (
                        self.neuronErrors_d[layer])

    def calculate_neuronOutputs(self, sample, layer):
        """
        Calculate the continuous outputs for the current layer for the current
        input sample
        """
        if self.useCuda is False:
            # Evaluating the first layer
            if layer == 0:
                self.neuronOutputs_h[layer] = (
                    np.dot(self.networkWeights_h[layer][:, 1:],
                           self.inputs_h[sample]) +
                    self.networkWeights_h[layer][:, 0])

            elif 0 < layer <= self.numLayers_1:
                self.neuronOutputs_h[layer] = (
                    np.dot(self.networkWeights_h[layer][:, 1:],
                           self.neuronOutputs_h[layer - 1]) +
                    self.networkWeights_h[layer][:, 0])

            # Transfer the calculated output values to the weighted sum
            # container
            self.weightedSum_h[layer] = self.neuronOutputs_h[layer]

            # If the current layer isn't the output layer, use activation func
            if 0 <= layer < self.numLayers_1:
                self.neuronOutputs_h[layer] = self.activateFunc(
                    self.neuronOutputs_h[layer])

        # Else compute on device
        else:
            if layer == 0:
                self.neuronOutputs_d[layer] = (
                    culinalg.dot(self.networkWeights_d[layer],
                                 self.inputs_d[sample]) +
                    self.networkWeights_bias_d[layer])

            elif 0 < layer <= self.numLayers_1:
                self.neuronOutputs_d[layer] = (
                    culinalg.dot(self.networkWeights_d[layer],
                                 self.neuronOutputs_d[layer - 1]) +
                    self.networkWeights_bias_d[layer])

            self.weightedSum_d[layer] = self.neuronOutputs_d[layer]

            if 0 <= layer < self.numLayers_1:
                self.neuronOutputs_d[layer] = self.activateFunc(
                    self.neuronOutputs_d[layer])

    def calculate_networkOutputs(self, sample):
        """
        Calculate the outputs of the whole network for the current sample
        """

        if self.useCuda is False:
            # If network outputs are discrete
            if self.discreteOutput is True:
                # Calculate the argument of the output neurons
                self.neuronOutputs_h[-1] = (np.mod(
                    np.angle(self.neuronOutputs_h[-1]), self.twoPi_h))

                # Export argument of the output, angular rmse only
                if self.stopKey is 'armse':
                    self.argOutputs_h[sample] = np.copy(
                        self.neuronOutputs_h[-1])

                # Floor the argument to get the discrete output
                self.neuronOutputs_h[-1] = np.floor(
                    self.neuronOutputs_h[-1] / self.sectorSize_h)

            # Else if network outputs are continuous
            else:
                # Use the continuous activation function
                self.neuronOutputs_h[-1] = self.activateFunc(
                    self.neuronOutputs_h[-1])

            # Transfer output neuron values to network outputs container
            self.networkOutputs_h[sample] = self.neuronOutputs_h[-1]

        # Else compute on device
        else:
            if self.discreteOutput is True:
                self.neuronOutputs_d[-1] = (cumath.floor(
                    self.__gpu_mod__((cumath.log(
                        self.neuronOutputs_d[-1])).imag, self.twoPi_d)
                    / self.sectorSize_d.real)).astype(np.complex64)

                # Use scikits misc function to set the output elements
                # corresponding to current sample.
                # Operation might be memory inefficient.
                misc.set_by_index(self.networkOutputs_d, self.samplesIndex_d[sample],
                             self.neuronOutputs_d[-1], ind_which='dest')

            else:
                self.neuronOutputs_d[-1] = self.activateFunc(
                    self.neuronOutputs_d[-1])

                misc.set_by_index(self.networkOutputs_d, self.samplesIndex_d[sample],
                             self.neuronOutputs_d[-1], ind_which='dest')

    def calculate_neuronError(self, sample, layer):
        """
        Handles error calculation for the output layer and backpropagation of
        errors in the hidden and input layers
        """
        if self.useCuda is False:
            # Evaluate the output layer first
            if layer == self.numLayers_1:
                # Normalize the output values of the output neurons with the
                # continuous activation function
                self.neuronOutputs_h[layer] = self.activateFunc(
                    self.weightedSum_h[layer])

                # Global error for each output neuron equals a root of unity
                # corresponding to the difference between the desired output
                # and the normalized weighted sum for the same output neuron
                self.neuronErrors_h[layer] = (
                    (self.complex_desiredOutputs_h[sample] -
                     self.neuronOutputs_h[layer]) /
                    (self.networkSize_h[layer - 1] + 1))

            # Evaluating the hidden and input layers
            elif 0 <= layer < self.numLayers_1:
                # Evaluating the hidden layers
                if layer > 0:
                    self.neuronErrors_h[layer] = (
                        np.dot(self.neuronErrors_h[layer + 1] /
                               (self.networkSize_h[layer - 1] + 1),
                               self.networkWeights_h[layer + 1][:, 1:] ** -1))

                # Evaluating the input layer
                else:
                    self.neuronErrors_h[layer] = (
                        np.dot(self.neuronErrors_h[layer + 1] /
                               (self.inputsPerSample_h + 1),
                               self.networkWeights_h[layer + 1][:, 1:] ** -1))
            else:
                print 'Layer index out of bounds'

        # Computing on device
        else:
            if layer == self.numLayers_1:
                self.neuronOutputs_d[layer] = self.activateFunc(
                    self.weightedSum_d[layer])

                self.neuronErrors_d[layer] = (
                    (self.complex_desiredOutputs_d[sample] -
                     self.neuronOutputs_d[layer]) /
                    (self.networkSize_d[layer - 1] + 1))

            elif 0 <= layer < self.numLayers_1:
                if layer > 0:
                    self.neuronErrors_d[layer] = (culinalg.dot(
                        culinalg.transpose(
                            self.networkWeights_d[layer + 1].__pow__(-1)),
                        (self.neuronErrors_d[layer + 1] /
                         (self.networkSize_d[layer - 1] + 1))))

                    #self.neuronErrors_d[layer] = (culinalg.dot(
                        #(self.neuronErrors_d[layer + 1] /
                         #(self.networkSize_d[layer - 1] + 1)),
                        #culinalg.transpose(
                            #self.networkWeights_d[layer + 1].__pow__(-1))))

                else:
                    self.neuronErrors_d[layer] = (culinalg.dot(
                        culinalg.transpose(
                            self.networkWeights_d[layer + 1].__pow__(-1)),
                        (self.neuronErrors_d[layer + 1] /
                         (self.inputsPerSample_d + 1))))

                    #self.neuronErrors_d[layer] = (culinalg.dot(
                        #(self.neuronErrors_d[layer + 1] /
                         #(self.inputsPerSample_d + 1)),
                        #culinalg.transpose(
                            #self.networkWeights_d[layer + 1].__pow__(-1))))

    def activateFunc(self, array):
        """
        Continuous activation function for the complex-valued neurons
        """
        return array / array.__abs__()

    def correct_outputErrors(self):
        if self.useCuda is False:
            # If network output values are discrete
            if self.discreteOutput is True:
                self.outputErrors_h = (np.abs(self.networkOutputs_h -
                                            self.desiredOutputs_h))

                self.badOutput_h = (self.outputErrors_h > self.halfSectors_h)
                self.goodOutput_h = (
                    self.outputErrors_h <= self.halfSectors_h)

                if np.count_nonzero(self.badOutput_h) > 0:
                    self.outputMask_h = self.badOutput_h * self.numSectors_h

                    self.outputErrors_h = (
                        (self.outputErrors_h * self.goodOutput_h) +
                        (self.outputMask_h -
                            (self.outputErrors_h * self.badOutput_h)))

            # Else if network outputs are continuous
            else:
                self.outputErrors_h = (np.abs(np.mod(
                    np.angle(self.networkOutputs_h), self.twoPi_h) -
                    self.angular_desiredOutputs_h))

                self.badOutput_h = (self.outputErrors_h > np.pi)
                self.goodOutput_h = (self.outputErrors_h <= np.pi)

                if np.count_nonzero(self.badOutput_h) > 0:
                    self.outputMask_h = self.badOutput_h * self.twoPi_h

                    self.outputErrors_h = (
                        (self.outputErrors_h * self.goodOutput_h) +
                        (self.outputMask_h -
                            (self.outputErrors_h * self.badOutput_h)))

            # Reset the soft margins mask
            self.outputMask_h = np.zeros_like(self.outputMask_h)

        # Else compute on devices
        else:
            if self.discreteOutput is True:
                self.outputErrors_d = (
                    self.networkOutputs_d - self.desiredOutputs_d).__abs__()

                self.badOutput_d = (self.outputErrors_d > self.halfSectors_d)
                self.goodOutput_d = (self.outputErrors_d <= self.halfSectors_d)

                # if_positive function elementwise checks if the current
                # element of the badOutput array > 0, and calculates the
                # adjustment factor for those elements
                self.outputMask_d = gpuarray.if_positive(
                    self.badOutput_d, (self.badOutput_d * self.numSectors_d),
                    self.outputMask_d)

                self.outputErrors_d = gpuarray.if_positive(
                    self.badOutput_d, (
                        (self.outputErrors_d * self.goodOutput_d) +
                        (self.outputMask_d -
                         (self.outputErrors_d * self.badOutput_d))),
                    self.outputErrors_d)

            # Continuous output error calculation
            else:
                self.outputErrors_d = ((cumath.fmod(
                    (cumath.log(self.networkOutputs_d).imag),
                    self.twoPi_d)).__abs__())

                self.badOutput_d = (self.outputErrors_d > self.pi_d)
                self.goodOutput_d = (self.outputErrors_d <= self.pi_d)

                self.outputMask_d = gpuarray.if_positive(
                    self.badOutput_d, (self.badOutput_d * self.twoPi_2_d),
                    self.outputMask_d)

                self.outputErrors_d = gpuarray.if_positive(
                    self.badOutput_d, (
                        (self.outputErrors_d * self.goodOutput_d) +
                        (self.outputMask_d -
                         (self.outputErrors_d * self.badOutput_d))),
                    self.outputErrors_d)

            self.outputMask_d = gpuarray.zeros_like(self.outputMask_d)

    def calculate_angularError(self):
        """
        Angular RMSE only: calculate the angular error over all samples
        """

        if self.useCuda is False:
            # Angular network error calculation
            self.networkErrors_h = (np.mod(
                np.abs(self.angular_desiredOutputs_h - self.argOutputs_h),
                self.twoPi_h))

            self.networkErrors_h = (np.sum(
                self.networkErrors_h, axis=1, keepdims=True) /
                self.numOutputs_h)

        else:
            # Placeholder for GPU method
            pass

    def calculate_networkError(self, sample):
        """
        Calculate the global error for the network, based on the soft
        margin errors
        """
        # Compute on host
        if self.useCuda is False:
            self.networkErrors_h[sample] = (
                np.sum(np.power(self.outputErrors_h, 2)) / self.numOutputs_h)

        # Else compute on device
        else:
            error = (((gpuarray.sum(
                (self.outputErrors_d.__pow__(2))) /
                self.numOutputs_d).astype(np.complex64)).reshape(1, 1))

            misc.set_by_index(self.networkErrors_d,
                              np.array([sample], dtype=np.int32), error)

    def check_globalError(self):
        """
        Check if the global error exceeds the global threshold value
        """
        if self.useCuda is False:
            # Special case: maximum error algorithm skips global error calc.
            if self.stopKey is 'max' and self.iterations == 1:
                return False

            # Special case: angular rmse with soft margins
            elif self.stopKey is 'armse':

                if (self.globalError_h.real <= self.globalThreshold_h and
                        np.count_nonzero(self.max_outputErrors_h) == 0):
                    return True

            # Everything else is evaluated normally
            else:
                if self.globalError_h.real <= self.globalThreshold_h:
                    return True
        else:
            if self.stopKey is 'max' and self.iterations == 1:
                return False
            # This code is very memory inefficient, needs to be replaced somehow
            # How, I do not know

            if (self.globalError_d.real <= self.globalThreshold_d)[0][0].get():
                return True
        return False

    def check_localError(self, sample):
        """
        Check if the local error for the current sample exceeds the
        local threshold value

        Note: function only accepts a sample index argument when calculating
        error for angular rmse with soft margins
        """

        if self.useCuda is False:
            if self.stopKey is 'armse':
                if (self.networkErrors_h[sample].real <= self.localThreshold_h
                        and self.localError_h == 0):
                    return True

            else:
                if self.localError_h.real <= self.localThreshold_h:
                    return True

                else:
                    # For the max error algorithm, increment the error count for
                    # the current iteration
                    if self.stopKey == 'max':
                        self.globalError_h += 1

            return False

        else:
            if (self.localError_d.real <= self.localThreshold_d)[0][0].get():
                return True

            else:
                if self.stopKey == 'max':
                    self.globalError_d += 1

                return False

    def calculate_globalError(self):
        """
        Calculate the global error for the network over all samples, then
        compare to the global threshold values and modify learning flag
        if necessary
        """

        # Compute on host
        if self.useCuda is False:
            if self.stopKey is 'error':
                # Simple error rate
                self.outputErrors_h = (self.outputErrors_h > self.localThreshold_h)

                self.networkErrors_h = (np.max(
                    np.sum(self.outputErrors_h, axis=self.errorRate_axis)))

                self.globalError_h = (100.0 * self.networkErrors_h /
                                      self.numSamples_h)

                print 'Error Rate = {0:.2f}%'.format(self.globalError_h)

            elif self.stopKey is 'armse':

                # Get maximum error value for all output neurons over all samples
                self.max_outputErrors_h = (np.max(
                    np.abs(self.networkOutputs_h - self.desiredOutputs_h),
                    axis=1, keepdims=True))

                # Calculate angular rmse over all samples
                self.globalError_h = (np.sqrt(np.sum(
                    np.power(self.networkErrors_h, 2)) / self.numSamples_h))

                #print 'ARMSE = {0:.4f}'.format(self.globalError_h)

            elif self.stopKey is 'mse' or 'rmse':

                self.networkErrors_h = (
                    np.sum(np.power(self.outputErrors_h, 2), axis=1) / self.numOutputs_h)

                # Calculate Mean-Square Error
                self.globalError_h = (np.sum(
                    self.networkErrors_h) / self.numSamples_h)

                # Calculate Root Mean-Square Error
                if self.stopKey is 'rmse':
                    self.globalError_h = np.sqrt(self.globalError_h)

            elif self.stopKey is 'max':
                # absolute max error, just a placeholder
                print 'Error Count = {0:.2f}%'.format(self.globalError_h)

        # Else compute on device
        else:
            if self.stopKey is 'error':
                # Simple error rate, not implemented yet
                print 'Not there yet'

            elif self.stopKey is 'armse':
                # Angular root-mean square error, not implemented yet
                print 'Not there yet'

            elif self.stopKey is 'mse' or 'rmse':

                #error = (((gpuarray.sum(
                    #(self.softErrors_d.__pow__(2)) /
                    #self.numOutputs_d)).astype(np.complex64)).reshape(1, 1))

                #set_by_index(self.networkErrors_d,
                            #np.array([sample], dtype=np.int32), error)

                #self.networkErrors_d = (
                    #misc.sum(self.softErrors_d.__pow__(2),
                        #axis=1, keepdims=True) / self.numOutputs_d)

                self.networkErrors_d = (
                    misc.mean(self.outputErrors_d.__pow__(2),
                              axis=1, keepdims=True))

                #self.networkErrors_d = (((
                    #gpuarray.sum((self.softErrors_d.__pow__(2))) /
                                 #self.numOutputs_d)))

                # Calculate Mean-Square Error
                self.globalError_d = (misc.mean(
                    self.networkErrors_d, keepdims=True))

                #self.globalError_d = ((gpuarray.sum(
                    #self.networkErrors_d)).reshape(1, 1) / self.numSamples_d)

                # Calculate Root Mean-Square Error
                if self.stopKey is 'rmse':
                    self.globalError_d = cumath.sqrt(self.globalError_d)
                    #print 'RMSE = {0:.4f}'.format(self.globalError_d.get()[0][0])

            elif self.stopKey is 'max':
                pass

    def calculate_localError(self, sample):
        if self.useCuda is False:
            self.localError_h = np.max(self.outputErrors_h[sample])

        else:
            self.localError_d = (gpuarray.max(
                self.outputErrors_d[sample]).reshape(1, 1))

    def learn(self):
        """
        Function calls learning cycle method until finished learning - only
        used by the CLI version of the program
        """

        if self.__learning__ is False:
            self.__learning__ = True
            self.finishedLearning = False
            self.terminate = False
            self.iterations = 0

        # Initialize the default system timer; only really needed for Windows,
        # but doesn't do any harm
        timeit.default_timer()

        # While there is still work to do, keep learning
        while self.checkContinue():
            self.learnCycle()

        # Send a message that the job is done
        self.__doneMessage__()

        # Reset the learning flag
        self.__learning__ = False

    def learnCycle(self):
        """
        When called, executes a single learning iteration based on the
        given network parameters
        """

        # Temporary timing code
        start = timeit.default_timer()

        self.iterations += 1

        # Initial output and global error calculation - skip if using
        # max stopping criteria
        if self.stopKey is not 'max':
            # Calculate neuron and network outputs
            for sample in xrange(self.numSamples_h):
                for layer in xrange(self.numLayers):
                    # Outputs for all neurons
                    self.calculate_neuronOutputs(sample, layer)

                # Outputs of the network
                self.calculate_networkOutputs(sample)

            # If desired, use the soft margins error correction
            # to adjust the errors for each input sample
            if self.stopKey is not 'armse':
                self.correct_outputErrors()

            else:
                self.calculate_angularError()

        # Get the specified global error, check if global error exceeds
        # defined global threshold value
        self.calculate_globalError()

        # Finish learning if the global error is less than the threshold
        # value - else continue working
        self.finishedLearning = self.check_globalError()

        # Send a report on global error
        self.__action_sendMetrics__()

        if not self.checkContinue():
            # Return to while loop
            return

        # If further learning is required, calculate the weighted sums
        else:
            for sample in xrange(self.numSamples_h):
                for layer in xrange(self.numLayers):
                    self.calculate_neuronOutputs(sample, layer)

                # Outputs of the network
                self.calculate_networkOutputs(sample)

                # Special case: angular rmse calculate angular error
                if self.stopKey is 'armse':
                    self.calculate_angularError()

                # Soft margin errors for the new outputs
                self.correct_outputErrors()

                # Get the largest error from the soft margin errors
                self.calculate_localError(sample)

                # If the network error exceeds the local threshold,
                # keep working
                if not self.check_localError(sample):
                    # Calculate errors of all neurons for current sample
                    # using the error backpropagation rule
                    for layer in xrange(self.numLayers_1, -1, -1):
                        self.calculate_neuronError(sample, layer)

                    # Now correct the neuron weights for each layer
                    for layer in xrange(self.numLayers):
                        self.correct_networkWeights(sample, layer)

        # Temporary timing code
        end = timeit.default_timer()
        self.timing += end - start

        # Cycle complete, return control to while loop
        return

    def learnOld(self):
        """
        Old implementation of the learning algorithm, not GUI-friendly or
        thread safe. May be a little faster than the newer algorithm
        """

        # Set learning indicator flag
        self.__learning__ = True

        self.iterations = 0

        self.finishedLearning = False
        self.terminate = False

        while not self.finishedLearning:

            # If termination flag is set or iteration limit is reached,
            # break learning loop
            if not self.checkContinue():
                break

            self.iterations += 1  # Increment the number of learning cycles
            #self.errorCounter = 0

            # Initial output and global error calculation - skip if using
            # max stopping criteria
            if self.stopKey is not 'max':
                # Calculate neuron and network outputs
                for sample in xrange(self.numSamples_h):
                    for layer in xrange(self.numLayers):
                        # Outputs for all neurons
                        self.calculate_neuronOutputs(sample, layer)

                    # Outputs of the network
                    self.calculate_networkOutputs(sample)

                # If desired, use the soft margins error correction
                # to adjust the errors for each input sample
                if self.stopKey is not 'armse':
                    self.correct_outputErrors()

                else:
                    self.calculate_angularError()

            # Get the specified global error, check if global error exceeds
            # defined global threshold value
            self.calculate_globalError()

            # Finish learning if the global error is less than the threshold
            # value - else continue working
            self.finishedLearning = self.check_globalError()
            #errorCounter = np.count_nonzero(self.max_outputErrors_h)

            #print 'Errors: ', errorCounter

            #errorCounter = 0

            # If further learning is required, calculate the weighted sums
            if not self.finishedLearning:
                #print 'local'
                for sample in xrange(self.numSamples_h):
                    for layer in xrange(self.numLayers):
                        self.calculate_neuronOutputs(sample, layer)

                    # Outputs of the network
                    self.calculate_networkOutputs(sample)

                    # Special case: angular rmse calculate angular error
                    if self.stopKey is 'armse':
                        self.calculate_angularError()

                    # Soft margin errors for the new outputs
                    self.correct_outputErrors()

                    # Get the largest error from the soft margin errors
                    self.calculate_localError(sample)

                    # If the network error exceeds the local threshold,
                    # keep working
                    if not self.check_localError(sample):
                        #errorCounter += 1

                        # Calculate errors of all neurons for current sample
                        # using the error backpropagation rule
                        for layer in xrange(self.numLayers_1, -1, -1):
                            self.calculate_neuronError(sample, layer)

                        # Now correct the neuron weights for each layer
                        for layer in xrange(self.numLayers):
                            self.correct_networkWeights(sample, layer)

            # Send a report
            self.__action_sendMetrics__()
        # self.timing /= self.iterations

        if self.useCuda is True:
            self.networkOutputs_h = self.networkOutputs_d.get()

        # Reset termination flag, if necessary
        self.terminate = False

        self.__learning__ = False

        return

    def test(self):
        """
        Tests the classification accuracy of the current network weights
        """

        # Set bool flag for testing mode
        self.__testing__ = True

        for sample in xrange(self.numSamples_h):
            for layer in xrange(self.numLayers):
                self.calculate_neuronOutputs(sample, layer)

            self.calculate_networkOutputs(sample)

        for sample in xrange(self.numSamples_h):
            self.correct_outputErrors()

        if self.useCuda is True:
            self.outputErrors_h = self.outputErrors_d.get()

        if self.numOutputs_h > 1:
            self.networkErrors_h = np.max(self.outputErrors_h, axis=0)

        else:
            self.networkErrors_h = self.outputErrors_h

        indicator = (self.networkErrors_h > self.localThreshold_h)
        self.AbsErrors = np.sum(indicator).astype(np.float32)
        self.Accuracy = 100.0 - (self.AbsErrors/self.numSamples_h) * 100

        if self.numOutputs_h > 1:
            self.networkErrors_h = (np.sum(self.outputErrors_h ** 2, axis=0) /
                                    self.numOutputs_h)

        else:
            self.networkErrors_h = self.outputErrors_h ** 2

        mse = np.sum(self.networkErrors_h) / self.numSamples_h
        self.rmse = np.sqrt(mse)

        self.__action_sendMetrics__()

        self.__testing__ = False

        self.__doneMessage__()

        return

    def filter(self):
        """
        Applies current weights to all input samples and returns output values
        without further analysis - useful for filtering images
        """

        # Set bool flag for testing mode
        self.__testing__ = True

        for sample in xrange(self.numSamples_h):
            for layer in xrange(self.numLayers):
                self.calculate_neuronOutputs(sample, layer)

            self.calculate_networkOutputs(sample)

        self.__testing__ = False

        return self.__getOutputs__()

