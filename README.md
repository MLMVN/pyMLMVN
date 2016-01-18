# pyMLMVN - a simulator for MLMVN in Python
### Note - Please do not confuse with [MiraHead’s excellent but unaffiliated PyMLMVN](https://github.com/MiraHead/mlmvn)

## Goals
* Implement a functional simulator for Multi-Layered neural networks with Multi-Valued Neurons (MLMVN)
* Leverage the computational power of general-purpose graphics processing units (GPGPUs) to decrease simulator run-time
* Provide a convenient interface to the MLMVN that is conducive to programming multiple simulations & experiments
* Allow users to manage several network configurations and simulation instances with an easy-to-use GUI
* Provide a variety of run-time metrics to monitor network performance

## System Requirements

### Core Requirements
* Python 2.7
* [SciPy](http://www.scipy.org/)
### CUDA Requirements
* [CUDA Framework (>= 6.5)](http://www.nvidia.com/object/cuda_home_new.html)
* [pycuda (>= 2014.1)](http://mathema.tician.de/software/pycuda/)
* [scikit-cuda (>= 0.5.0a3)](https://github.com/lebedov/scikit-cuda)
### GUI Requirements
* [Qt 4 Framework](http://www.qt.io/)
* [PySide (>= 1.2.2)](https://pypi.python.org/pypi/PySide/1.2.4)
* [PyQtGraph (>= 0.9.1)](http://www.pyqtgraph.org/)

## Basic Use - Training a network

```Python
import mlmvn
testNet = mlmvn.network(PARAMETERS HERE)
testNet.learn()
```
 The above is a simplified example of how to invoke the learning algorithm of MLMVN from a Python interpreter. As of this writing, the `mlmvn` module contains a single object class called `network`. Please see the network parameters section for a complete list of required simulator arguments.

### `network` Simulator Commands

1. learn(): invokes the learning algorithm, using the simulation parameters passed during network initialization
2. test(): test classification of network using current weights
3. filter(): calculates and returns network outputs without calculating global error
4. exportWeights(outputFile=None): exports the current network weights to a `.mat` file, file name specified either at network initialization or as a function argument

### `network` Initialization Parameter List
1. inputName: string or `numpy.ndarray`, dataset used for the supervised learning algorithm
  - a string will be interpreted as a path for a `.txt` file and will be imported with the `np.loadtxt` method
  - argument is assumed to contain both input and desired output values
2. outputName: string, filename to be used to save current network weights
3. netSize: list of integers, topology for the MLMVN
  - For example, a network with 3 hidden layers of 100 neurons each, and an output layer with 6 neurons, will need the argument [100, 100, 100, 6]
4. discInput: boolean, denotes whether the passed input values are discrete or continuous (non-rational) numbers
5. discOutput: ditto, but for the passed output values
6. globalThresh: integer/float value, minimum threshold classification error value for the network over all learning samples
7. localThresh: ditto, but for the per-sample network classification error
8. sectors: integer, number of equidistant sectors to divide the complex unit circle into, discrete outputs only
  - in general, this should be equal to the number of unique classification features from the learning set, although you may need to experiment to get optimal results
9. weightKey: string, determines the initial network weights. Acceptable arguments are:
  - `‘random’`: network will pseudo-randomize the network weights with the normal distribution based on the provided topology
  - Ends with`’.mat’`: network will assume a path to a MATLAB file containing a cell array, the dimensions of which match exactly those of the provided topology (this may change in the future)
10. stopKey: string, denotes the type of statistical global network error that is used for the learning algorithm. Acceptable arguments are:
  - `’error’`: simple error rate, calculated as a percentage of the incorrect network outputs
  - `’max’` : maximum number of absolute network errors
  - `’mse’` : mean square error
  - `’rmse’` : root mean square error
  - `’armse’` : angular root mean square error, used only for continuous desired output values
11. softMargins: boolean, determines whether the soft margins method is used,  angular rmse only. As of this writing, it doesn’t  really do anything, so just leave it to its default value, `True`.
12. cuda: boolean, setting to `True` will make the simulator use GPGPU acceleration via the CUDA framework
13. iterLimit: integer, determines the maximum number of iterations the simulator will run.
  - set to `None` (default) or 0 to disable
14. refreshLimit: integer, determines how many iterations before the simulator prints an update of the global error
  - set to `None` or 0 to disable, default is 1 (updates every iteration)
15. timeLimit: integer, number of seconds (by system clock) the simulator’s learning algorithm will run before quitting



