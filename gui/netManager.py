import sys

import mlmvn

from collections import deque
from PySide import QtCore

import numpy as np
from scipy.io import savemat


class manager(QtCore.QObject):

    dataUpdated = QtCore.Signal(dict)

    # Thread status signal emits a network index and the currently running
    # command of the network
    threadStatus = QtCore.Signal(object)
    stopNet = QtCore.Signal(int)

    def __init__(self, numNetworks, netArgs):
        """
        Network manager constructor. By default, one network instance is
        initialized

        Arguments
            numNetworks:    number of networks to initialize, defaults to at
                            least 1 network instance

            netArgs:        dictionary with network instance index as
                            key, dictionary containing actual arguments
                            as value - if not passed, constructor will
                            append Null values to the network list
        """

        QtCore.QObject.__init__(self)

        # netList is a python dictionary with network indices as keys, and
        # a nested dictionary as values
        self.netList = {}

    def addNetwork(self, netIndex, netParams):
        """
        Add a single new network instance to the manager container

        Arguments
            netIndex:   index of the network instance to be added
            netArgs:    dictionary with network function arguments as
                        keys, actual network parameters as values
        """

        self.netList[netIndex] = {
            # Copy the passed simulator parameters
            'args': netParams.copy(),
            # errorData is a numpy array that holds the global error data,
            # default starting size is 1000 elements (i.e. iterations)
            'errorData': np.empty(1000, dtype=np.float32),
            # iters holds the total number of iterations (duh)
            'iter': 0,
            # Create a new worker thread with specified parameters
            'thread': thread_simWorker(netIndex, netParams)
        }

        # Every time the thread emits new data, the manager will update
        # the data container for the thread's network
        self.netList[netIndex]['thread'].sendData.connect(
            self.updateData)

        # Every time the state of the thread (e.g. learning, testing, stopping)
        # changes, the manager will emit a signal to the GUI for that network
        self.netList[netIndex]['thread'].currentStatus.connect(
            self.action_updateStatus)

        # stopNet is used to stop a running process
        self.stopNet.connect(self.netList[netIndex]['thread'].stopProcess)

        # When a thread stops running, it emits a finished signal and causes
        # the manager to delete the thread
        self.netList[netIndex]['thread'].finished.connect(self.delThread)

    def delThread(self):
        """
        Utility function that deletes a worker thread once it has exited its
        event loop. Should be called only if the finished signal is emitted from
        a worker thread - should not be called from outside (seg faults)
        """

        # Get the netIndex for the sending thread
        netIndex = self.sender().index

        # If the thread is no longer running, delete it and all related data
        if self.netList[netIndex]['thread'].isFinished():
            # Delete the thread
            del self.netList[netIndex]['thread']

            # Delete all network data
            del self.netList[netIndex]

        print 'Successfully deleted thread ', netIndex

    def delNetwork(self, netIndex):
        """
        Deletes a network instance with the specified index and
        exits the thread event loop to prepare the thread for
        deletion.
            <`0`>
        """

        if netIndex in self.netList:
            self.action_Stop(netIndex)
            self.netList[netIndex]['thread'].addCommand('deleting')

    def getData(self, netIndex):
        """
        Returns the current iteration and global error array for
        specified network instance
        """

        currentIter = self.netList[netIndex]['iter']
        currentError = self.netList[netIndex]['errorData']

        if netIndex in self.netList:
            return currentIter, currentError

        else:
            return None, None

    def updateData(self, data):
        """
        Helper function, receives data from the worker thread and saves it in
        an error data array that corresponds to the sending thread

        Arguments
            data:   python dictionary containing iterations, error data, and an
                    index value for the network instance
        """

        netIndex = self.sender().index

        currentIter = data['iter']
        currentError = data['error']
        currentRange = self.netList[netIndex]['errorData'].shape[0]

        refreshLimit = self.netList[netIndex]['args']['refreshLimit']


        self.netList[netIndex]['iter'] = currentIter
        # Store the value of the global error for the current iteration
        self.netList[netIndex]['errorData'][currentIter] = currentError

        # If the array for error data is too small, make it bigger
        if (currentIter >= (currentRange - refreshLimit)):
            tmp = self.netList[netIndex]['errorData']

            print 'Expanding array for index ', netIndex
            # Double the size of the error data array
            self.netList[netIndex]['errorData'] = np.empty(
                self.netList[netIndex]['errorData'].shape[0] * 2)

            # Fill the resized array (partially) with the previous data
            self.netList[netIndex]['errorData'][:tmp.shape[0]] = tmp

        # Send a signal to the GUI letting it know error data has updated
        self.dataUpdated.emit(netIndex)

    def updateNetwork(self, netIndex, netParams):
        """
        Helper function, re-initializes the specified network instance
        with the specified network arguments. The network is only modified
        if the incoming parameters are different than those on record for
        the specified net instance

        Arguments
            netIndex:       the index of the network instance to be
                            re-initialized

            netParams:      dictionary with network initialization
                            arguments as keys, argument values as values
        """

        # If the passed parameters don't match those stored for this
        # network instance
        if netParams != self.netList[netIndex]['args']:

            # Replace the old parameters
            self.netList[netIndex]['args'].update(netParams)

            if self.action_getStatus(netIndex) == 'stopped':

                self.netList[netIndex]['thread'].initNet(netParams)

                self.netList[netIndex]['errorData'] = np.empty(
                    1000, dtype=np.float32)

                self.netList[netIndex]['iter'] = 0

    def action_Learn(self, netIndex):
        """
        Enqueues a learning command for the given network instance into
        the worker thread
        """

        self.netList[netIndex]['thread'].addCommand('learning')

    def action_Test(self, netIndex):
        """
        Enqueues a testing command for the given network instance into
        the worker thread
        """

        self.netList[netIndex]['thread'].addCommand('testing')

    def action_Stop(self, netIndex):
        """
        Sends a signal to the specified network to stop processing
        """
        self.stopNet.emit(netIndex)

    def action_Export(self, netIndex, fileName):
        """
        Enqueues an export weights command for the given network instance into
        the worker thread

        Arguments
            netIndex:       integer index of particular network instance
            fileName:       string, name of file to be exported
        """

        self.netList[netIndex]['thread'].addCommand('exporting', fileName)

    def action_Randomize(self, netIndex):
        """
        Enqueues a weights randomization command for the given network instance
        into the worker thread
        """

        self.netList[netIndex]['thread'].addCommand('randomizing')

    def action_getStatus(self, netIndex):
        """
        Returns the current status of the specified thread, or
        None value if the thread does not exist
        """

        if netIndex in self.netList:
            return self.netList[netIndex]['thread'].status

        else:
            return None

    def action_updateStatus(self, netIndex=None):
        """
        Notifies the GUI about the current status of a network instance

        Arguments
            netIndex:       integer index assigned to each network instance

        Returns
            (1) If a netIndex was provided, it is assumed the command is coming
            from the GUI, in which case the function returns a tuple with
            the index and the current status of the simulator with that index.
            If there is no match for the specified index, a None value is
            returned

            (2) If a netIndex was not passed, the call came from a signal in a
            worker thread after a state change - the function then uses a signal
            to pass the index and status of the signaling thread to the GUI
        """

        # If index is not passed, the signal is coming from a thread
        if netIndex == None:
            self.threadStatus.emit(self.sender().index)


class thread_simWorker(QtCore.QThread):
    """
    Worker thread for the mlmvn simulator. Each thread is responsible
    for evaluating a single network instance. Commands are passed
    from the network manager and enqueued in a FIFO queue, which
    is evaluated in a running event loop. Once all enqueued commands
    are completed, the thread will shut down until new commands are
    received. When thread object is destroyed, the event loop exits
    gracefully.
    """

    # Current state signal is emitted whenever a command is invoked in
    # the simulator or the command finishes execution
    currentStatus = QtCore.Signal()

    # Send data signal is emitted during every learning cycle to transfer
    # simulation data to the GUI
    sendData = QtCore.Signal(dict)

    # initialization success signal is emitted if the mlmvn simulator object for
    # this thread is created without errors
    initSuccess = QtCore.Signal(bool)


    def __init__(self, netIndex, netArgs):
        """
        Constructor for worker thread

        Arguments
            netArgs:    python dictionary, contains parameters for the
                        network instance
        """
        # Initialize the thread using its base constructor
        QtCore.QThread.__init__(self)

        # network is the actual mlmvn simulator object
        self.network = None

        # Keep processing flag is used to loop the next command until some
        # condition is met (e.g. learning process is completed)
        self.keepProcessing = False

        # Index is the same as the net Instance index: an integer key that
        # identifies a unique network instance
        self.index = netIndex
        # Initialize queue that holds commands to be processed
        self.simQueue = deque()
        # Status flag reflects the current state of the simulation
        self.status = 'stopped'
        # nextCommand holds the next simulator command to be executed, as well
        # as a filename if it's the export weights command
        self.nextCommand = None

        # commandAdded and mutex are used to synchronize this thread with the
        # main GUI thread
        self.commandAdded = QtCore.QWaitCondition()
        self.mutex = QtCore.QMutex()

        # Initialize the mlmvn simulator object
        self.initNet(netArgs)

        self.finished.connect(self.deleteLater)

        # If a time limit was passed
        if netArgs['timeLimit']:
            # Convert passed time limit to milliseconds
            timeLimit = QtCore.QTime(0, 0, 0, 0).msecsTo(
                QtCore.QTime.fromString(netArgs['timeLimit'], 'hh:mm'))

            # QTimer is used if a time limit was passed as a network parameter
            self.stopTimer = QtCore.QTimer()
            self.stopTimer.setInterval(self.timeLimit)

            # Connect the timer to the stopProcess method
            self.stopTimer.timeout.connect(self.stopProcess)

            print 'created timer'
        else:
            self.stopTimer = None

        # Start the thread's event loop
        #self.start()

    def __del__(self):
        """
        Destructor for the worker thread, makes sure that it exits event
        loop gracefully by enqueuing a destructor flag
        """

        locker = QtCore.QMutexLocker(self.mutex)

        # If there are still commands in the queue, clear it
        if len(self.simQueue) != 0:
            self.simQueue.clear()

        self.quit()
        self.wait()

        # If a simulator job is running, stop it
        #self.stopProcess()
        # Put bool deconstructor flag in the queue
        #self.simQueue.append({'command': 'deleting', 'fileName': None})

        #print 'deleting ', self.index
        #self.addCommand('deleting')

        #self.wait()

        # Wake a worker thread and finish the run method
        #self.commandAdded.wakeOne()

        #print self.isFinished()

    def initNet(self, netArgs):
        """
        Initializes an mlmvn object to use for the simulation. Called during
        thread initialization or when the network parameters are modified
        """
        # Try to initialize the network - if initialization fails, a failure
        # signal is emitted. Otherwise the constructor emits a success
        # signal and continues work
        try:
            # If the network object has been initialized already, delete it
            if self.network is not None:
                del self.network
            # Initialize a network manager object
            self.network = networkGUI(netArgs)
            self.initSuccess.emit(True)
        except:
            print 'Could not initialize network: ', sys.exc_info()[1]
            self.initSuccess.emit(False)
            raise

        # Connect signals in the network object to appropriate slots in
        # the thread so the network can send data to the GUI and let the
        # thread know when it completes a task
        self.network.signalMetrics.connect(self.relayMetrics)
        self.network.taskComplete.connect(self.stopProcess)
    def relayMetrics(self, data):
        """
        Helper function, receives data from the currently running network and
        passes it along to the network manager

        Arguments
            data:   a python dictionary containing iterations, error data
        """

        # Add network instance value of the currently running command to data
        # packet and pass it along
        self.sendData.emit(data)

    def addCommand(self, command, fileName=None):
        """
        Enqueue a new command to the worker thread queue
        """

        locker = QtCore.QMutexLocker(self.mutex)

        # Only enqueue a new command if the thread isn't working on an
        # identical command
        #if command != self.status:
            # Enqueue new command
        self.simQueue.append({'command': command, 'fileName': fileName})

        # If the thread has not started yet, start it
        if not self.keepProcessing and not self.isRunning():
            self.keepProcessing = True
            self.start()

        self.keepProcessing = True

        # Wake the worker thread
        self.commandAdded.wakeOne()

    def run(self):
        """
        Event loop for worker thread
        """

        self.nextCommand = None
        self.keepProcessing = True

        # Enter event loop
        while True:
            # Lock the thread resources
            locker = QtCore.QMutexLocker(self.mutex)

            # If there are no more commands in the queue, unlock the mutex
            # and wait for a command to be added
            if (self.status == 'stopped' and len(self.simQueue) == 0):
                self.keepProcessing = False
                self.commandAdded.wait(self.mutex)

            # Get the next command from the queue
            self.nextCommand = self.simQueue.popleft()

            # Update status to reflect next command to be run
            self.status = self.nextCommand['command']

            # Let the GUI know that the next command has started
            self.getStatus()

            # If next command comes from the destructor method, break the
            # run loop, delete the network, and exit
            if self.status == 'deleting':
                # Delete the network object and break the loop
                del self.network
                break

            # If a time limit was passed, start the timer only if the next
            # command is a learning task
            if self.stopTimer and self.status == 'learning':
                self.stopTimer.start()

            # stopped flag acts as a check for the learning algorithm's event
            # loop; each iteration of the loop is a single learning cycle. For
            # non-iterative simulator commands, this loop only iterates once
            if self.keepProcessing:
                # Run the next command in queue
                self.network.run(**self.nextCommand)

            # If the next command finishes before the timer times out,
            # stop the timer
            if self.stopTimer and self.stopTimer.isActive():
                self.stopTimer.stop()

            # Unlock the mutex
            locker.unlock()
            # Check if all requested commands are completed
            if len(self.simQueue) == 0:
                # Special case: let the manager know that the thread is stopped
                self.status = 'stopped'
                self.keepProcessing = False

                self.getStatus()

                # Reset the simulator status flags
                self.network.__reset__()

    def stopProcess(self, netIndex=None):
        """
        Tells the network simulator to stop working - if the index value passed
        by the incoming signal matches the index of this thread object or
        if no index was passed (in which case it's assumed the signal came
        from the network object itself)
        """

        # Break the process loop if the signal comes from the correct
        # network, or if no index was provided
        if ((netIndex == self.index or netIndex == None) and
                (self.status != 'stopped')):
            # Set the keepProcessing flag to False so the next command won't
            # keep looping
            self.keepProcessing = False
            self.network.__stop__()

            # Set the simulator status to stopping
            self.status = 'stopping'

        # Regardless of if the keep processing flag was unset or not, send an
        # update of the current status of the simulator
        self.getStatus()

    def getStatus(self, netIndex=None):
        """
        Emits the current status/command of the simulator
        """

        # If netIndex equals the index of this thread, the signal
        # came from the GUI. If netIndex is None, it came from this thread
        if netIndex == self.index or netIndex == None:
            self.currentStatus.emit()


class networkGUI(QtCore.QObject, mlmvn.network):
    """
    MLMVN simulator object, inherits QObject properties to allow
    transmission of data to network manager
    """

    signalMetrics = QtCore.Signal(dict)
    cudaError = QtCore.Signal()
    taskComplete = QtCore.Signal()

    def __init__(self, netArgs=None, parent=None):
        """
        Constructor - initializes network object with given netArgs

        Arguments
            netArgs:    python dictionary with network function arguments as
                        keys, parameters as values
            parent:     QObject that is the parent/owner of the network instance
        """

        QtCore.QObject.__init__(self)

        # First initialize the network
        mlmvn.network.__init__(self, **netArgs)

        self.terminate = False

        self.taskComplete.connect(self.__reset__)

    def initCuda(self):
        """
        Imports and initializes the libraries necessary to utilize CUDA,
        reimplemented for use with GUI mlmvn
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
            self.cudaError.emit()


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
            self.cudaError.emit()

    def __action_sendMetrics__(self):
        """
        Overloaded helper function, emits a signal to send
        to the network manager
        """

        # Learning case - send a report per number of iterations
        if self.__learning__ and self.iterations % self.refreshLimit == 0:
            # If using CUDA, transfer global error from device to host
            if self.useCuda is True:
                self.globalError_h = self.globalError_d.get()[0][0]

            # Emit signal to GUI
            self.signalMetrics.emit({
                'iter': self.iterations,
                'error': self.globalError_h
            })

        # Testing case
        elif self.__testing__:
            self.signalMetrics.emit(
                {'numErrors': self.AbsErrors,
                    'accuracy': self.Accuracy,
                    'rmse': self.rmse})

    def __doneMessage__(self):
        """
        Reimplemented from parent mlmvn class, just emits a signal letting
        the GUI know the running task has completed
        """

        self.taskComplete.emit()

    def run(self, command, fileName=None):

        if command == 'learning':
            self.learn()

        elif command == 'testing':
            self.test()

        elif command == 'randomizing':
            self.__randomizeWeights__()

        elif command == 'exporting' and fileName is not None:
            self.exportWeights(fileName)
