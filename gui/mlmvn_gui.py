import sys

sys.path.insert(0, './../common/')

import numpy as np
import re

from PySide import QtGui, QtCore

from ui_mainWindow import Ui_MainWindow

from netManager import manager


class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.outputName = None

        self.netCount = -1

        # Python dictionary keeps track of all parameters
        self.netParams = {}

        QtCore.QCoreApplication.setOrganizationName('Texas A&M Texarkana')
        QtCore.QCoreApplication.setApplicationName('Project Gondor')

        self.createGraph()

        self.createButtons()
        self.createActions()
        self.createStatusBar()

        self.createManager()
        self.readSettings('test.ini')

        self.show()

    def createManager(self):
        """
        #Initialize a network manager object
        """

        numNetworks = len(self.netButtonGroup.buttons())

        self.Manager = manager(numNetworks, self.netParams)

        self.Manager.dataUpdated.connect(self.action_updateCurve)
        self.Manager.threadStatus.connect(self.action_checkStatus)

    def closeEvent(self, event):
        """
        Called when the main window is closed, saves current network settings
        """

        self.killAll()

        self.writeSettings()
        event.accept()

    def killAll(self):
        """
        Kills all running threads
        """

        allNetworks = self.netButtonGroup.buttons()
        numNetworks = len(allNetworks)

        for i in xrange(numNetworks):

            self.Manager.delNetwork(i)

    def __updateNetLabels__(self):
        """
        Helper function, update the labels and button group id's
        of existing buttons
        """

        allNetworks = self.netButtonGroup.buttons()
        numButtons = len(allNetworks)

        # Update the labels of all net instance buttons
        for i in xrange(numButtons):
            currentKey = self.netButtonGroup.id(allNetworks[i])
            allNetworks[i].setText(self.tr(str(i + 1)))
            allNetworks[i].setToolTip(str(self.netParams[currentKey]))

        # If the add network button is disabled but the net instance
        # cap hasn't been reached, enable the add network button
        if not self.button_addNet.isEnabled() and numButtons < self.netCap:
            self.button_addNet.setEnabled(True)

        # Else, if the add network button is enabled but the net instance
        # cap has been reached, disable the add network button
        elif self.button_addNet.isEnabled() and numButtons == self.netCap:
            self.button_addNet.setEnabled(False)

    def action_addNetwork(self):
        """
        Add a new tool button to the network instance buttons group
        """

        # Increment the net instance count
        self.netCount += 1

        # Declare a new tool button object dynamically
        button_netNew = netButton()

        # Connect the new button to appropriate slots
        button_netNew.delSignal.connect(self.action_delNetwork)
        button_netNew.swchSignal.connect(self.action_switchNetwork)

        # Add the new button to the net instance button group
        self.netButtonGroup.addButton(button_netNew)
        self.netButtonGroup.setId(button_netNew, self.netCount)

        # numButtons variable is the currently available number of instances
        allNetworks = self.netButtonGroup.buttons()
        numButtons = len(allNetworks)

        # newId variable is the button group id of the newly-created button
        newId = self.netButtonGroup.id(button_netNew)

        # Add the new button to the scroll area layout, offset of -1 to
        # account for the adder button
        self.layout_netInstance.layout().insertWidget(
            numButtons - 1, button_netNew)

        # If there is only one button, automatically select it
        if numButtons == 1:
            self.netButtonGroup.button(newId).click()

            # Current network variable keeps track of the currently selected
            # network instance (by net button group button id), necessary
            # for saving current simulator parameters into QSettings
            self.currentNetwork = newId
            self.action_switchNetwork(button_netNew)

        if newId not in self.netParams:
            print 'Creating ', newId
            self.action_clearParams(newId)

        self.__updateNetLabels__()

    def action_delNetwork(self, buttonDelete):
        """
        Remove a network instance button
        """

        allNetworks = self.netButtonGroup.buttons()
        numButtons = len(allNetworks)

        # delId variable holds the button group id of the button
        # that is to be deleted, delButton is a pointer to the
        # actual button object, and delIndex is the list index
        # of that button
        delId = self.netButtonGroup.id(buttonDelete)
        delButton = self.netButtonGroup.button(delId)
        delIndex = allNetworks.index(delButton)

        # Don't delete the button if it's the only one left
        if numButtons != 1:
            # If the button to be deleted is not the leftmost one and is also
            # the currently selected button, toggle the button to the left of it
            if delIndex != 0 and self.netButtonGroup.checkedId() == delId:
                self.action_switchNetwork(allNetworks[delIndex - 1])
                allNetworks[delIndex - 1].click()

            # Else, if the button to be deleted is the leftmost one and is also
            # the currently selected button, toggle the button to right of it
            elif delIndex == 0 and self.netButtonGroup.checkedId() == delId:
                self.action_switchNetwork(allNetworks[delIndex + 1])
                allNetworks[delIndex + 1].click()

            # Remove the button to be deleted from the net button group and the
            # net instance layout widget
            self.netButtonGroup.removeButton(buttonDelete)
            self.layout_netInstance.layout().removeWidget(buttonDelete)

            # Cleanly delete the button
            buttonDelete.deleteLater()

            self.__updateNetLabels__()

            # Delete the parameters for the net instance to be deleted
            self.action_delParams(delId)

            self.Manager.delNetwork(delId)

    def action_switchNetwork(self, buttonSwitch):
        """
        Select a different network instance in the GUI
        """

        allNetworks = self.netButtonGroup.buttons()
        numNetworks = len(allNetworks)

        # Switch id is the button group id of the button to be switched to
        switchId = self.netButtonGroup.id(buttonSwitch)

        if numNetworks != 1:
            self.action_readParams(self.currentNetwork)

        # Update the currently selected network
        self.currentNetwork = switchId

        self.action_setParams(switchId)

        self.action_updateLine(switchId)
        self.action_updateCurve()
        # Get the current status of the network instance that's being
        # switched to
        self.action_checkStatus(switchId)

    def action_importWeights(self):
        """
        Open file dialog to import network weights from matlab (.mat) file,
        assign file name to field text
        """
        self.weightsFile = QtGui.QFileDialog.getOpenFileName(
            self, self.tr("Open network weights file"),
            filter=self.tr("MAT-Files (*.mat)"))

        if self.weightsFile[0]:
            self.field_weightsFile.setText(self.weightsFile[0])

    def action_importSamples(self):
        """
        Open file dialog to import learning samples file, assign file
        to samples line_edit widget
        """
        self.samplesFile = QtGui.QFileDialog.getOpenFileName(
            self, self.tr("Open learning samples file"),
            filter=self.tr("Text files (*.txt *.rtf)"))

        if  self.samplesFile[0]:
            self.field_samples.setText(self.samplesFile[0])

    def action_readConfig(self):
        """
        Reads simulator configuration from .ini files
        """

        readName = QtGui.QFileDialog.getOpenFileName(
            self, self.tr('Open simulator config file'),
            filter=self.tr('*.ini'))

        self.readSettings(readName)

    def action_saveConfig(self):

        writeName = QtGui.QFileDialog.getSaveFileName(
            self, self.tr('Save simulator config'))

        self.writeSettings(writeName)

    def readSettings(self, fileName=None):

        # If a configuration file (.ini only) was provided, parse it
        if fileName:
            simSettings = QtCore.QSettings(fileName, QtCore.QSettings.IniFormat)

        else:
            simSettings = QtCore.QSettings()

        # Get the number of network instances to create from the config file
        numNetworks = simSettings.beginReadArray(self.tr('Networks'))

        # If no networks are configured, configure at least one
        if numNetworks == 0:
            numNetworks = 1

        for i in xrange(numNetworks):

            simSettings.setArrayIndex(i)

            self.netParams[i] = {
                'inputName': str(simSettings.value(self.tr('Samples'), '')),
                'outputName': str(simSettings.value(
                    self.tr('Weight_export'), None)),
                'netSize': (str(simSettings.value(self.tr('Topology'), ''))),
                'discInput': (True if str(simSettings.value(
                    self.tr('Discrete_input'), False)).lower() == 'true'
                    else False),
                'discOutput': (True if str(simSettings.value(
                    self.tr('Discrete_output'), False)).lower() == 'true'
                    else False),
                'sectors': int(simSettings.value(self.tr('Sectors'), 0)),
                'weightKey': str(simSettings.value(
                    self.tr('Weights'), 'random')),
                'stopKey': str(simSettings.value(
                    self.tr('Error'), 'max')),
                'localThresh': float(simSettings.value(
                    self.tr('Local_threshold'), 0)),
                'globalThresh': float(simSettings.value(
                    self.tr('Global_threshold'), 0)),
                'softMargins': bool(simSettings.value(
                    self.tr('Soft_margins'), True)),
                'cuda': False,
                'iterLimit': int(simSettings.value(
                    self.tr('Iteration_limit'), 0)),
                'timeLimit': str(simSettings.value(
                    self.tr('Time_limit'), 0)),
                'refreshLimit': int(simSettings.value(
                    self.tr('Refresh_limit'), 1))
            }

            self.action_addNetwork()

        simSettings.endArray()

        #simSettings.clear()
        del simSettings

    def writeSettings(self, fileName=None):

        self.action_readParams()

        if fileName:
            simSettings = QtCore.QSettings(fileName, QtCore.QSettings.IniFormat)

        else:
            simSettings = QtCore.QSettings()

        allNetworks = self.netButtonGroup.buttons()
        numNetworks = len(allNetworks)

        simSettings.beginWriteArray('Networks', size=numNetworks)

        for i in xrange(numNetworks):

            oldId = self.netButtonGroup.id(allNetworks[i])

            self.netParams[i] = self.netParams.pop(oldId)
            self.netButtonGroup.setId(allNetworks[i], i)

            simSettings.setArrayIndex(i)

            simSettings.setValue(self.tr('Samples'),
                                 self.netParams[i]['inputName'])

            if 'outputName' in self.netParams[i]:
                simSettings.setValue(self.tr('Weight_export'),
                                     self.netParams[i]['outputName'])

            simSettings.setValue(self.tr('Topology'),
                                 self.netParams[i]['netSize'])

            simSettings.setValue(self.tr('Discrete_input'),
                                 True if self.netParams[i]['discInput'] is True
                                 else False)
            simSettings.setValue(self.tr('Discrete_output'),
                                 True if self.netParams[i]['discOutput'] is True
                                 else False)

            simSettings.setValue(self.tr('Sectors'),
                                 self.netParams[i]['sectors'])
            simSettings.setValue(self.tr('Weights'),
                                 self.netParams[i]['weightKey'])
            simSettings.setValue(self.tr('Error'),
                                 self.netParams[i]['stopKey'])

            simSettings.setValue(self.tr('Local_threshold'),
                                 self.netParams[i]['localThresh'])
            simSettings.setValue(self.tr('Global threshold'),
                                 self.netParams[i]['globalThresh'])

            simSettings.setValue(self.tr('Soft_margins'),
                                 self.netParams[i]['globalThresh'])

            if 'iterLimit' in self.netParams[i]:
                simSettings.setValue(self.tr('Iteration_limit'),
                                     self.netParams[i]['iterLimit'])

            if 'timeLimit' in self.netParams[i]:
                simSettings.setValue(self.tr('Time_limit'),
                                     self.netParams[i]['timeLimit'])

            if 'refreshLimit' in self.netParams[i]:
                simSettings.setValue(self.tr('Refresh_limit'),
                                     self.netParams[i]['refreshLimit'])

        simSettings.endArray()

    def action_resetGUI(self):
        """
        Resets all GUI parameters, deletes all network objects
        """

        # In the I/O section
        self.field_samples.clear()
        self.field_topology.clear()

        self.box_sectors.setValue(0)

        self.check_discreteIn.setChecked(False)
        self.check_discreteOut.setChecked(False)

        # In the weights section
        self.combo_initWeights.setCurrentIndex(0)
        self.field_weightsFile.clear()

        # In the errors section
        self.combo_stopKey.setCurrentIndex(0)
        self.box_maxGlobal.setValue(0)
        self.box_maxLocal.setValue(0)
        self.box_errorGlobal.setValue(0.0)
        self.box_errorLocal.setValue(0)
        self.box_mseGlobal.setValue(0.0)
        self.box_mseLocal.setValue(0.0)
        self.box_rmseGlobal.setValue(0.0)
        self.box_rmseLocal.setValue(0.0)
        self.box_armseGlobal.setValue(0.0)
        self.box_armseLocal.setValue(0.0)
        self.check_softMargins.setChecked(False)

        # In the control section
        self.check_timeLimit.setChecked(False)
        self.box_timeLimit.setTime(QtCore.QTime.fromString("00:00", "hh:mm"))
        self.check_iterLimit.setChecked(False)
        self.box_iterLimit.setValue(0)
        self.check_useCuda.setChecked(False)
        self.check_useConsole.setChecked(True)
        self.check_plotError.setChecked(True)

        # In error plot pane
        self.box_refreshLimit.setValue(0)
        self.box_currentIter.setValue(0)

        allNetworks = self.netButtonGroup.buttons()
        numNetworks = len(allNetworks)

        for i in xrange(numNetworks - 1, -1, -1):
            if i > 0:
                self.action_delNetwork(allNetworks[i])
            elif i == 0:
                self.netParams[i] = self.netParams.pop(
                    self.netButtonGroup.id(allNetworks[i]))
                self.action_clearParams(0)

    def action_setParams(self, index=None):

        if index == None:
            index = self.netButtonGroup.checkedId()

        self.field_samples.setText(self.netParams[index]['inputName'])

        self.field_topology.setText(self.netParams[index]['netSize'])

        self.box_sectors.setValue(self.netParams[index]['sectors'])

        self.check_discreteIn.setChecked(self.netParams[index]['discInput'])
        self.check_discreteOut.setChecked(self.netParams[index]['discOutput'])

        # In the weights section
        weightKey = self.netParams[index]['weightKey']

        if weightKey == 'random':
            self.combo_initWeights.setCurrentIndex(0)

        else:
            self.combo_initWeights.setCurrentIndex(1)
            self.field_weightsFile.setText(weightKey)

        # In the errors section

        stopKey = self.netParams[index]['stopKey']
        localThresh = self.netParams[index]['localThresh']
        globalThresh = self.netParams[index]['globalThresh']

        if stopKey == 'max':
            self.combo_stopKey.setCurrentIndex(0)
            self.box_maxGlobal.setValue(int(globalThresh))
            self.box_maxLocal.setValue(int(localThresh))
        elif stopKey == 'error':
            self.combo_stopKey.setCurrentIndex(1)
            self.box_errorGlobal.setValue(globalThresh)
            self.box_errorLocal.setValue(int(localThresh))

        elif stopKey == 'mse':
            self.combo_stopKey.setCurrentIndex(2)
            self.box_mseGlobal.setValue(globalThresh)
            self.box_mseLocal.setValue(localThresh)

        elif stopKey == 'rmse':
            self.combo_stopKey.setCurrentIndex(3)
            self.box_rmseGlobal.setValue(globalThresh)
            self.box_rmseLocal.setValue(localThresh)

        elif stopKey == 'armse':
            self.combo_stopKey.setCurrentIndex(4)
            self.box_armseGlobal.setValue(globalThresh)
            self.box_armseLocal.setValue(localThresh)
            self.check_softMargins.setChecked(
                self.netParams[index]['softMargins'])

        del stopKey, localThresh, globalThresh

        # In the control section

        self.check_useCuda.setChecked(self.netParams[index]['cuda'])

        if self.netParams[index]['timeLimit']:
            self.check_timeLimit.setChecked(True)
            self.box_timeLimit.setTime(QtCore.QTime.fromString(
                self.netParams[index]['timeLimit'], 'hh:mm'))

        else:
            self.check_timeLimit.setChecked(False)
            self.box_timeLimit.setTime(QtCore.QTime(0, 0, 0, 0))

        if self.netParams[index]['iterLimit'] != None or 0:
            self.check_iterLimit.setChecked(True)
            self.box_iterLimit.setValue(self.netParams[index]['iterLimit'])

        else:
            self.check_iterLimit.setChecked(False)
            self.box_iterLimit.setValue(0)

        if 'refreshLimit' in self.netParams[index]:
            self.box_refreshLimit.setValue(
                self.netParams[index]['refreshLimit'])

        else:
            self.box_refreshLimit.setValue(1)

    def action_readParams(self, index=None):
        """
        Extract current values of GUI widget elements and assign them
        to the specified network index
        """

        if index == None:
            index = self.netButtonGroup.checkedId()

        newParams = {
            'inputName': str(self.field_samples.text()),
            'outputName': None,
            'netSize': None,
            'discInput': self.check_discreteIn.isChecked(),
            'discOutput': self.check_discreteOut.isChecked(),
            'sectors': self.box_sectors.value(),
            'weightKey': None,
            'stopKey': str(self.combo_stopKey.currentText()),
            'localThresh': None,
            'globalThresh': None,
            'softMargins': True,
            'cuda': self.check_useCuda.isChecked(),
            'refreshLimit': self.box_refreshLimit.value(),
            'iterLimit': None,
            'timeLimit': None
        }

        netSize = str(self.field_topology.text())
        netSize = netSize.strip('[](){}')
        netSize = ', '.join(re.split('\W+|_', netSize))

        newParams['netSize'] = netSize

        weightKey = str(self.combo_initWeights.currentText())

        if weightKey == 'user':
            weightKey = str(self.field_weightsFile.text())

        newParams['weightKey'] = weightKey

        stopKey = newParams['stopKey']

        if stopKey == 'max':
            newParams['localThresh'] = self.box_maxLocal.value()
            newParams['globalThresh'] = self.box_maxGlobal.value()
        elif stopKey == 'error':
            newParams['localThresh'] = self.box_errorLocal.value()
            newParams['globalThresh'] = self.box_errorGlobal.value()
        elif stopKey == 'mse':
            newParams['localThresh'] = self.box_mseLocal.value()
            newParams['globalThresh'] = self.box_mseGlobal.value()

        elif stopKey == 'rmse':
            newParams['localThresh'] = self.box_rmseLocal.value()
            newParams['globalThresh'] = self.box_rmseGlobal.value()

        elif stopKey == 'armse':
            newParams['localThresh'] = self.box_armseLocal.value()
            newParams['globalThresh'] = self.box_armseGlobal.value()
            newParams['softMargins'] = self.check_softMargins.isChecked()

        if self.check_iterLimit.isChecked() and self.box_iterLimit.value() != 0:
            newParams['iterLimit'] = self.box_iterLimit.value()

        timeLimit = self.box_timeLimit.time()

        if (self.check_timeLimit.isChecked() and
                timeLimit != QtCore.QTime(0, 0, 0, 0)):
            # Convert time to string
            newParams['timeLimit'] = timeLimit.toString('hh:mm')

        self.netParams[index] = newParams

        del stopKey, timeLimit

    def action_clearParams(self, index=None):

        if index == None:
            index = self.netButtonGroup.checkedId()

        self.netParams[index] = {
            'inputName': '',
            'outputName': None,
            'netSize': '',
            'discInput': False,
            'discOutput': False,
            'sectors': 0,
            'weightKey': 'random',
            'stopKey': 'max',
            'localThresh': 0,
            'globalThresh': 0,
            'softMargins': True,
            'cuda': False,
            'refreshLimit': 1,
            'iterLimit': None,
            'timeLimit': None
        }

    def action_delParams(self, index=None):

        if index == None:
            index = self.netButtonGroup.checkedId()

        del self.netParams[index]

    def clr_NetworkSettings(self, index=None):
        """
        Clears network settings for the selected network instance
        """

        if index == None:
            index = self.netButtonGroup.checkedId()

        print 'Clearing ', index

        self.simSettings.beginWriteArray(self.tr('Networks'))
        self.simSettings.setArrayIndex(index)

        # In the I/O section
        self.simSettings.beginGroup(self.tr('I/O'))

        self.simSettings.setValue(self.tr('input_file'), '')

        if self.outputName:
            self.simSettings.setValue(self.tr('output_file'), '')

        self.simSettings.setValue(self.tr('topology'), '')

        self.simSettings.setValue(self.tr('sectors'), int(0))

        self.simSettings.setValue(self.tr('discrete_input'), False)

        self.simSettings.setValue(self.tr('discrete_output'), False)

        self.simSettings.endGroup()

        # In the weights section
        self.simSettings.beginGroup(self.tr('Weights'))

        self.simSettings.setValue(self.tr('initial_weights'), self.tr('random'))

        self.simSettings.endGroup()

        # In the errors section
        self.simSettings.beginGroup(self.tr('Learning'))

        self.simSettings.setValue(self.tr('stop_criteria'), 'max')

        self.simSettings.setValue(self.tr('global_threshold'), 0)
        self.simSettings.setValue(self.tr('local_threshold'), 0)
        self.simSettings.setValue(self.tr('soft_margins'), False)

        self.simSettings.endGroup()

        # In the control section
        self.simSettings.beginGroup(self.tr('Control'))

        if self.simSettings.contains(self.tr('time_limit')):
            self.simSettings.remove(self.tr('time_limit'))

        if self.simSettings.contains(self.tr('iter_limit')):
            self.simSettings.remove(self.tr('iter_limit'))

        self.simSettings.endGroup()
        self.simSettings.endArray()

    def del_NetworkSettings(self, index):
        """
        Remove all keys associated with the specified net instance
        """

        simSettings = QtCore.QSettings()

        simSettings.beginWriteArray(self.tr('Networks'))
        simSettings.setArrayIndex(index)

        simSettings.remove(str(index))
        simSettings.endArray()

    def action_initNetwork(self, index=None):
        """
        Checks if the specified net instance is already initialized.
        If not, the network is initialized using the existing parameters
        matching its index. If the net instance has been initialized in
        the network manager, calls the update network function to update
        the network configuration, if necessary
        """

        if index == None:
            index = self.netButtonGroup.checkedId()

        currentStatus = self.Manager.action_getStatus(index)

        # If the network with specified index isn't in the manager
        # initialize it using the current parameters
        if currentStatus is None:
            print 'initializing network', index
            self.Manager.addNetwork(index, self.netParams[index])

        # If the net instance is in the manager, and the function
        # is not being called from an existing net instance
        else:
            # Only update the network if it isn't doing work
            if currentStatus == 'stopped':
                self.Manager.updateNetwork(index, self.netParams[index])

    def action_updateLine(self, index):
        """
        Updates the value of the global threshold line in the error graph
        """

        self.graph_errorLine.setValue(self.netParams[index]['globalThresh'])

    def action_updateCurve(self, index=None):
        """
        Updates the error graph to reflect most current error data
        for the specified network instance
        """

        currentIndex = self.netButtonGroup.checkedId()

        if index == None and currentIndex in self.Manager.netList:

            newIters, newData = self.Manager.getData(currentIndex)

            self.graph_errorCurve.setData(newData[:newIters])

            self.box_currentIter.setValue(newIters)

        elif index in self.Manager.netList and index == currentIndex:
            newIters, newData = self.Manager.getData(index)

            self.graph_errorCurve.setData(newData[:newIters])

            self.box_currentIter.setValue(newIters)

        else:
            self.graph_errorCurve.clear()
            self.box_currentIter.setValue(0)


    def action_switchCurve(self, index):

        self.graph_errorCurve.clear()

        self.action_updateCurve(index)
    def action_resetGraph(self):
        """
        Resets the error graph
        """

        self.graph_plotError.clear()

    def action_checkStatus(self, index=None):
        """
        Checks the current state of a specific network instance and
        sets the control buttons of the GUI for that instance
        """

        currentStatus = None
        currentIndex = self.netButtonGroup.checkedId()

        if index == None:
            currentStatus = self.Manager.action_getStatus(currentIndex)

        else:
            currentStatus = self.Manager.action_getStatus(index)

        if currentStatus == None:
            currentStatus = 'stopped'

        print 'Network ', index, ' is ', currentStatus

        # If the simulator is currently stopped for this instance
        if currentStatus == 'stopped':

            self.button_Learn.setEnabled(True)
            self.button_Test.setEnabled(True)
            self.button_Export.setEnabled(True)
            self.button_randomWeights.setEnabled(True)

            self.button_Learn.setText(self.tr('Learn'))
            self.button_Test.setText(self.tr('Test'))
            self.button_Export.setText(self.tr('Export'))
            self.button_randomWeights.setText(self.tr('Randomize Weights'))

            self.button_Learn.clicked.disconnect()
            self.button_Learn.clicked.connect(self.action_startLearning)

            self.button_Export.setDown(False)
            self.button_randomWeights.setDown(False)

            self.check_lockNet.setEnabled(True)

            self.statusbar.showMessage(self.tr('Ready'), 2000)

        # Else the simulator is currently running some task for this instance
        else:
            # Prevent the user from adding or deleting any networks
            if not self.check_lockNet.isChecked():
                self.check_lockNet.setChecked(True)

            self.check_lockNet.setEnabled(False)

            # If learning
            if currentStatus == 'learning':

                self.button_Learn.setEnabled(True)
                self.button_Test.setEnabled(False)
                self.button_Export.setEnabled(False)
                self.button_randomWeights.setEnabled(False)

                self.button_Learn.setText(self.tr('Stop'))

                self.statusbar.showMessage(self.tr('Learning'))

                self.button_Learn.clicked.disconnect()
                self.button_Learn.clicked.connect(self.action_startStopping)

            elif currentStatus == 'testing':

                self.button_Learn.setEnabled(False)
                self.button_Test.setEnabled(False)
                self.button_Export.setEnabled(False)
                self.button_randomWeights.setEnabled(False)

                self.statusbar.showMessage(self.tr('Testing'))

            elif currentStatus == 'exporting':

                self.button_Export.setDown(True)

                self.button_Learn.setEnabled(False)
                self.button_Test.setEnabled(False)
                self.button_Export.setEnabled(False)
                self.button_randomWeights.setEnabled(False)

                self.statusbar.showMessage(self.tr('Exporting weights'))

            elif currentStatus == 'randomizing':

                self.button_Learn.setEnabled(False)
                self.button_Test.setEnabled(False)
                self.button_Export.setEnabled(False)
                self.button_randomWeights.setEnabled(False)

                self.button_randomWeights.setDown(True)
                self.button_randomWeights.setText(self.tr('Randomizing...'))
                self.statusbar.showMessage(self.tr('Randomizing weights'))

            elif currentStatus == 'stopping':

                self.button_Learn.setEnabled(False)
                self.button_Test.setEnabled(False)
                self.button_Export.setEnabled(False)
                self.button_randomWeights.setEnabled(False)

                self.button_Learn.clicked.disconnect()
                self.button_Learn.clicked.connect(self.action_startLearning)

                self.button_Learn.setText(self.tr('Stopping'))

                self.statusbar.showMessage(self.tr('Stopping...'))

    def action_startLearning(self):
        """
        Action starts learning process for the currently selected network,
        disables other GUI features
        """

        index = self.netButtonGroup.checkedId()

        # Update the parameters for the network
        self.action_readParams(index)
        # If necessary, initialize the network with this index
        self.action_initNetwork(index)

        # Check the status of the specified network
        self.action_checkStatus(index)

        # Enqueue a learning command for the current network instance
        self.Manager.action_Learn(index)

    def action_startTesting(self):
        """
        Action starts testing process for currently selected network,
        disables other GUI features
        """

        index = self.netButtonGroup.checkedId()

        self.action_readParams(index)
        self.action_initNetwork(index)
        self.action_checkStatus(index)

        self.Manager.action_Test(index)

    def action_startExport(self):
        """
        Open file dialog to export current network weights
        """

        weightsFile = QtGui.QFileDialog.getSaveFileName(
            self, self.tr("Export network weights"))[0]

        if (weightsFile is not None and weightsFile is not ''):

            index = self.netButtonGroup.checkedId()

            self.action_readParams(index)
            self.action_initNetwork(index)
            self.action_checkStatus(index)

            self.Manager.action_Export(index, weightsFile)

    def action_startRandom(self):
        """
        Passes a weights randomizing command to the net manager
        for the currently selected network instance
        """

        index = self.netButtonGroup.checkedId()

        self.action_readParams(index)
        self.action_initNetwork(index)
        self.action_checkStatus(index)

        self.Manager.action_Randomize(index)

    def action_startStopping(self, index=None):
        """
        Checks if the specified network instance is running something,
        and stop it if it is
        """

        if not index:
            index = self.netButtonGroup.checkedId()

        if index not in self.netParams:
            pass

        else:
            self.action_readParams(index)
            self.action_initNetwork(index)
            self.action_checkStatus(index)

            self.Manager.action_Stop(index)

    def printData(self, data):
        print data
        pass

    def createActions(self):
        # Placeholder
        self.button_importSamples.clicked.connect(self.action_importSamples)
        self.button_weightsFile.clicked.connect(self.action_importWeights)

        self.button_addNet.clicked.connect(self.action_addNetwork)

        self.button_Learn.clicked.connect(self.action_startLearning)
        #self.button_Learn.clicked.connect(self.action_startStopping)
        self.button_Test.clicked.connect(self.action_startTesting)
        self.button_Export.clicked.connect(self.action_startExport)
        self.button_randomWeights.clicked.connect(self.action_startRandom)

        self.button_Reset.clicked.connect(self.action_resetGUI)

        self.check_showThresh.toggled.connect(self.graph_errorLine.setVisible)

    def createMenus(self):
        # Placeholder
        pass

    def createButtons(self):
        """
        Creates a button group for the network instance buttons
        """

        # Net count variable keeps track of total number of created instances,
        # including those that were deleted. Button group id's come from here
        self.netCount = -1
        # Limit on number of network instances
        self.netCap = 10


        # Create a button group for the network instance buttons
        self.netButtonGroup = QtGui.QButtonGroup(self.layout_netInstance)
        self.netButtonGroup.setExclusive(True)

    def createGraph(self):
        """
        Initializes the graph window.
        """

        self.graph_plotError.setLabel('bottom', 'Iteration')
        self.graph_plotError.setLabel('left', 'Error')

        self.graph_plotError.setDownsampling(mode='peak')
        self.graph_plotError.setAutoPan(y=True)
        self.graph_plotError.setLimits(xMin=0, yMin=0)

        # errorCurve is the actual plot for the error graph - errorLine
        # is an infinite line representing the threshold value
        self.graph_errorCurve = self.graph_plotError.plot()
        self.graph_errorCurve.setClipToView(True)
        self.graph_errorLine = self.graph_plotError.addLine(pen='r')

        # Configure the threshold limit line and hide it initially
        self.graph_errorLine.setAngle(0)
        self.graph_errorLine.setValue(0)
        self.graph_errorLine.setVisible(True)

    def createStatusBar(self):
        self.statusbar.showMessage(self.tr("Ready"), 2000)


class netButton(QtGui.QToolButton):
    """
    Special network instance button, used to manage network instances
    """

    delSignal = QtCore.Signal(QtGui.QToolButton)
    swchSignal = QtCore.Signal(QtGui.QToolButton)

    def __init__(self):
        """
        Constructor function, initializes QToolButton and defines network
        instance index

        Arguments
            parent:     object pointer to parent widget
            group:      object pointer to button group widget
            index:      the integer index of the network instance
        """

        QtGui.QToolButton.__init__(self)

        self.setCheckable(True)

    def mousePressEvent(self, event):
        """
        Overloaded mouse press event
        """

        # If user right-clicked the button, signal the button for deletion
        if event.button() is QtCore.Qt.RightButton:
            self.delSignal.emit(self)

        # Else, pass the event along
        elif event.button() is QtCore.Qt.LeftButton:
            # If the network instance button is not already selected,
            # emit a switching signal
            if not self.isChecked():
                self.swchSignal.emit(self)

            # Pass left click events normally
            QtGui.QToolButton.mousePressEvent(self, event)


if __name__ == '__main__':
    app = QtGui.QApplication.instance()
    if not app:
        app = QtGui.QApplication(sys.argv)
    #app.aboutToQuit.connect(app.deleteLater)
    mainWin = MainWindow()
    ret = app.exec_()
    sys.exit(ret)
