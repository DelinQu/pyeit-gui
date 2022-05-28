import os
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QDialog
from PyQt5 import QtGui
from sqlalchemy import true

from ui.main_ui import Ui_MainWindow
import ui.ui_ok as ui_ok
from eit import Eit
from utils.utils import Timer, TIME_LIMIT
import utils.utils as Util

class MyApp(QMainWindow, Ui_MainWindow):
    ''' initial function. '''
    def __init__(self, parent=None):
        super(MyApp, self).__init__(parent)
        self.setupUi(self)
        self.connectSlots()

        # ! global config.
        self.model_detail = {
            'Thoracic Model': 'The thoracic model to complete the image reconstruction.',
            'Circle Model': 'Select the unit circle model to complete the image reconstruction.'
        }

        self.model2key = {
            'Thoracic Model': 'thorax',
            'Circle Model': 'circle'
        }

        # ! inliner values.
        self.fpath = './coordinate.csv'
        self.mat_path = './data.npy'
        self.id = 0
        self.model = 'Thoracic Model'
        self.mtd = 'BP'
        self.state = 'not_start'                    # 
        self.state_list = ['not_start', 'paused', 'running', 'end'] 
        self.thread_num = 5

        # ! text content
        self.log_content = ''
        self.md_detail.setText(
            self.model_detail[self.model]
        )

        # ! set eit
        self.eit = Eit(fpath=self.fpath, model=self.model2key[self.model])
        self.update_log('Init EIT Obj with {}'.format(self.fpath))

        #! Threads
        self.timer = Timer()

    ''' bind functions to widgets.'''
    def connectSlots(self):
        self.file.clicked.connect(self.openFileNameDialog)              # file picker
        self.start_stop.clicked.connect(self.start_or_stop)             # start and stop
        self.save.clicked.connect(self.savefig_mat)                         # save figures
        self.mtds.currentChanged.connect(self.get_methods)              # get current method
        self.model_box.currentIndexChanged.connect(self.get_model)      # get current model
        self.model_box.currentIndexChanged.connect(self.startProcess)   # bar
        self.model_box.currentIndexChanged.connect(self.set_model)      # set cur model
        self.threadSlider.valueChanged.connect(self.get_thread_num)     # get current model
        self.reset_button.clicked.connect(self.reset)                   # reset
        self.checkBox.clicked.connect(self.pop_checkbox)                # pop

    def get_thread_num(self):
        self.thread_num = self.threadSlider.value()                 # get current slider data.

    ''' get method. '''
    def get_methods(self):
        self.mtd = self.mtds.tabText(self.mtds.currentIndex())
        self.update_log('OK: get current method {}'.format(self.mtd))

    ''' get model. '''
    def get_model(self):
        self.model = self.model_box.currentText()
        self.update_log('OK: get current model {}'.format(self.model))

    '''set model detail.'''
    def set_model(self):
        self.md_detail.setText(
            self.model_detail[self.model]
        )
        self.reset()
        self.eit = Eit(fpath=self.fpath, model=self.model2key[self.model])
        self.update_log('OK: Set Model to {} Successfully !'.format(self.model))

        self.pop_dialog('Set Model to {} Successfully !'.format(self.model))

    ''' check box pop'''
    def pop_checkbox(self):
        if self.checkBox.isChecked():
            self.pop_dialog('Please pick a mat file for offline mode !', isOk=False)

    ''' pop dialog. '''
    def pop_dialog(self, msg, isOk = true):
        self.update_log('OK:' + msg)
        dialog = QDialog()
        dialog.ui = ui_ok.Ui_Error()
        dialog.ui.setupUi(dialog)
        if not isOk:
            dialog.ui.error_icon.setPixmap(QtGui.QPixmap("/home/qdl/Desktop/pyeit-gui/ui/../icons/warn.png"))
        dialog.ui.error.setText(msg)
        dialog.exec_()
        dialog.show()
   

    ''' open a Dialog to pick a data file. '''
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        if not self.checkBox.isChecked():
            fpath, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","CSV Files (*.csv)", options=options)
        else:
            fpath, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","mat Files (*.npy)", options=options)
        _, filename = os.path.split(fpath)

        if fpath:
            if not self.checkBox.isChecked():
                self.fpath = fpath
            else:
                self.mat_path = fpath
            self.file.setText(filename)
            self.update_log('user picked a data file {} from {}'.format(filename, fpath))
    
    ''' update log '''
    def update_log(self, content: str):
        self.log_content = '{}.\n'.format(content) + self.log_content
        self.log.setText(self.log_content)

    ''' start progressBar '''
    def startProcess(self):
        # 初始化
        self.timer.initCount(0)
        self.progressBar.setValue(0)
        self.update_log('Reset EIT Object ....')
        self.timer.countChanged.connect(self.onCountChanged)
        self.update_log('Start ....')
        self.timer.start()

    ''' update progressBar '''
    def onCountChanged(self, value):
        if value >= TIME_LIMIT:
            self.update_log('Reset Successfully !')
        self.progressBar.setValue(value)
        self.update_log('+++{}+++'.format(value))

    ''' start or pause ?'''
    def start_or_stop(self):
        if self.state == 'not_start':
            self.update_log('start all the threads')
            self.state = 'running'
            Util.exitFlag = 0
            Util.distrib(self.thread_num, self.eit, self.mtd, self.mplWidget.canvas, offline=self.checkBox.isChecked(), fpath = self.mat_path)
            self.start_stop.setText('Pause')
        
        elif self.state == 'running':
            self.update_log('pause all the threads')
            self.state = 'paused'

            self.update_log('Reset all the state.')
            Util.exitFlag = 1
            Util.join()

            self.start_stop.setText('Continue')
        
        elif self.state == 'paused':
            self.update_log('contine to run all the threads')
            self.state = 'running'
            Util.exitFlag = 0

            Util.distrib(self.thread_num, self.eit, self.mtd, self.mplWidget.canvas, offline=self.checkBox.isChecked(), fpath = self.mat_path)
            self.start_stop.setText('Pause')
        
        else:
            self.update_log('Error coccupy!')
            self.state = 'not_start'
            Util.exitFlag = 1
            self.start_stop.setText('Start')
            Util.join()

    ''' reset all the state. '''
    def reset(self):
        self.update_log('Reset all the state.')
        self.state = 'not_start'
        Util.exitFlag = 1
        self.start_stop.setText('Start')
        Util.join()
        self.mplWidget.canvas.ax.cla()
        # clear the figure.

    ''' save figure to dir.'''
    def savefig_mat(self):
        if self.saveBox.currentText() == 'figure':
            name = '{}_{:04d}'.format(self.mtd ,self.id)
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            
            fpath, _ = QFileDialog.getSaveFileName(
                self,"QFileDialog.getSaveFileName()",
                "{}.png".format(name),
                "png Files (*.png)", options=options
            )
            self.mplWidget.canvas.figure.savefig(fpath, dpi=100)
            self.update_log('Save image {}.png  to {}'.format(name, fpath))
        else:
            name = '{}_{}.npy'.format(self.model, self.mtd)
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            
            fpath, _ = QFileDialog.getSaveFileName(
                self,"QFileDialog.getSaveFileName()",
                name,
                "png Files (*.npy)", options=options
            )
            msg = Util.save_mat(fpath=fpath)
            self.update_log(msg)



