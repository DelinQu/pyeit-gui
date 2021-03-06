# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/qdl/Desktop/pyeit-gui/ui/main_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(801, 643)
        MainWindow.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setMinimumSize(QtCore.QSize(0, 100))
        font = QtGui.QFont()
        font.setPointSize(27)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.title.setFont(font)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        self.verticalLayout_2.addWidget(self.title)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.model_box = QtWidgets.QComboBox(self.centralwidget)
        self.model_box.setObjectName("model_box")
        self.model_box.addItem("")
        self.model_box.addItem("")
        self.horizontalLayout_2.addWidget(self.model_box)
        self.start_stop = QtWidgets.QPushButton(self.centralwidget)
        self.start_stop.setObjectName("start_stop")
        self.horizontalLayout_2.addWidget(self.start_stop)
        self.file = QtWidgets.QPushButton(self.centralwidget)
        self.file.setObjectName("file")
        self.horizontalLayout_2.addWidget(self.file)
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setChecked(False)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_2.addWidget(self.checkBox)
        self.horizontalLayout_2.setStretch(0, 2)
        self.horizontalLayout_2.setStretch(2, 1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 15))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setContentsMargins(-1, 16, -1, 16)
        self.formLayout.setObjectName("formLayout")
        self.md_id = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.md_id.setFont(font)
        self.md_id.setObjectName("md_id")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.md_id)
        self.md_name = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setItalic(True)
        self.md_name.setFont(font)
        self.md_name.setObjectName("md_name")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.md_name)
        self.md_detail = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.md_detail.setFont(font)
        self.md_detail.setObjectName("md_detail")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.md_detail)
        self.verticalLayout.addLayout(self.formLayout)
        self.mtds = QtWidgets.QTabWidget(self.centralwidget)
        self.mtds.setMinimumSize(QtCore.QSize(0, 200))
        self.mtds.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.mtds.setDocumentMode(False)
        self.mtds.setMovable(True)
        self.mtds.setTabBarAutoHide(True)
        self.mtds.setObjectName("mtds")
        self.BP = QtWidgets.QWidget()
        self.BP.setAccessibleName("")
        self.BP.setObjectName("BP")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.BP)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.BP)
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.horizontalLayout_6.addWidget(self.textBrowser_2)
        self.mtds.addTab(self.BP, "")
        self.GREIT = QtWidgets.QWidget()
        self.GREIT.setObjectName("GREIT")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.GREIT)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.textBrowser = QtWidgets.QTextBrowser(self.GREIT)
        self.textBrowser.setObjectName("textBrowser")
        self.horizontalLayout_5.addWidget(self.textBrowser)
        self.mtds.addTab(self.GREIT, "")
        self.JCA = QtWidgets.QWidget()
        self.JCA.setObjectName("JCA")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.JCA)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.JCA)
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.horizontalLayout_7.addWidget(self.textBrowser_3)
        self.mtds.addTab(self.JCA, "")
        self.verticalLayout.addWidget(self.mtds)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.threadSlider = QtWidgets.QSlider(self.centralwidget)
        self.threadSlider.setMinimum(4)
        self.threadSlider.setMaximum(15)
        self.threadSlider.setProperty("value", 5)
        self.threadSlider.setOrientation(QtCore.Qt.Horizontal)
        self.threadSlider.setObjectName("threadSlider")
        self.horizontalLayout.addWidget(self.threadSlider)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("/home/qdl/Desktop/pyeit-gui/ui/../icons/thread.png"))
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.thread_label = QtWidgets.QLabel(self.centralwidget)
        self.thread_label.setObjectName("thread_label")
        self.horizontalLayout.addWidget(self.thread_label)
        self.reset_button = QtWidgets.QPushButton(self.centralwidget)
        self.reset_button.setObjectName("reset_button")
        self.horizontalLayout.addWidget(self.reset_button)
        self.saveBox = QtWidgets.QComboBox(self.centralwidget)
        self.saveBox.setObjectName("saveBox")
        self.saveBox.addItem("")
        self.saveBox.addItem("")
        self.horizontalLayout.addWidget(self.saveBox)
        self.save = QtWidgets.QPushButton(self.centralwidget)
        self.save.setObjectName("save")
        self.horizontalLayout.addWidget(self.save)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.mplWidget = MplWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mplWidget.sizePolicy().hasHeightForWidth())
        self.mplWidget.setSizePolicy(sizePolicy)
        self.mplWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.mplWidget.setObjectName("mplWidget")
        self.horizontalLayout_3.addWidget(self.mplWidget)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.log = QtWidgets.QTextEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.log.sizePolicy().hasHeightForWidth())
        self.log.setSizePolicy(sizePolicy)
        self.log.setMaximumSize(QtCore.QSize(16777215, 80))
        self.log.setAutoFillBackground(False)
        self.log.setTabChangesFocus(True)
        self.log.setReadOnly(True)
        self.log.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        self.log.setObjectName("log")
        self.verticalLayout_2.addWidget(self.log)
        self.horizontalLayout_4.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        self.mtds.setCurrentIndex(0)
        self.reset_button.clicked.connect(self.mplWidget.update)
        self.model_box.activated['QString'].connect(self.md_detail.setText)
        self.model_box.activated['QString'].connect(self.md_name.setText)
        self.threadSlider.valueChanged['int'].connect(self.thread_label.setNum)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Reconstrction System Based on PyEIT"))
        self.title.setText(_translate("MainWindow", "Image Reconstrction System Based on PyEIT"))
        self.model_box.setItemText(0, _translate("MainWindow", "Thoracic Model"))
        self.model_box.setItemText(1, _translate("MainWindow", "Circle Model"))
        self.start_stop.setText(_translate("MainWindow", "start"))
        self.file.setText(_translate("MainWindow", "file"))
        self.checkBox.setText(_translate("MainWindow", "offline"))
        self.md_id.setText(_translate("MainWindow", "model"))
        self.md_name.setText(_translate("MainWindow", "thoracic model "))
        self.md_detail.setText(_translate("MainWindow", "the thoracic model to complete the image reconstruction."))
        self.textBrowser_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; font-weight:600;\">Image reconstruction based on back projection algorithm.</span></p></body></html>"))
        self.mtds.setTabText(self.mtds.indexOf(self.BP), _translate("MainWindow", "BP"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; font-weight:600;\">Image reconstruction based on Graz consensus Reconstruction algorithm for EIT.</span></p></body></html>"))
        self.mtds.setTabText(self.mtds.indexOf(self.GREIT), _translate("MainWindow", "GREIT"))
        self.textBrowser_3.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; font-weight:600;\">Image reconstruction based on linear Gauss-Newton solvers.</span></p></body></html>"))
        self.mtds.setTabText(self.mtds.indexOf(self.JCA), _translate("MainWindow", "JCA"))
        self.thread_label.setText(_translate("MainWindow", "5"))
        self.reset_button.setText(_translate("MainWindow", "Reset"))
        self.saveBox.setItemText(0, _translate("MainWindow", "figure"))
        self.saveBox.setItemText(1, _translate("MainWindow", "npy"))
        self.save.setText(_translate("MainWindow", "save"))
        self.log.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Some Log here.</p></body></html>"))
from ui.mplwidget import MplWidget
