# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/qdl/Desktop/pyeit-gui/ui/ui_ok.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Error(object):
    def setupUi(self, Error):
        Error.setObjectName("Error")
        Error.resize(250, 150)
        Error.setMinimumSize(QtCore.QSize(250, 150))
        Error.setMaximumSize(QtCore.QSize(250, 150))
        self.horizontalLayout = QtWidgets.QHBoxLayout(Error)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(Error)
        self.frame.setStyleSheet("background:rgb(51,51,51);")
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_top = QtWidgets.QFrame(self.frame)
        self.frame_top.setMinimumSize(QtCore.QSize(0, 55))
        self.frame_top.setMaximumSize(QtCore.QSize(16777215, 55))
        self.frame_top.setStyleSheet("background:rgb(91,90,90);")
        self.frame_top.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_top.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_top.setObjectName("frame_top")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_top)
        self.horizontalLayout_2.setContentsMargins(15, 5, 0, 0)
        self.horizontalLayout_2.setSpacing(5)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.error_icon = QtWidgets.QLabel(self.frame_top)
        self.error_icon.setMinimumSize(QtCore.QSize(35, 35))
        self.error_icon.setMaximumSize(QtCore.QSize(35, 35))
        self.error_icon.setText("")
        self.error_icon.setPixmap(QtGui.QPixmap("/home/qdl/Desktop/pyeit-gui/ui/../icons/ok.png"))
        self.error_icon.setObjectName("error_icon")
        self.horizontalLayout_2.addWidget(self.error_icon)
        self.error = QtWidgets.QLabel(self.frame_top)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.error.setFont(font)
        self.error.setStyleSheet("color:rgb(255,255,255);")
        self.error.setWordWrap(True)
        self.error.setObjectName("error")
        self.horizontalLayout_2.addWidget(self.error)
        self.verticalLayout.addWidget(self.frame_top)
        self.frame_bottom = QtWidgets.QFrame(self.frame)
        self.frame_bottom.setStyleSheet("background:rgb(91,90,90);")
        self.frame_bottom.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_bottom.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_bottom.setObjectName("frame_bottom")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_bottom)
        self.gridLayout.setContentsMargins(-1, -1, -1, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.bn_ok = QtWidgets.QPushButton(self.frame_bottom)
        self.bn_ok.setMinimumSize(QtCore.QSize(69, 25))
        self.bn_ok.setMaximumSize(QtCore.QSize(69, 25))
        self.bn_ok.setStyleSheet("QPushButton {\n"
"    border: 2px solid rgb(51,51,51);\n"
"    border-radius: 5px;    \n"
"    color:rgb(255,255,255);\n"
"    background-color: rgb(51,51,51);\n"
"}\n"
"QPushButton:hover {\n"
"    border: 2px solid rgb(0,143,150);\n"
"    background-color: rgb(0,143,150);\n"
"}\n"
"QPushButton:pressed {    \n"
"    border: 2px solid rgb(0,143,150);\n"
"    background-color: rgb(51,51,51);\n"
"}\n"
"")
        self.bn_ok.setObjectName("bn_ok")
        self.gridLayout.addWidget(self.bn_ok, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.frame_bottom)
        self.horizontalLayout.addWidget(self.frame)

        self.retranslateUi(Error)
        self.bn_ok.clicked.connect(Error.close)
        QtCore.QMetaObject.connectSlotsByName(Error)

    def retranslateUi(self, Error):
        _translate = QtCore.QCoreApplication.translate
        Error.setWindowTitle(_translate("Error", "Dialog"))
        self.error.setText(_translate("Error", "Please pick a data file first !"))
        self.bn_ok.setText(_translate("Error", "OK"))
