import sys
import time

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem
from myApp import MyApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mapp = MyApp()
    mapp.show()
    sys.exit(app.exec_())
