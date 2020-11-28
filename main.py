import sys

from PyQt5 import QtWidgets, QtCore
from segmentation import SegWidget
from widget import Ui_MainWindow

app = QtWidgets.QApplication(sys.argv)#定义一个窗口程序

widget = SegWidget()#定义一个窗口对象，用户可以通过接口来自定义


widget.show()

sys.exit(app.exec_())