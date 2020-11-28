# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dlg.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_dlg(object):
    def setupUi(self, dlg):
        dlg.setObjectName("dlg")
        dlg.resize(206, 108)
        self.verticalLayout = QtWidgets.QVBoxLayout(dlg)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.radioButton = QtWidgets.QRadioButton(dlg)
        self.radioButton.setChecked(True)
        self.radioButton.setObjectName("radioButton")
        self.gridLayout.addWidget(self.radioButton, 0, 0, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(dlg)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 0, 1, 1, 1)
        self.radioButton_2 = QtWidgets.QRadioButton(dlg)
        self.radioButton_2.setObjectName("radioButton_2")
        self.gridLayout.addWidget(self.radioButton_2, 1, 0, 1, 1)
        self.spinBox = QtWidgets.QSpinBox(dlg)
        self.spinBox.setEnabled(False)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout.addWidget(self.spinBox, 1, 1, 1, 1)
        self.pushButton = QtWidgets.QPushButton(dlg)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 2, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(dlg)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 2, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)

        self.retranslateUi(dlg)
        QtCore.QMetaObject.connectSlotsByName(dlg)

    def retranslateUi(self, dlg):
        _translate = QtCore.QCoreApplication.translate
        dlg.setWindowTitle(_translate("dlg", "Camera Setting"))
        self.radioButton.setText(_translate("dlg", "Video"))
        self.pushButton_3.setText(_translate("dlg", "Select Source"))
        self.radioButton_2.setText(_translate("dlg", "Camera"))
        self.pushButton.setText(_translate("dlg", "Cancel"))
        self.pushButton_2.setText(_translate("dlg", "OK"))

