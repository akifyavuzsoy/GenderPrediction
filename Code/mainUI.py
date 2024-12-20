# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(967, 638)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_Train = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_Train.setGeometry(QtCore.QRect(20, 10, 451, 551))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_Train.setFont(font)
        self.groupBox_Train.setObjectName("groupBox_Train")
        self.btn_TrainModel = QtWidgets.QPushButton(self.groupBox_Train)
        self.btn_TrainModel.setGeometry(QtCore.QRect(290, 240, 151, 51))
        self.btn_TrainModel.setObjectName("btn_TrainModel")
        self.layoutWidget = QtWidgets.QWidget(self.groupBox_Train)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 40, 181, 41))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.cBox_TrainModel = QtWidgets.QComboBox(self.layoutWidget)
        self.cBox_TrainModel.setObjectName("cBox_TrainModel")
        self.cBox_TrainModel.addItem("")
        self.cBox_TrainModel.addItem("")
        self.cBox_TrainModel.addItem("")
        self.cBox_TrainModel.addItem("")
        self.cBox_TrainModel.addItem("")
        self.cBox_TrainModel.addItem("")
        self.horizontalLayout.addWidget(self.cBox_TrainModel)
        self.layoutWidget1 = QtWidgets.QWidget(self.groupBox_Train)
        self.layoutWidget1.setGeometry(QtCore.QRect(30, 100, 181, 31))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget1)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.sBox_TrainModel = QtWidgets.QSpinBox(self.layoutWidget1)
        self.sBox_TrainModel.setMaximum(1000)
        self.sBox_TrainModel.setSingleStep(5)
        self.sBox_TrainModel.setObjectName("sBox_TrainModel")
        self.horizontalLayout_2.addWidget(self.sBox_TrainModel)
        self.progressBar_Train = QtWidgets.QProgressBar(self.groupBox_Train)
        self.progressBar_Train.setGeometry(QtCore.QRect(30, 200, 391, 23))
        self.progressBar_Train.setProperty("value", 0)
        self.progressBar_Train.setObjectName("progressBar_Train")
        self.layoutWidget2 = QtWidgets.QWidget(self.groupBox_Train)
        self.layoutWidget2.setGeometry(QtCore.QRect(30, 310, 261, 29))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_6 = QtWidgets.QLabel(self.layoutWidget2)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_5.addWidget(self.label_6)
        self.txt_ModelAccuracy = QtWidgets.QLineEdit(self.layoutWidget2)
        self.txt_ModelAccuracy.setEnabled(True)
        self.txt_ModelAccuracy.setReadOnly(True)
        self.txt_ModelAccuracy.setObjectName("txt_ModelAccuracy")
        self.horizontalLayout_5.addWidget(self.txt_ModelAccuracy)
        self.layoutWidget_4 = QtWidgets.QWidget(self.groupBox_Train)
        self.layoutWidget_4.setGeometry(QtCore.QRect(30, 150, 271, 29))
        self.layoutWidget_4.setObjectName("layoutWidget_4")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget_4)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_5 = QtWidgets.QLabel(self.layoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 1)
        self.txt_NewModelPath = QtWidgets.QLineEdit(self.layoutWidget_4)
        self.txt_NewModelPath.setObjectName("txt_NewModelPath")
        self.gridLayout_2.addWidget(self.txt_NewModelPath, 0, 1, 1, 1)
        self.tBtn_NewSelectModel = QtWidgets.QToolButton(self.layoutWidget_4)
        self.tBtn_NewSelectModel.setObjectName("tBtn_NewSelectModel")
        self.gridLayout_2.addWidget(self.tBtn_NewSelectModel, 0, 2, 1, 1)
        self.groupBox_Test = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_Test.setGeometry(QtCore.QRect(490, 10, 461, 551))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_Test.setFont(font)
        self.groupBox_Test.setObjectName("groupBox_Test")
        self.btn_TestModel = QtWidgets.QPushButton(self.groupBox_Test)
        self.btn_TestModel.setGeometry(QtCore.QRect(300, 150, 151, 51))
        self.btn_TestModel.setObjectName("btn_TestModel")
        self.btn_NewAudio = QtWidgets.QPushButton(self.groupBox_Test)
        self.btn_NewAudio.setGeometry(QtCore.QRect(320, 90, 91, 31))
        self.btn_NewAudio.setObjectName("btn_NewAudio")
        self.layoutWidget3 = QtWidgets.QWidget(self.groupBox_Test)
        self.layoutWidget3.setGeometry(QtCore.QRect(30, 40, 271, 29))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget3)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(self.layoutWidget3)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.txt_ModelPath = QtWidgets.QLineEdit(self.layoutWidget3)
        self.txt_ModelPath.setObjectName("txt_ModelPath")
        self.gridLayout.addWidget(self.txt_ModelPath, 0, 1, 1, 1)
        self.tBtn_SelectModel = QtWidgets.QToolButton(self.layoutWidget3)
        self.tBtn_SelectModel.setObjectName("tBtn_SelectModel")
        self.gridLayout.addWidget(self.tBtn_SelectModel, 0, 2, 1, 1)
        self.layoutWidget4 = QtWidgets.QWidget(self.groupBox_Test)
        self.layoutWidget4.setGeometry(QtCore.QRect(30, 90, 271, 29))
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget4)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.txt_AudioPath = QtWidgets.QLineEdit(self.layoutWidget4)
        self.txt_AudioPath.setObjectName("txt_AudioPath")
        self.horizontalLayout_4.addWidget(self.txt_AudioPath)
        self.tBtn_SelectFile = QtWidgets.QToolButton(self.layoutWidget4)
        self.tBtn_SelectFile.setObjectName("tBtn_SelectFile")
        self.horizontalLayout_4.addWidget(self.tBtn_SelectFile)
        self.btn_OpenAudio = QtWidgets.QPushButton(self.groupBox_Test)
        self.btn_OpenAudio.setGeometry(QtCore.QRect(410, 90, 41, 31))
        self.btn_OpenAudio.setObjectName("btn_OpenAudio")
        self.layoutWidget_2 = QtWidgets.QWidget(self.groupBox_Test)
        self.layoutWidget_2.setGeometry(QtCore.QRect(40, 230, 261, 46))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.layoutWidget_2)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_7 = QtWidgets.QLabel(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_6.addWidget(self.label_7)
        self.txt_Prediction = QtWidgets.QLineEdit(self.layoutWidget_2)
        self.txt_Prediction.setEnabled(True)
        self.txt_Prediction.setReadOnly(True)
        self.txt_Prediction.setObjectName("txt_Prediction")
        self.horizontalLayout_6.addWidget(self.txt_Prediction)
        self.layoutWidget_3 = QtWidgets.QWidget(self.groupBox_Test)
        self.layoutWidget_3.setGeometry(QtCore.QRect(40, 290, 261, 29))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.layoutWidget_3)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_8 = QtWidgets.QLabel(self.layoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_7.addWidget(self.label_8)
        self.txt_TestModelAccuracy = QtWidgets.QLineEdit(self.layoutWidget_3)
        self.txt_TestModelAccuracy.setEnabled(True)
        self.txt_TestModelAccuracy.setReadOnly(True)
        self.txt_TestModelAccuracy.setObjectName("txt_TestModelAccuracy")
        self.horizontalLayout_7.addWidget(self.txt_TestModelAccuracy)
        self.lbl_version = QtWidgets.QLabel(self.centralwidget)
        self.lbl_version.setGeometry(QtCore.QRect(30, 570, 251, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbl_version.setFont(font)
        self.lbl_version.setText("")
        self.lbl_version.setObjectName("lbl_version")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 967, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", " Gender Prediction App"))
        self.groupBox_Train.setTitle(_translate("MainWindow", "TRAIN MODEL"))
        self.btn_TrainModel.setText(_translate("MainWindow", "Train Model"))
        self.label.setText(_translate("MainWindow", "Model"))
        self.cBox_TrainModel.setItemText(0, _translate("MainWindow", "CNN"))
        self.cBox_TrainModel.setItemText(1, _translate("MainWindow", "RNN"))
        self.cBox_TrainModel.setItemText(2, _translate("MainWindow", "LSTM"))
        self.cBox_TrainModel.setItemText(3, _translate("MainWindow", "CoLSTM"))
        self.cBox_TrainModel.setItemText(4, _translate("MainWindow", "CNN_LSTM"))
        self.cBox_TrainModel.setItemText(5, _translate("MainWindow", "GRU"))
        self.label_2.setText(_translate("MainWindow", "Epoch Count"))
        self.label_6.setText(_translate("MainWindow", "Model Accuracy"))
        self.label_5.setText(_translate("MainWindow", "Model     "))
        self.tBtn_NewSelectModel.setText(_translate("MainWindow", "..."))
        self.groupBox_Test.setTitle(_translate("MainWindow", "TEST MODEL"))
        self.btn_TestModel.setText(_translate("MainWindow", "Test Model"))
        self.btn_NewAudio.setText(_translate("MainWindow", "New Audio"))
        self.label_3.setText(_translate("MainWindow", "Model     "))
        self.tBtn_SelectModel.setText(_translate("MainWindow", "..."))
        self.label_4.setText(_translate("MainWindow", "Audio File"))
        self.tBtn_SelectFile.setText(_translate("MainWindow", "..."))
        self.btn_OpenAudio.setText(_translate("MainWindow", "►"))
        self.label_7.setText(_translate("MainWindow", "Prediction       "))
        self.label_8.setText(_translate("MainWindow", "Model Accuracy"))
