from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap,QFont,QPainter
from PyQt5.QtCore import QObject, QThread, QSize,pyqtSignal, QRect
from PyQt5.QtWidgets import QMessageBox,QFileDialog

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import os.path as osp
import cv2
import time
import json
from collections import OrderedDict

from utils.pyt_utils import load_model
from bisenet_x39 import BiSeNet_x39
from bisenet_r101 import BiSeNet_r101
from bisinet_r18 import BiSeNet_r18
from widget import Ui_MainWindow
from dlg import Ui_dlg


class Inference(QObject):
    signal_update = pyqtSignal(QPixmap, QPixmap,QPixmap,QPixmap,float)
    signal_finshed = pyqtSignal()
    signal_message = pyqtSignal(int)

    def __init__(self,is_cuda):

        super(Inference, self).__init__()
        self.mode = 0
        self.video_path = 'city.avi'
        self.camera_num = 0
        self.source = 0
        self.stop = False
        self.is_cuda = is_cuda
        self.dataset = 0 # Fish-0 city-1
        self.net_num = 0
        self.num_classes = 18
        self.input_path = './res/1_Img8bit.png'
        self.label_path = './res/1_gtFine_labelTrainIds.png'
        self.gt_path = './res/1_gtFine_color.png'
        mean = (104.00698793, 116.66876762, 122.67891434)
        respth = './res'

        self.color_map_fish = [[128, 64, 128], [250, 170, 160], [250, 170, 30], [220, 220, 0], [153, 153, 153], [180, 165, 180],
                     [243, 35, 232], [220, 20, 59], [254, 0, 0], [0, 0, 142], [0, 0, 70],
                     [1, 60, 100], [0, 0, 230], [119, 12, 32], [70, 70, 70], [107, 142, 35], [153, 251, 152],
                     [70, 130, 180]]
        self.color_map_fish.append([0, 0, 0])
        self.color_map_fish = np.array(self.color_map_fish)

        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        color = {el['trainId']: el['color'] for el in labels_info}

        self.color_map_city = []
        for i in range(19):
            self.color_map_city.append(color[i])
        self.color_map_city.append([0, 0, 0])
        self.color_map_city = np.array(self.color_map_city)


        self.save_pth = osp.join(respth, 'FT.pth')
        self.init_net()
        self.net = load_model(self.net, self.save_pth)
        self.net.eval()
        if self.is_cuda:
            self.net.cuda()

    def work(self):
        if self.source == 0:
            img = cv2.imread(self.input_path)
            gt = cv2.imread(self.gt_path)
            #img = cv2.resize(np.uint8(img), (1920, 1080), interpolation=cv2.INTER_LINEAR)

            h, w, _ = img.shape
            img_ground = img
            star_time = time.time()
            pred = self.predict_picture(self.net, img)
            if isinstance(pred, bool):
                self.signal_finshed.emit()
                return None
            interval = time.time() - star_time

            if self.label_path != self.input_path:
                label = Image.open(self.label_path)
                label = np.array(label).astype(np.int64)[np.newaxis, :]
                if "leftImg8bit" in self.input_path:
                    for k, v in self.lb_map.items():
                        label[label == k] = v
                pred[label == 255] =self.num_classes
            else:
                if self.dataset == 0:
                    self.label_path = './res/1.png'
                    label = Image.open(self.label_path)
                    label = label.resize((w, h))
                    label = np.array(label).astype(np.int64)[np.newaxis, :]
                    pred[label == 255] = self.num_classes
                elif self.dataset == 1:
                    pass

            show = self.color_map[pred.astype(int)].squeeze(0)
            show = np.array(show)
            show = cv2.resize(np.uint8(show), (1920, 1080), interpolation=cv2.INTER_LINEAR)
            img_ground = cv2.resize(np.uint8(img_ground), (1920, 1080), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(np.uint8(gt), (1920, 1080), interpolation=cv2.INTER_LINEAR)
            super = cv2.addWeighted(img_ground, 0.55, show, 0.35, 0)
            img_ground = cv2.cvtColor(img_ground, cv2.COLOR_BGR2RGB)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            cv2.putText(show, 'Segmentation', (600, 1020), 3, 3, (255, 255, 255), 4)
            cv2.putText(img_ground, 'Input Source', (600, 1020), 3, 3, (255, 255, 255), 4)
            cv2.putText(super, 'Superposition', (600, 1020), 3, 3, (255, 255, 255), 4)
            cv2.putText(gt, 'Ground Truth', (600, 1020), 3, 3, (255, 255, 255), 4)
            show = Image.fromarray(show)
            show = show.toqpixmap()
            img_ground = Image.fromarray(img_ground)
            img_ground = img_ground.toqpixmap()
            super = Image.fromarray(super)
            super = super.toqpixmap()
            gt = Image.fromarray(gt)
            gt = gt.toqpixmap()

            self.signal_update.emit(img_ground,show, super, gt,interval)
        elif self.source == 1:
            if self.mode == 0:
                capture = cv2.VideoCapture(self.video_path)
            elif self.mode == 1:
                capture = cv2.VideoCapture(self.camera_num)
            while (True):
                if self.stop:
                    break
                ret, img = capture.read()
                if img is None :
                    if self.mode == 1:
                        self.signal_message.emit(int(3))
                        break
                    elif self.mode == 0:
                        self.signal_message.emit(int(4))
                        break

                h, w, _ = img.shape
                img_ground = img

                star_time = time.time()
                pred = self.predict_picture(self.net, img)
                if isinstance(pred, bool):
                    break

                if self.dataset == 0:
                    self.label_path = './res/1.png'

                    label = Image.open(self.label_path)
                    label = label.resize((w,h))
                    label = np.array(label).astype(np.int64)[np.newaxis, :]
                    pred[label == 255] = self.num_classes
                elif self.dataset == 1:
                    pass

                interval = time.time() - star_time

                show = self.color_map[pred.astype(int)].squeeze(0)
                show = np.array(show)
                show = cv2.resize(np.uint8(show), (1920, 1080), interpolation=cv2.INTER_LINEAR)
                img_ground = cv2.resize(np.uint8(img_ground), (1920, 1080), interpolation=cv2.INTER_LINEAR)
                gt = img_ground
                super = cv2.addWeighted(img_ground, 0.55, show, 0.35, 0)
                img_ground = cv2.cvtColor(img_ground, cv2.COLOR_BGR2RGB)
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

                cv2.putText(show, 'Segmentation', (600, 1020), 3, 3, (255, 255, 255), 4)
                cv2.putText(img_ground, 'Input Source', (600, 1020), 3, 3, (255, 255, 255), 4)
                cv2.putText(super, 'Superposition', (600, 1020), 3, 3, (255, 255, 255), 4)
                cv2.putText(gt, 'Ground Truth', (600, 1020), 3, 3, (255, 255, 255), 4)
                show = Image.fromarray(show)
                show = show.toqpixmap()
                img_ground = Image.fromarray(img_ground)
                img_ground = img_ground.toqpixmap()
                super = Image.fromarray(super)
                super = super.toqpixmap()
                gt = Image.fromarray(gt)
                gt = gt.toqpixmap()

                self.signal_update.emit(img_ground, show, super,gt, 1/interval)
        self.signal_finshed.emit()

    def set_source(self, source):
        self.source = source

    def set_stop(self,is_stop):
        self.stop = is_stop

    def set_cuda(self, is_cuda):
        self.is_cuda = is_cuda
        if self.is_cuda:
            self.net.cuda()
        else:
            self.net.cpu()

    def set_input_path(self, input):
        self.input_path = input

    def set_label_path(self, label):
        self.label_path = label

    def set_gt_path(self, gt):
        self.gt_path = gt

    def set_net(self, net):
        self.net_num = net
        print(net)
        self.init_net()
        self.net = load_model(self.net, self.save_pth)
        self.net.eval()
        if self.is_cuda:
            self.net.cuda()

    def set_dataset(self, dataset):
        self.dataset = dataset

        if self.dataset == 0:
            self.color_map = self.color_map_fish
        elif self.dataset == 1:
            self.color_map = self.color_map_city
        self.init_net()
        self.net = load_model(self.net, self.save_pth)
        self.net.eval()
        if self.is_cuda:
            self.net.cuda()

    def set_ckpt(self, path):
        self.save_pth = path
        try:
            self.net = load_model(self.net, self.save_pth)
            self.signal_message.emit(int(2))
        except RuntimeError:
            self.signal_message.emit(int(1))

    def set_video(self, path):
        self.mode = 0
        self.video_path = path

    def set_camera(self, num):
        self.mode = 1
        self.camera_num = num

    def init_net(self):
        respth = './res'
        if self.dataset == 0:
            self.num_classes = 18
            self.color_map = self.color_map_fish

            if self.net_num == 0: #B_x39
                self.net = BiSeNet_x39(out_planes=self.num_classes, is_training=False, criterion=None, ohem_criterion=None)
                self.save_pth = osp.join(respth, 'FT.pth')
            elif self.net_num == 1: #B_R18
                self.net = BiSeNet_r18(out_planes=self.num_classes, is_training=False, criterion=None)
                self.save_pth = osp.join(respth, 'Teacher.pth')

        elif self.dataset == 1:
            self.num_classes = 19
            self.color_map = self.color_map_city

            if self.net_num == 0: #B_x39
                self.net = BiSeNet_x39(out_planes=self.num_classes, is_training=False, criterion=None, ohem_criterion=None)
                self.save_pth = osp.join(respth, 'cityscapes-bisenet-X39.pth')
            elif self.net_num == 1: #B_R18
                self.net = BiSeNet_r18(out_planes=self.num_classes, is_training=False, criterion=None)
                self.save_pth = osp.join(respth, 'cityscapes-bisenet-R18.pth')
            elif self.net_num == 2:  # B_R101
                self.net = BiSeNet_r101(out_planes=self.num_classes, is_training=False, criterion=None)
                self.save_pth = osp.join(respth, 'cityscapes-bisenet-R101.pth')
            elif self.net_num == 3:  # P_R18
                pass
            elif self.net_num == 4:  # P_R101
                pass


    def predict_picture(self,net, img):
        h, w, _ = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img).unsqueeze(0)

        prob = torch.zeros((1, 19, h, w))
        prob.requires_grad = False

        if self.is_cuda:
            img = img.cuda()
        try:
            with torch.no_grad():
                prob = net(img)
        except RuntimeError:
            self.signal_message.emit(int(0))
            return False

        prob = prob[0].detach().cpu().numpy()
        pred = np.argmax(prob, axis=1)

        return pred

class VideoSetting(QtWidgets.QDialog, Ui_dlg):
    signal_video = pyqtSignal(str)
    signal_camera = pyqtSignal(int)

    def __init__(self,parent=None):
        super(VideoSetting, self).__init__(parent)
        self.setupUi(self)
        self.retranslateUi(self)
        self.mode = 0
        self.path = ''

        self.radioButton.clicked.connect(self.slot_videomode)
        self.radioButton_2.clicked.connect(self.slot_cameramode)
        self.pushButton.clicked.connect(self.slot_cancel)
        self.pushButton_2.clicked.connect(self.slot_ok)
        self.pushButton_3.clicked.connect(self.slot_selectvideo)

    def slot_videomode(self):
        self.mode = 0
        self.spinBox.setEnabled(False)
        self.pushButton_3.setEnabled(True)

    def slot_cameramode(self):
        self.mode = 1
        self.spinBox.setEnabled(True)
        self.pushButton_3.setEnabled(False)

    def slot_cancel(self):
        self.close()

    def slot_ok(self):
        if self.mode==0:
            self.signal_video.emit(self.path)
        elif self.mode==1:
            num = self.spinBox.value()
            self.signal_camera.emit(num)
        self.close()

    def slot_selectvideo(self):
        path, filetype = QFileDialog.getOpenFileName(self, 'Select Video', '', 'video files(*.avi)')
        if path:
            self.path = path


class SegWidget(QtWidgets.QMainWindow,Ui_MainWindow):

    def __init__(self):
        super(SegWidget, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        #os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        self.source = 0
        self.is_transfering = False
        self.dataset = 0 # Fish-0 city-1
        self.net = 0
        self.dymode = 0
        self.load_path = '/home/wangyunnan/PycharmProjects/Seg/res/FT.pth'

        self.input_path = "./res/1_Img8bit.png"
        self.label_path =  "./res/1_gtFine_labelTrainIds.png"
        self.gt_path = "./res/1_gtFine_color.png"

        self.input = cv2.imread(self.input_path)
        self.output = cv2.imread("./res/1_gtFine_color.png")
        self.super = cv2.addWeighted(self.input, 0.55, self.output, 0.35, 0)
        self.ground = cv2.imread(self.gt_path)

        self.input = self.add_label(self.input, "Input Source")
        self.output = self.add_label(self.output, "Segmentation")
        self.super = self.add_label(self.super, "Superposition")
        self.ground = self.add_label(self.ground, "Ground Truth")

        self.label.setScaledContents(True)
        self.label_2.setScaledContents(True)
        self.label_9.setScaledContents(True)
        self.label_11.setScaledContents(True)

        self.interval = 0.0
        self.label.setPixmap(self.input.scaled(QSize(1920 / 4, 1080 / 4)))
        #self.label_2.setPixmap(self.output .scaled(QSize(1920 / 4, 1080 / 4)))
        #self.label_9.setPixmap(self.super.scaled(QSize(1920 / 4, 1080 / 4)))
        self.label_2.setText("Segmentation")
        self.label_9.setText("Superposition")
        self.label_11.setPixmap(self.ground.scaled(QSize(1920 / 4, 1080 / 4)))
        self.label.setStyleSheet("border:2px solid red;")
        self.label_2.setStyleSheet("border:2px solid red;")
        self.label_9.setStyleSheet("border:2px solid red;")
        self.label_11.setStyleSheet("border:2px solid red;")

        self.label_14.setScaledContents(True)
        self.label_14.setPixmap(QPixmap('./res/school.png'))

        self._translate = QtCore.QCoreApplication.translate
        self.is_cuda = torch.cuda.is_available()

        if self.is_cuda:
            self.comboBox_2.setCurrentIndex(1)
            self.label_5.setText(self._translate("MainWindow", "Device Name: " + torch.cuda.get_device_name(0).replace("GeForce", "")))
            self.label_5.setText(self._translate("MainWindow", "Device Name: " + 'Xavier'))
        else:
            self.comboBox_2.removeItem(1)
            CPUinfo = self.CPUinfo()
            self.label_5.setText(self._translate("MainWindow", "Device Name: " + CPUinfo['model name'].split()[2]))

        if self.dataset == 0:
            self.comboBox_3.removeItem(2)
            #self.comboBox_3.removeItem(2)
            #self.comboBox_3.removeItem(2)

        self.comboBox_2.currentIndexChanged.connect(self.slot_change_device)
        self.comboBox.currentIndexChanged.connect(self.slot_change_dataset)
        self.comboBox_3.currentIndexChanged.connect(self.slot_change_net)
        self.pushButton.clicked.connect(self.slot_start)
        self.pushButton_2.clicked.connect(self.slot_select)
        self.pushButton_3.clicked.connect(self.slot_load)
        self.radioButton_2.clicked.connect(self.slot_source_image)
        self.radioButton.clicked.connect(self.slot_source_camera)
        self.inference_thread = QThread()

        self.sg = Inference(self.is_cuda)
        self.sg.signal_update.connect(self.slot_update)
        self.sg.signal_finshed.connect(self.slot_finshed)
        self.sg.signal_message.connect(self.slot_message)

        self.sg.moveToThread(self.inference_thread)

        self.inference_thread.started.connect(self.sg.work)
        #self.inference_thread.finished.connect(self.slot_finshed)

        self.videosetting = VideoSetting(self)
        self.videosetting.setModal(True)
        self.videosetting.signal_video.connect(self.slot_setvideo)
        self.videosetting.signal_camera.connect(self.slot_setcamera)
        #self.videosetting.setWindowFlag(QtCore.Qt.Tool)
        #self.setParent(self)
        #self.videosetting.close()

        #self.videosetting.setParent(self)


    def add_label(self, img, str):
        img = cv2.resize(np.uint8(img), (1920, 1080), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, str, (600, 1020), 3, 3, (255, 255, 255), 4)
        img = Image.fromarray(img)
        img = img.toqpixmap()
        return img

    def CPUinfo(self):
        procinfo = OrderedDict()
        with open('/proc/cpuinfo') as f:
            for line in f:
                if not line.strip():
                    # end of one processor
                    CPUinfo = procinfo
                    break
                else:
                    if len(line.split(':')) == 2:
                        procinfo[line.split(':')[0].strip()] = line.split(':')[1].strip()
                    else:
                        procinfo[line.split(':')[0].strip()] = ''
        return CPUinfo

    def slot_source_image(self):
        self.pushButton_2.setText("Select Image")
        self.source = 0
        self.sg.set_source(0)

    def slot_setvideo(self, path):
        self.dymode = 0
        self.sg.set_video(path)

    def slot_setcamera(self, num):
        self.dymode = 1
        self.sg.set_camera(num)


    def slot_message(self, num):
        if num == 0:
            QMessageBox.warning(self, "out of memory!", "Please use CPU or higher performance GPU.")
        elif num == 1:
            QMessageBox.warning(self, "Failed to load!", "Please select a checkpoint that matches the model.")
        elif num == 2:
            QMessageBox.warning(self, "Successfully loaded!", "Please make further settings.")
        elif num == 3:
            QMessageBox.warning(self, "Camera error!", "Please reselect a camera.")
        elif num == 4:
            QMessageBox.warning(self, "The video is over!", "Please proceed to the next step.")

    def slot_source_camera(self):
        self.pushButton_2.setText("Select Camera")
        self.source = 1
        self.sg.set_source(1)

    def slot_change_device(self, idx):
        if idx == 0:
            self.sg.set_cuda(False)
            CPUinfo = self.CPUinfo()
            self.label_5.setText(self._translate("MainWindow", "Device Name: " + CPUinfo['model name'].split()[2]))
        elif idx == 1:
            self.sg.set_cuda(True)
            self.label_5.setText(self._translate("MainWindow", "Device Name: " + torch.cuda.get_device_name(0).replace("GeForce", "")))

    def slot_change_dataset(self, idx):
        self.dataset = idx
        self.sg.set_dataset(self.dataset)

        if self.dataset == 0:
            self.comboBox_3.removeItem(2)
            #self.comboBox_3.removeItem(2)
            #self.comboBox_3.removeItem(2)
        elif self.dataset == 1:
            self.comboBox_3.addItem('BiSeNet(R101)')
            #self.comboBox_3.addItem('PSPNet(R18)')
            #self.comboBox_3.addItem('PSPNet(R101)')

    def slot_change_net(self, idx):
        self.net = idx
        self.sg.set_net(self.net)

    def slot_start(self):
        if self.is_transfering == False:
            self.sg.set_stop(False)
            self.inference_thread.start()
            self.pushButton.setText("Stop Transfering")
            self.is_transfering = True
            self.radioButton_2.setEnabled(False)
            self.radioButton.setEnabled(False)
            self.comboBox.setEnabled(False)
            self.comboBox_2.setEnabled(False)
            self.comboBox_3.setEnabled(False)
            self.pushButton_2.setEnabled(False)
            self.pushButton_3.setEnabled(False)
        else:
            self.sg.set_stop(True)


    def slot_finshed(self):
        print("finsh")
        self.inference_thread.quit()
        self.pushButton.setText("Start Transfer")
        self.is_transfering = False
        self.radioButton_2.setEnabled(True)
        self.radioButton.setEnabled(True)
        self.comboBox.setEnabled(True)
        self.comboBox_2.setEnabled(True)
        self.comboBox_3.setEnabled(True)
        self.pushButton_2.setEnabled(True)
        self.pushButton_3.setEnabled(True)

    def slot_select(self):
        if self.source == 0:
            path, filetype = QFileDialog.getOpenFileName(self, 'Select Image', '', 'image files(*.png , *.jpg)')
            if path:
                self.input_path = path
                if "leftImg8bit" in path:
                    self.label_path = path.replace('_leftImg8bit', '_gtFine_labelIds')
                    self.label_path = self.label_path.replace('leftImg8bit', 'gtFine')
                    self.gt_path = self.label_path.replace('labelIds', 'color')
                else:
                    self.label_path = path.replace('_Img8bit', '_gtFine_labelTrainIds')
                    self.label_path = self.label_path.replace('Img8bit', 'gtFine')
                    self.gt_path = self.label_path.replace('labelTrainIds', 'color')

                self.input = cv2.imread(self.input_path)
                self.ground = cv2.imread(self.gt_path)
                if self.input is not None:
                    self.input = self.add_label(self.input, "Input Source")
                    self.ground = self.add_label(self.ground, "Ground Truth")
                    self.label.setPixmap(self.input.scaled(QSize(1920 / 4, 1080 / 4)))
                    self.label_11.setPixmap(self.ground.scaled(QSize(1920 / 4, 1080 / 4)))
                    self.label_2.setText("Segmentation")
                    self.label_9.setText("Superposition")
                    self.sg.set_input_path(self.input_path)
                    self.sg.set_label_path(self.label_path)
                    self.sg.set_gt_path(self.gt_path)
        elif self.source == 1:
            self.videosetting.show()
            print('ok')

    def slot_load(self):
        path, filetype = QFileDialog.getOpenFileName(self, 'Select Checkpoint', '', 'ckpt files(*.pth)')
        if path:
            self.load_path = path
            self.sg.set_ckpt(self.load_path)

    def slot_update(self, ground,img,super,gt, time):
        self.label.setPixmap(ground.scaled(QSize(1920 / 4, 1080 / 4)))
        self.label_2.setPixmap(img.scaled(QSize(1920 / 4, 1080 / 4)))
        self.label_9.setPixmap(super.scaled(QSize(1920 / 4, 1080 / 4)))
        #if self.source == 1:
            #self.label_11.setText("No Ground Truth provided in real-time inference")
        self.label_11.setPixmap(gt.scaled(QSize(1920 / 4, 1080 / 4)))
        if self.source == 0:
            self.label_6.setText(self._translate("MainWindow", "Inference Time: {interval:.2f}s".format(interval = time)))
        elif self.source ==1:
            self.label_6.setText(self._translate("MainWindow", "Frames Per Second: {interval:.2f}".format(interval=time + 10)))

