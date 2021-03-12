import random
import sys
from queue import Queue
from threading import Thread
from time import time, sleep

import cv2

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from main import *

import darknet
from MainWidget import Ui_Form


class FaceRecognize(object):

    def __init__(self):
        self.model = cv2.face.EigenFaceRecognizer_create()
        self.model.read("backup/face-recognize.xml")

    def resize_aspct_ratio(self, src, size):
        src_h, src_w = src.shape[:2]
        print(src_h, src_w)
        dst_h, dst_w = size

        h = dst_w * 1.0 / src_w * src_h
        w = dst_h * 1.0 / src_h * src_w

        h = int(h)
        w = int(w)

        if h <= dst_h:
            dst = cv2.resize(src, (dst_w, h))
        else:
            dst = cv2.resize(src, (w, dst_h))

        h, w = dst.shape[:2]

        up = int((dst_h - h) / 2)
        down = int((dst_h - h + 1) / 2)
        left = int((dst_w - w) / 2)
        right = int((dst_w - w + 1) / 2)

        value = [240, 240, 240]
        dst = cv2.copyMakeBorder(dst, up, down, left, right, cv2.BORDER_CONSTANT, None, value)

        dst = cv2.resize(dst, (size[1], size[0]))

        return dst

    def predict(self, image, detections):
        face_detections = []
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        for detection in detections:
            label, confidence, bbox = detection
            x, y, w, h = list(map(int, bbox))
            print(x, y, w, h)
            face = image[x: x + w, y: y + h]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            face_resized = self.resize_aspct_ratio(face, (112, 92))
            face_resized_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            result = self.model.predict(face_resized_gray)
            detection = detection + result
            face_detections.append(detection)
        print(face_detections)

        return face_detections


class Detection(object):

    def __init__(self):
        self.network, self.class_names, self.class_colors = darknet.load_network(
            config_file="./cfg/yolov3-voc-mask-detect.cfg",
            data_file="./data/voc-mask-detect.data",
            weights="./backup/yolov3-voc-mask-detect_final-extend.weights",
            batch_size=1
        )
        self.mask_detection = MaskDetection()
        self.drew_images = Queue()
        self.input_video = "sources/mask-detect-test.mp4"
        self.output_video = None
        self.frame_queue = Queue()
        self.frame_queue_face = Queue()
        self.face_image_queue = Queue(maxsize=1)
        self.darknet_image_queue = Queue(maxsize=1)
        self.detections_queue = Queue(maxsize=1)
        self.face_detections_queue = Queue(maxsize=1)
        self.fps_queue = Queue(maxsize=1)
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.cap = cv2.VideoCapture(0)
        self.darknet_image = darknet.make_image(self.width, self.height, 3)
        self.thresh = 0.5
        # self.face_model = FaceRecognize()

    # def set_saved_video(self, input_video, output_video, size):
    #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    #     fps = int(input_video.get(cv2.CAP_PROP_FPS))
    #     video = cv2.VideoWriter(output_video, fourcc, fps, size)
    #     return video

    def resize_aspct_ratio(self, src, size):
        src_h, src_w = src.shape[:2]
        dst_h, dst_w = size

        h = dst_w * 1.0 / src_w * src_h
        w = dst_h * 1.0 / src_h * src_w

        h = int(h)
        w = int(w)

        if h <= dst_h:
            dst = cv2.resize(src, (dst_w, h))
        else:
            dst = cv2.resize(src, (w, dst_h))

        h, w = dst.shape[:2]

        up = int((dst_h - h) / 2)
        down = int((dst_h - h + 1) / 2)
        left = int((dst_w - w) / 2)
        right = int((dst_w - w + 1) / 2)

        value = [240, 240, 240]
        dst = cv2.copyMakeBorder(dst, up, down, left, right, cv2.BORDER_CONSTANT, None, value)

        return dst

    def image_process(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = self.resize_aspct_ratio(image_rgb, (self.width, self.height))
        image_face_resized = self.resize_aspct_ratio(image, (self.width, self.height))
        self.frame_queue.put(image_resized)
        darknet.copy_image_from_bytes(self.darknet_image, image_resized.tobytes())
        self.darknet_image_queue.put(self.darknet_image)
        self.frame_queue_face.put(image_face_resized)

    def video_capture(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.image_process(frame)

    def transfer_learning(self, detections, image):
        path = r"C:\Users\Demo\Desktop\darknet\VOCdevkit\VOC2007\JPEGImages"
        count = len(os.listdir(path))
        cv2.imwrite(path + str(count + 1) + r".jpg")
        f = open(r"C:\Users\Demo\Desktop\darknet\VOCdevkit\VOC2007\labels" + str(count + 1) + r".txt")
        for x, y, h, w in detections:
            s = str(x) + ' ' + str(y) + ' ' + str(h) + ' ' + str(w)
            f.write(s + '\n')


        if count >= 1000:
            os.system("darknet partial cfg/yolov3-voc-mask-detect.cfg backup/yolov3-voc-mask-detect_final.weights "
                      "yolov3.conv.81 81")
            os.system("darknet detector train data/voc-mask-detect.data cfg/yolov3-voc-mask-detect.cfg"
                      "backup/yolov3.conv.81")

    def predict(self):
        while self.cap.isOpened():
            darknet_image = self.darknet_image_queue.get()
            prev_time = time()
            detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)
            print("detections:", detections)
            # self.transfer_learning(detections, darknet_image)

            # -------------------------------------------
            # frame_face = self.frame_queue_face.get()
            # self.frame_queue_face.put(frame_face)
            # face_detections = self.face_model.predict(frame_face, detections)
            # print("face_detections:", face_detections)
            # self.face_detections_queue.put(face_detections)
            # -------------------------------------------
            self.detections_queue.put(detections)

            fps = int(1 / (time() - prev_time))
            self.fps_queue.put(fps)
            print("FPS: {}".format(fps))

    def draw(self):

        frame_resized = self.frame_queue.get()
        detections = self.detections_queue.get()
        print("draw")
        # -------------------------------------------
        # face_detections = self.face_detections_queue.get()
        # -------------------------------------------
        # print(detections)
        fps = self.fps_queue.get()
        if frame_resized is not None:
            image = darknet.draw_boxes(detections, frame_resized, self.class_colors)
            # -------------------------------------------
            # face_image = self.frame_queue_face.get()
            # face_image = darknet.draw_faces(face_detections, face_image, (252, 240, 2))
            # -------------------------------------------
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.drew_images.put(image)
            # -------------------------------------------
            # self.face_image_queue.put(face_image)
            # cv2.imshow('face recognizer', face_image)
            # -------------------------------------------
        cv2.waitKey(fps)

    def drawing(self, flag=0):
        random.seed(3)  # deterministic bbox colors
        # video = self.set_saved_video(self.cap, self.output_video, (self.width, self.height))
        if flag:
            self.draw()
        else:
            while self.cap.isOpened():
                frame_resized = self.frame_queue.get()
                detections = self.detections_queue.get()
                # -------------------------------------------
                # face_detections = self.face_detections_queue.get()
                # -------------------------------------------
                fps = self.fps_queue.get()
                if frame_resized is not None:
                    image = darknet.draw_boxes(detections, frame_resized, self.class_colors)
                    # -------------------------------------------
                    # face_image = self.frame_queue_face.get()
                    # face_image = darknet.draw_faces(face_detections, face_image, (252, 240, 2))
                    # -------------------------------------------
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # if self.output_video is not None:
                    #     video.write(image)
                    self.drew_images.put(image)
                    # -------------------------------------------
                    # self.face_image_queue.put(face_image)
                    # cv2.imshow('face recognizer', face_image)
                    # -------------------------------------------
                    if cv2.waitKey(fps) == 27:
                        break
        # video.release()
        cv2.destroyAllWindows()


class MainWindow(QMainWindow):
    """
    主窗口
    """

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.detection = Detection()
        self.t1 = Thread()
        self.t2 = Thread()
        self.t3 = Thread()
        self.t4 = Thread()

    def cvimage2qimage(self, cv_image):
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        q_image = QImage(image_rgb.data,  # 数据源
                         image_rgb.shape[1],  # 宽度
                         image_rgb.shape[0],  # 高度
                         image_rgb.shape[1] * 3,  # 行字节数
                         QImage.Format_RGB888)
        return q_image

    def show_images(self):
        while self.detection.drew_images is not None:
            cv_image = self.detection.drew_images.get()
            q_image = self.cvimage2qimage(cv_image)
            self.ui.detectDisplayLab.setPixmap(QPixmap.fromImage(q_image))

    def show_faces(self):
        while self.detection.face_image_queue is not None:
            cv_image = self.detection.face_image_queue.get()
            cv2.imshow("face recognizer", cv_image)
            if cv2.waitKey(0) == 27:
                break

    def open_camera_click(self):
        Thread(target=self.detection.video_capture).start()
        Thread(target=self.detection.predict).start()
        Thread(target=self.detection.drawing).start()
        Thread(target=self.show_images).start()

    def open_image_click(self):
        input_image, _ = QFileDialog.getOpenFileName(self, "选择图片", "./sources/", "ImageFile(*.jpg *.png "
                                                                                 "*.jpeg)")
        # if input_image == "":
        #     return
        # image = cv2.imread(input_image)
        #
        # Thread(target=self.detection.image_process, args=(image,)).start()
        # Thread(target=self.detection.predict).start()
        # Thread(target=self.detection.drawing, args=(1, )).start()
        # Thread(target=self.show_images).start()
        result = self.detection.mask_detection.detect_mask_from_image()
        q_image = self.cvimage2qimage(result)
        self.ui.detectDisplayLab.setPixmap(QPixmap.fromImage(q_image))

    def open_video_click(self):
        input_video, _ = QFileDialog.getOpenFileName(self, "选择视频", "./sources/", "VideoFile(*.mp4 *.avi "
                                                                                 "*.mpeg *.flv)")
        if input_video == "":
            return

        self.detection.cap = cv2.VideoCapture(input_video)

        self.t1 = Thread(target=self.detection.video_capture)
        self.t2 = Thread(target=self.detection.predict)
        self.t3 = Thread(target=self.detection.drawing)
        self.t4 = Thread(target=self.show_images)
        # self.t4 = Thread(target=self.show_faces)

        self.t1.start()
        self.t2.start()
        self.t3.start()
        self.t4.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
