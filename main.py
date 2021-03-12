from time import *

import numpy as np
import cv2
import os
import argparse
import pathlib


class MaskDetection(object):
    config_path = "cfg/yolov3-voc-mask-detect.cfg"
    weights_path = "backup/yolov3-voc-mask-detect_final.weights"
    conf_threshold = 0.5
    nms_threshold = 0.4

    def __init__(self):
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detect_mask_from_image(self, image_path="sources/mask-detect-test.jpg", image=None, save_image=False):
        global x, y, w, h, mask, text
        mask = False
        if image_path != "":
            image = cv2.imread(image_path)
        (H, W) = image.shape[0: 2]
        blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (416, 416), swapRB=True, crop=False)

        layer_name = self.net.getLayerNames()
        unconnected_out_layers = self.net.getUnconnectedOutLayers()
        # print(unconnected_out_layers)


        temp = []
        for layers in unconnected_out_layers:
            temp.append(layer_name[layers[0] - 1])
        start = time()
        self.net.setInput(blob)
        layer_outputs = self.net.forward(temp)
        t = time() - start
        print("t:", t)
        # print("detections:", layer_outputs)

        boxes = []
        confidences = []
        class_indexes = []

        for output in layer_outputs:
            # print("output:", output)
            for detection in output:
                scores = detection[5:]
                # print(len(scores))
                # print(scores)
                class_index = np.argmax(scores)
                # print(class_index)
                confidence = scores[class_index]
                if confidence > 0.5:
                    # print(confidence)
                    mask = True
                    box = detection[0: 4] * np.array([W, H, W, H])
                    (cx, cy, w, h) = box.astype("int")

                    x = int(cx - w / 2)
                    y = int(cy - h / 2)

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_indexes.append(class_index)
        box_indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        k = 0
        for box_index in box_indexes:
            i = box_index[0]
            box = boxes[i]
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])
            # t = 28.0 * w / 23.0 - h
            # h = int(h + t / 2)
            # if y - t / 2 < 0:
            #     y = 0
            # elif y > image.shape[0] - h:
            #     y = image.shape[0] - h
            # else:
            #     y = int(y - t / 2)
            # print(w, h, w * 1.0 / h)
            # face_image = image[x: x + w][y: y + h]
            # face_image = cv2.resize(face_image, (92, 112))
            # face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            # result = rec.predict(face_image)
            # print(result)
            if class_indexes[k] == 0:
                color = [0, 255, 0]
                text = '%.2f%%' % (confidences[i] * 100)
            else:
                color = [0, 0, 255]
                text = '%.2f%%' % (confidences[i] * 100)
            k = k + 1
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
            print(w, h, w * 1.0 / h)
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

        if save_image:
            path = "sources" + image_path.split('.')[-2] + '.jpg'
            cv2.imread(path, image)
        # t, _ = self.net.getPerfProfile()
        # freq = cv2.getTickFrequency()
        # t = t / freq
        # print(t)
        return image

    def detect_mask_from_video(self, video_path="sources/mask-detect-test.mp4", save_video=False):
        global video_writer
        cap = cv2.VideoCapture(video_path)
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter("sources/temp.avi", fourcc, 5, (width, height), True)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                result_image = self.detect_mask_from_image("", frame)
                cv2.imshow("result", frame)
                if save_video:
                    video_writer.write(result_image)
                for i in range(0, 30):
                    ret, frame = cap.read()
                    if ret:
                        color = [0, 0, 255]
                        if mask:
                            color = [0, 255, 0]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
                        cv2.imshow("result", frame)
                        if save_video:
                            video_writer.write(result_image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        video_writer.release()

    def detect_mask_from_camera(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                result_image = self.detect_mask_from_image("", frame)
            cv2.imshow("result", result_image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    detection = MaskDetection()
    result_image = detection.detect_mask_from_image()
    cv2.imshow("result image", result_image)
    cv2.waitKey(0)

    # detection.detect_mask_from_video()

    # detection.detect_mask_from_camera()

    cv2.destroyAllWindows()
