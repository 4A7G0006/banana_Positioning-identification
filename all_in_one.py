from __future__ import division
import os
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras_frcnn import roi_helpers
import math

sys.setrecursionlimit(40000)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
"Location to read the metadata related to the training (generated when training).",
                  default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.",
                  default='resnet50')

(options, args) = parser.parse_args()

config_output_filename = options.config_filename

import keras_frcnn.resnet as nn

# turn off any data augmentation at test time


img_path = options.test_path


def format_img_size(img):
    """ formats the image size based on config """
    img_min_side = float(416)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img_channel_mean = [103.939, 116.779, 123.68]
    img[:, :, 0] -= img_channel_mean[0]
    img[:, :, 1] -= img_channel_mean[1]
    img[:, :, 2] -= img_channel_mean[2]
    img_scaling_factor = 1.0
    img /= img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img)
    img = format_img_channels(img)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


class_mapping = {'banana': 0, 'bg': 1}

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
num_rois = int(options.num_rois)

num_features = 1024
input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)

roi_input = Input(shape=(num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
anchor_box_scales = [128, 256, 512]
anchor_box_ratios = [[1, 1], [1. / math.sqrt(2), 2. / math.sqrt(2)], [2. / math.sqrt(2), 1. / math.sqrt(2)]]
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

model_path = './model_frcnn_0045.hdf5'
print(f'Loading weights from {model_path}')
model_rpn.load_weights(model_path, by_name=True)
model_classifier.load_weights(model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.8

visualise = True

# --------------------------

import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import imutils
from time import sleep

cap = cv2.VideoCapture("http://192.168.45.234:8080/?action=stream")
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] loading model....")
model = load_model('./vgg16_banana2.model')

while (True):

    ret, frame = cap.read()
    orig = frame.copy()
    X, ratio = format_img(frame)

    X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)
    C = ''  # C is nothing
    R = roi_helpers.rpn_to_roi(Y1, Y2, C, "tf", overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // num_rois + 1):
        ROIs = np.expand_dims(R[num_rois * jk:num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):

            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
                tx /= classifier_regr_std[0]
                ty /= classifier_regr_std[1]
                tw /= classifier_regr_std[2]
                th /= classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            rpn_stride = 16
            bboxes[cls_name].append(
                [rpn_stride * x, rpn_stride * y, rpn_stride * (x + w), rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []
    all_dets_banana = []
    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            cv2.rectangle(orig, (real_x1, real_y1), (real_x2, real_y2),
                          (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)
            all_dets_banana.append([(real_x1, real_y1), (real_x2, real_y2)])
            textLabel = f'{key}: {int(100 * new_probs[jk])}'
            all_dets.append((key, 100 * new_probs[jk]))
    # print(len(all_dets_banana)) #測試是否找到香蕉
    #print(frame.shape)
    if len(all_dets_banana) != 0:
        for i in all_dets_banana:
            (real_x1, real_y1), (real_x2, real_y2) = i
            hh, ww, dd = frame.shape
            if real_x1 < 0:
                real_x1 = 0
            if real_x2 < 0:
                real_x2 = 0
            if real_y1 < 0:
                real_y1 = 0
            if real_y2 < 0:
                real_y2 = 0
            if real_x1 > ww:
                real_x1 = ww
            if real_x2 > ww:
                real_x2 = ww
            if real_y1 > hh:
                real_y1 = hh
            if real_y2 > hh:
                real_y2 = hh    #避免破圖
            # cv2.imshow("Output",frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # 裁切圖片
            # print(type(frame))
            crop_img = frame[real_y1:real_y2, real_x1:real_x2]
            # cv2.imshow("Output",crop_img)
            # print(type(crop_img))
            cv2.rectangle(orig, (real_x1, real_y1), (real_x2, real_y2),
                          (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)

            image = cv2.resize(crop_img, (224, 224))    #重設大小
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            (immature, ripe, rotten) = model.predict(image)[0]
            if immature > ripe and immature > rotten:
                label = "immature"
                proba = immature
            elif ripe > immature and ripe > rotten:
                label = "ripe"
                proba = ripe
            else:
                label = "rotten"
                proba = rotten
            label = "{}: {:.2f}%".format(label, proba * 100)
            cv2.putText(orig, label, (real_x1, real_y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
    cv2.imshow("Output", orig)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# -----------------------------
