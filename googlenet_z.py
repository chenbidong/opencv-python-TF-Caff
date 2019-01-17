# coding:utf-8
from __future__ import print_function
import numpy as np
import cv2
from cv2 import dnn
import sys

cm_path = 'caffemodel/'


def get_class_list():
    with open (cm_path + 'bvlc_googlenet_synset_words_cn1.txt', 'rt') as f:
        return [x[x.find (" ") + 1:] for x in f]


def dnn_show_result(prob, classes, n):
    y = sorted (prob[0], reverse=True)  # 从大到小排序
    #异常踩坑：# 前n名 # range(50)
    # range(50)报错，因为range返回range对象，不返回数组对象
    # 将range对象转换为list列表对象在进行遍历
    z = list(range (n))
    for i in range (0, n):
        z[i] = np.where (prob[0] == y[i])[0][0]
        print (u"第", i + 1, u"匹配：", classes[z[i]], end='')
        print (u"类所在行：", z[i] + 1, "  ", u"可能性:", y[i])


if __name__ == "__main__":
    if len (sys.argv) < 2:
        print ("USAGE: googlenet.py images/tiger.jpg")
        sys.exit ()
    print("dddd")
    fn = sys.argv[1]
    blob = dnn.blobFromImage (cv2.imread (fn), 1, (224, 224), (104, 117, 123))
    print ("Input:", blob.shape, blob.dtype)

    net = dnn.readNetFromCaffe (cm_path + 'bvlc_googlenet.prototxt', cm_path + 'bvlc_googlenet.caffemodel')
    net.setInput (blob)
    prob = net.forward ()
    print ("Output:", prob.shape, prob.dtype)

    classes = get_class_list ()
    dnn_show_result (prob, classes, 3)
