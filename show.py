# -*- coding: utf-8 -*-
"""
 show.py
 @created Mon Jan 21 2019 17:11:14 GMT+0800 (CST)
 @author: Jimmy Pan
 @E-mail: jimmypan@forwardx.com

 @description:
"""

import os
import argparse
import collections
import cv2

parser = argparse.ArgumentParser(description='FaceBoxesShow')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--dataset', default='FWDX', type=str, choices=['AFW', 'PASCAL', 'FDDB','FWDX'], help='dataset')
args = parser.parse_args()

if __name__ == '__main__':
    # testing dataset
    testset_folder = os.path.join('data', args.dataset, 'images/')
    testset_result = os.path.join(args.save_folder, args.dataset+'_dets.txt')
    fr = open(testset_result, 'r')
    all_dets = fr.readlines()

    dets_dict = collections.OrderedDict()
    for i, one_line in enumerate(all_dets):
        one_det = one_line.rstrip().split()
        key = one_det[0]
        val = []
        val.append(float(one_det[1]))
        for idx in range(2, 6):
            val.append(int(round(float(one_det[idx]))))
        if key in dets_dict.keys():
            dets_dict[key].append(val)
        else:
            dets_dict[key] = []
            dets_dict[key].append(val)

    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    for img_name, dets_of_one_img in dets_dict.items():
        image_path = testset_folder + img_name + '.jpg'
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        for one_rt in dets_of_one_img:
            if one_rt[0] > 0.5:
                cv2.putText(img, str(one_rt[0]), tuple(one_rt[1:3]),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),1)
                cv2.rectangle(img, tuple(one_rt[1:3]), tuple(one_rt[3:5]), (0, 255, 0),2)
        cv2.imshow("test",img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
