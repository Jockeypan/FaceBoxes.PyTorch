# -*- coding: utf-8 -*-
"""
 save_onnx.py
 @created 2019-06-10T19:31:19.174Z+08:00
 @last-modified 2019-06-13T17:56:15.939Z+08:00
 @author: Jockey Pan
 @E-mail: jockeypan@hotmail.com

 @description:
"""

import torch
import torchvision
from models.faceboxes import FaceBoxes

dummy_input = torch.randn(1, 3, 416, 608, device='cpu')
model = FaceBoxes(
    phase='trans', size=None, num_classes=2)  # initialize detector
pretrained_dict = torch.load('weights/FaceBoxes.pth', map_location='cpu')
model.load_state_dict(pretrained_dict, strict=False)
input_names = ["data"] + ["learned_%d" % i for i in range(147)]
output_names = ["l1", "c1", "l2", "c2", "l3", "c3"]
print(model)
torch.onnx.export(
    model,
    dummy_input,
    "FaceBoxes-416x608.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names)
