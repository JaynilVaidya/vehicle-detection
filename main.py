import onnxruntime as rt
import os
import cv2
from load_data import dataloader
from preprocess import preprocess
from run_model import run

annotfilename = 'anontations.csv'
result_dict = dataloader('test',annotfilename)
sess = rt.InferenceSession('ssd_mobilenet_v1_10.onnx')
target=[]
preds= []

for filename, values in result_dict.items():
    target_labels = values['class']
    boxes_target = values['bbox_coordinates']
    img=cv2.imread(os.path.join('test',filename))
    output_image,boxes_pred,ds_preds,dc_preds=run(preprocess(img),sess)

    target.append(dict(
    boxes=boxes_pred,
    scores=ds_preds,
    labels=target_labels,
      ))

    preds.append(dict(
      filename=filename,
      boxes=boxes_pred,
      scores=ds_preds,
      labels=dc_preds,
        ))

