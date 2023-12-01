import numpy as np
import cv2


def draw_det(img,det_boxes, conf):
  width,height=img.shape[-2],img.shape[-3]
  top = max(0, np.ceil(det_boxes[0] * height).astype('int32'))
  left = max(0, np.ceil(det_boxes[1] * width).astype('int32'))
  bottom = min(height, np.ceil(det_boxes[2] * height).astype('int32'))
  right = min(width, np.ceil(det_boxes[3] * width).astype('int32'))

  img=cv2.rectangle(img, (left,top), (right,bottom), color=(255, 0, 0), thickness=2)

  return img,[left,top,right,bottom]
