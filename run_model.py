import numpy as np
from draw_detections import draw_det

def run(img,sess):
    img_data = np.expand_dims(img.astype(np.uint8), axis=0)
    outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]
    result = sess.run(outputs, {"image_tensor:0": img_data})
    num_detections, detection_boxes, detection_scores, detection_classes = result
    boxes=[]
    dc=[]
    ds=[]
    
    for detection in range(0, int(num_detections[0])):
        if(detection_classes[0][detection] not in [3,4,6,8] ) or detection_scores[0][detection]<0.35: continue
        d = detection_boxes[0][detection]
        img,db=draw_det(img,d,detection_scores[0][detection])
        boxes.append(db)
        dc.append(int(detection_classes[0][detection]))
        ds.append(int(detection_scores[0][detection]))
    
    return img,boxes,ds,dc