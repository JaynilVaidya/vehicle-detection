def calc_iou(boxes1,boxes2): #1:target, 2:predicted
    iou_values = []

    for box1 in boxes1:
        maxiou=0.0
        for box2 in boxes2:
            # Calculate intersection coordinates
            x_i = max(box1[0], box2[0])
            y_i = max(box1[1], box2[1])
            x_f = min(box1[2], box2[2])
            y_f = min(box1[3], box2[3])

            intersection = max(0, x_f - x_i) * max(0, y_f - y_i)

            # Calculate areas of both bounding boxes
            area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

            union = area_box1 + area_box2 - intersection

            # Calculate IoU
            maxiou = max(maxiou,intersection / union if union > 0 else 0.0)
        iou_values.append(maxiou)

    return iou_values


