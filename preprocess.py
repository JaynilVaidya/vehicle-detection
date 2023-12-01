import numpy as np
import cv2



def preprocess(test,t1=20, t2=80, clip_percent=20):
    
    #cany
    canyimg = cv2.Canny(test, t1, t2)
    dilated = cv2.dilate(canyimg, np.ones((3, 3)), iterations=10)
    eroded = cv2.erode(dilated, np.ones((3, 3)), iterations=7)

    #foreground extraction
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #test=cv2.cvtColor(test,cv2.COLOR_BGR2RGB)
    mask=cv2.fillPoly(test.copy(), contours, (0,0,0))
    test=test-mask

    #unsharp masking
    smooth = cv2.GaussianBlur(test, (3, 3), 1.0)
    unsharp = cv2.addWeighted(test, 3, smooth, -2, 0)
    
    #contrast and brightness adjustment
    gray = cv2.cvtColor(test.copy(), cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    maxhist = hist.sum()
    clip_percent *= (maxhist / 100.0) / 2.0
    
    min_gray = next(i for i, val in enumerate(hist) if val >= clip_percent)
    max_gray = next(i for i, val in enumerate(hist[::-1]) if val >= (maxhist - clip_percent))
    
    alpha = 255 / (max_gray - min_gray)
    beta = -min_gray * alpha
    
    processed = cv2.convertScaleAbs(unsharp, alpha=alpha, beta=beta)

    return processed
