# Enhanced Image Processing for Vehicle Detection

This project focuses on developing an image preprocessing pipeline to improve the detection accuracy of state-of-the-art (SOTA) models for vehicle detection. The dataset used includes various vehicle classes such as cars, buses, motorcycles, ambulances, and trucks.

## Project Structure
```
.
├── draw_detections.py
├── iou.py
├── load_data.py
├── main.py
├── preprocess.py
├── README.dataset.txt
├── README.roboflow.txt
├── run_model.py
├── ssd_mobilenet_v1_10.onnx
└── test/
    └── annotations.csv
```

## Files Description

- **draw_detections.py**: Contains the function to draw bounding boxes on images.
- **iou.py**: Contains the function to calculate Intersection over Union (IoU) between bounding boxes.
- **load_data.py**: Contains the function to load and preprocess the dataset annotations.
- **main.py**: The main script to run the preprocessing pipeline and model inference.
- **preprocess.py**: Contains the image preprocessing pipeline to enhance detection accuracy.
- **README.dataset.txt**: Provides details about the dataset used.
- **README.roboflow.txt**: Provides details about the dataset export via Roboflow.
- **run_model.py**: Contains the function to run the model inference and draw detections.
- **ssd_mobilenet_v1_10.onnx**: The ONNX model file used for vehicle detection.
- **test/annotations.csv**: Contains the annotations for the test images.

## Preprocessing Pipeline

The preprocessing pipeline in [`preprocess.py`](preprocess.py) includes the following steps:

1. **Canny Edge Detection**: Detects edges in the image for foreground extraction.
2. **Dilation and Erosion**: Used to get rid of blobs formed in foreground extraction to get a single foreground and background.
3. **Foreground Extraction**: Extracts the foreground objects using contours.
4. **Unsharp Masking**: Enhances the sharpness of the image.
5. **Contrast and Brightness Adjustment**: Adjusts the contrast and brightness of the image.

## Running the Project

1. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
2. **Run the Main Script**:
   ```sh
   python main.py
   ```

## Dataset

The dataset used is the Vehicles-OpenImages dataset, which includes 627 images of various vehicle classes. The dataset was exported via Roboflow and annotated in Tensorflow Object Detection format.

For more details, refer to `README.dataset.txt` and `README.roboflow.txt`.