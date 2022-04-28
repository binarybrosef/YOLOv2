# YOLOv2
Object detection with YOLOv2 implemented in Python and Tensorflow.

## Objective
Real-time object detection has become an increasing area of focus in computer vision and machine learning. While real-time object detection can be found in many different applications, the power of and need for real-time object detection is perhaps no better exemplified by autonomous driving, a rapidly burgeoning technology for which many high-profile companies are competing to deliver to consumers en masse. 

This repository implements a pre-trained neural network-based algorithm, proven for its high performance in real-time object detection, to identify and delineate objects in static images.

## The YOLOv2 Algorithm
A large variety of object detection algorithms have been developed, including neural network-based approaches. Among the most popular of the neural network-based approaches is the "You Only Look Once" (YOLO) algorithm, whose name refers to the algorithm's ability to output encodings with one single forward propagation through its network. 

In this implementation of the YOLOv2 algorithm, input images consist of color images resized to 608x608. A grid consisting of 19x19 cells is overlaid upon an input image, with five bounding boxes being provided in each cell. Each bounding box is represented by (1) a probability that an object is present in the box, (2) four coordinates that encode the x/y midpoint and height/width of the box, and (3) a set of class probabilities that each indicate the probability that a corresponding object class is present in the box.  

For typical image types to which YOLOv2 may be applied, such as images that capture traffic from a forward-facing car-mounted camera, the algorithm may produce a number of boxes that is visually and informationally excessive. As such, a typical YOLO implementation performs filtering to reduce the number of boxes in a final output image. First, boxes may be filtered by their object probability scores as compared to a threshold. Second, multiple overlapping boxes may be filtered by comparing object probability scores and their degree of overlap as measured by intersection-over-union (IoU).

## Application and Dataset
This implementation of YOLOv2 detects eighty different object classes derived from the [COCO dataset](https://cocodataset.org/). Class names are retrieved from `/model/classes.txt`.

Bounding box sizes are selected that are suitable for detecting vechicles and other objects captured from the perspective of a forward-facing car-mounted camera that captures scenes of driving and traffic. These bounding boxes sizes are encoded in `/model/yolo_anchors.txt`.

## Loading a YOLOv2 Model
A pre-trained YOLOv2 model named `yolo.h5` is expected in the `/model` folder. One way to obtain this model is to: 
- download YOLOv2 weights from [https://pjreddie.com/media/files/yolov2.weights](https://pjreddie.com/media/files/yolov2.weights)
- from [Allan Zelener's YAD2K Github repository](https://github.com/allanzelener/YAD2K), download `yad2k.py`
- in the folder containing `yad2k.py`, execute the following:
```
/yad2k.py yolo.cfg yolov2.weights model_data/yolo.h5
```
- finally, place the `yolo.h5` file in the `/models` directory.

## Script Use
`script.py` loads a pre-trained YOLOv2 model located at `/model/yolo.h5` and applies the model to a user-specified image of default name `image.jpg` located in the `/images` directory. A resultant output image is saved in the `/predict` directory, consisting of the original user-specified image with bounding boxes produced by the YOLOv2 model indicating detected objects and their spatial boundaries. 

`script.py` implements custom functions including:
- `filter_boxes()`, for filtering boxes by object probability
- `non_max_suppression()`, for filtering boxes via Tensorflow's `tf.image.non_max_suppression()`

`script.py` also implements, and imports from `utils.py`, functions available from [Allan Zelener's YAD2K Github repository](https://github.com/allanzelener/YAD2K) for various conversions, formatting, and rendering. 
