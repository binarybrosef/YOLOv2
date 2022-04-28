import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image


def filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """
    Filter every box in every cell by comparing highest class probability in that box to threshold.

    Arguments
    ---------
    boxes: tensor of shape (19,19,5,4)             
        midpoint (x,y) and dimensions (height,width) of each box in each cell
    box_confidence: tensor of shape (19,19,5,1)   
        probability that an object is present in each box in each cell
    box_class_probs: tensor of shape (19,19,5,80) 
        class probabilities for each box in each cell
    threshold: float
        if (highest class probability score < threshold), filter out box

    Returns
    -------
    scores: tensor of shape (None,) 
        class probability scores for selected boxes
    boxes: tensor of shape (None, 4) 
        coordinates (x,y,h,w) of selected boxes
    classes: tensor of shape (None,)
        index of class detected by selected boxes
    """
    
    box_scores = box_confidence * box_class_probs       #(19,19,5,80); scores for each cell, each box, each class
    
    # Get index, and value itself, of max class score for each box in each cell
    box_classes = tf.math.argmax(box_scores, axis=-1)                #(19,19,5); indices of classes having max probabilities
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)       #(19,19,5); scores for classes having max probabilities  
        
    # Create mask to filter out box scores below threshold
    mask = box_class_scores > threshold                              #(19,19,5)   
           
    # Apply mask to box_class_scores, boxes and box_classes; masked out values are discarded
    scores = tf.boolean_mask(box_class_scores, mask)  
    boxes = tf.boolean_mask(boxes, mask)
    classes = tf.boolean_mask(box_classes, mask)

    return scores, boxes, classes


def non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies non-max suppression (NMS) to boxes.
    
    Arguments
    ---------
    scores: tensor of shape (None,)
        class probability scores for selected boxes
    boxes: tensor of shape (None, 4)
        coordinates (x,y,h,w) of selected boxes scaled to image_size 
    classes: tensor of shape (None,)
        index of class detected by selected boxes
    max_boxes: integer
        maximum number of predicted boxes to output
    iou_threshold: float
        IoU threshold for NMS
    
    Returns
    -------
    scores: tensor of shape (, None)
        predicted score for each box
    boxes: tensor of shape (4, None)
        predicted box coordinates
    classes: tensor of shape (, None)
        predicted class for each box
    """

    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')     
      
    # Get indices for output boxes
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    
    # Select elements by index according to nms_indices
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    return scores, boxes, classes


def boxes_to_corners(box_xy, box_wh):
    """
    Convert box midpoint coordinates (x,y) and box dimensions (w,h) to box corner coordinates (x1,x2,y1,y2).

    Arguments
    ---------
    box_xy: tensor of shape (1,19,19,5,2)
        x/y coordinates of midpoint of boxes
    box_wh: tensor of shape (1,19,19,5,2)
        width, height of boxes

    Returns
    -------
    boxes: tensor of shape (1,19,19,5,4)
        corner coordinates of boxes
    
    """
    box_min = box_xy - (box_wh / 2.)
    box_max = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_min[..., 1:2],  # y1
        box_min[..., 0:1],  # x1
        box_max[..., 1:2],  # y2
        box_max[..., 0:1]   # x1
    ])


def yolo_head(feats, anchors, num_classes):
    """
    Convert YOLO model output features to bounding box parameters.

    Arguments
    ----------
    feats: tensor
        final convolutional layer features
    anchors: array-like
        anchor box widths and heights
    num_classes: int
        number of target classes

    Returns
    -------
    box_xy: tensor
        x,y box predictions adjusted by spatial location in conv layer
    box_wh: tensor
        w,h box predictions adjusted by anchors and conv spatial resolution
    box_conf: tensor
        probability estimate for whether each box contains any object
    box_class_pred: tensor
        probability distribution estimate for each box over class labels
    """

    num_anchors = len(anchors)

    # Reshape to (batch, height, width, num_anchors, box_params).
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

    # feats is of shape (m,19,19,5,85)
    # shape(feats)[1:3] results in a (19,19) tensor consisting of the second and third dimensions
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])

    # Construct a new conv_height_index tensor by replicating conv_height_index 
    # conv_dims[1] times - i.e., 19 times.
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # Insert dimension of length 1 at axis=0
    # Then tile expanded conv_width_index by (19,1)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])

    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.softmax(feats[..., 5:])

    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_eval(yolo_outputs, img_shape = (720,1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts YOLO encoding to predicted boxes with scores, coordinates, and classes.
    
    Arguments
    ---------
    yolo_outputs: tuple of length 4
        output of YOLO including 4 tensors:
            box_xy: tensor of shape (None,19,19,5,2)
            box_wh: tensor of shape (None,19,19,5,2)
            box_confidence: tensor of shape (None,19,19,5,1)
            box_class_probs: tensor of shape (None,19,19,5,80)

    img_shape: list of shape (2,)
        shape of input image

    max_boxes: int 
        maximum number of predicted boxes 

    score_threshold: float 
        if (highest class probability score < threshold), omit corresponding box

    iou_threshold: float 
        IoU threshold for NMS filtering
    
    Returns
    -------
    scores: tensor of shape (None, )
        predicted score for each box
    boxes: tensor of shape (None, 4)
        predicted box coordinates
    classes: tensor of shape (None,)
        predicted class for each box
    """
    
    # Get YOLO model outputs
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    
    # Convert box midpoint coordinates (x,y) and box dimensions (w,h) to corner coordinates
    boxes = boxes_to_corners(box_xy, box_wh)

    # Filter by score using score_threshold 
    scores, boxes, classes = filter_boxes(boxes, box_confidence, box_class_probs, threshold=score_threshold)
    
    # Scale boxes based on img_shape
    boxes = scale_boxes(boxes, img_shape)
    
    # Perform non-max suppression with max_boxes and iou_threshold 
    scores, boxes, classes = non_max_suppression(scores, boxes, classes, max_boxes=max_boxes, 
                                                      iou_threshold=iou_threshold)
    
    return scores, boxes, classes


def predict(image_file, max_boxes=10, score_threshold=.4, iou_threshold=.5):
    """
    Runs YOLO algorithm on image_file, producing score-filtered and non-max-suppressed 
    boxes rendered over image_file. Resultant image is saved in /predict folder.
    
    Arguments
    ---------
    image_file: name of image in /images folder.
    
    Returns
    -------
    scores: tensor of shape (None, )
        scores for predicted boxes
    boxes: tensor of shape (None, 4)
        coordinates of predicted boxes
    classes: tensor of shape (None, )
        class index of predicted boxes
    """

    # Preprocess image_file (resize, normalize, add batch dimension)
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
    
    # Get YOLO model encoding, process with yolo_head, and obtain resultant scores/boxes/classes
    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
    
    scores, boxes, classes = yolo_eval(yolo_outputs, [image.size[1],  image.size[0]], max_boxes,
        score_threshold, iou_threshold)

    # Output predicted box coordinates
    print(f'{len(boxes)} boxes detected for file {"images/" + image_file}')

    # Render boxes over image_file
    colors = get_colors_for_classes(len(class_names))
    draw_boxes(image, boxes, classes, class_names, scores)

    # Save resultant image with rendered boxes in /predict folder
    image.save(os.path.join("predict", image_file), quality=100)
    
    return scores, boxes, classes


################################
###----------Script----------###
################################

class_names = read_classes("model/classes.txt")
anchors = read_anchors("model/yolo_anchors.txt")
model_image_size = (608, 608)

# Load pre-trained YOLOv2 model. Model takes input images of shape (m,608,608,3) 
# and outputs feature tensor of shape (m,19,19,5,85). Apply model to user-specified image to
# obtain predicted bounding boxes for image.
yolo_model = load_model('model/yolo.h5')
predict("image.jpg")