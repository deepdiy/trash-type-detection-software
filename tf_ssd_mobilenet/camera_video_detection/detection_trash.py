import tensorflow as tf
import cv2
import numpy as np
from math import ceil
from sys import exit
from ssd_mobilenet_utils import *

PATH_TO_CKPT = '../models/frozen_inference_graph.pb'
SCORE_THRESHOLD = .76
class_names=['background','bottle','trash']

def get_graph():
    global PATH_TO_CKPT
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


def unpack_boxes(box, shape):
    height, width = shape
    y_min, x_min, y_max, x_max = tuple(box.tolist())
    y_min = ceil(height * y_min)
    y_max = ceil(height * y_max)
    x_min = ceil(width * x_min)
    x_max = ceil(width * x_max)
    return y_min, x_min, y_max, x_max


def resize_heart(frame, heart, boxes):
    y_min, x_min, y_max, x_max = unpack_boxes(boxes, frame.shape[:2])
    heart_size = (x_max - x_min, y_max - y_min)
    heart_draw = cv2.resize(heart, heart_size)
    alpha_heart = heart_draw[:, :, 2] / 255.0
    alpha_frame = 1.0 - alpha_heart

    for chan in range(0, 3):
        try:
            frame[y_min:y_max, x_min:x_max, chan] = (
                alpha_heart * heart_draw[:, :, chan] + alpha_frame *
                frame[y_min:y_max, x_min:x_max, chan])
        except ValueError:
            pass

    return frame


def get_available_cameras():
    camera_index = 0
    available_cameras = []

    print('Please wait while the program searches for available cameras...')
    while True:
        capture_test = cv2.VideoCapture(camera_index)
        if capture_test.read()[0]:
            available_cameras.append(camera_index)
            capture_test.release()
            print('Found camera with ID: [' + str(camera_index) + '].')
            camera_index += 1
        else:
            break

    return available_cameras

def draw_box(image,boxes,scores,classes):
    boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32)
    out_scores, out_boxes, out_classes = non_max_suppression(scores, boxes, classes)

    # Print predictions info
    #print('Found {} boxes for {}'.format(len(out_boxes), image_name))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    image = draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    return image

def run():
    global SCORE_THRESHOLD
    video_capture = cv2.VideoCapture(1)
    heart = cv2.imread('heart.png')

    detection_graph = get_graph()

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                ret, frame = video_capture.read()

                frame_expanded = np.expand_dims(frame, axis=0)
                image_tensor = detection_graph.get_tensor_by_name(
                    'image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                classes = detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')

                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: frame_expanded})

                draw_box(frame,boxes,scores,classes)

               
                cv2.imshow('Heart Gesture Detection (Press "q" to close.)',
                           frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break


if __name__ == "__main__":
    # selection = menu()

    run()
