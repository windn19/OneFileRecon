import logging

import cv2
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from model import build_model
try:
    from settings import ocr_weights
except Exception as e:
    print('Ошибка чтения весов распознавания номеров', e)
    ocr_weights = ''


characters = [' ', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K',
              'M', 'N', 'O', 'P', 'S', 'T', 'X', 'Y', '_']
max_length = 9
img_width = 200
img_height = 50
char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
model = build_model()
if ocr_weights:
    model.load_weights(ocr_weights)
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
log = logging.getLogger('file1')


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
              :, :max_length
              ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def prepare_image(img):
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = canny(grayimg, sigma=3.0)
    out, angles, distances = hough_line(img1)
    h, theta, d = out, angles, distances
    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]
    _, angles_peaks, _ = hough_line_peaks(out, angles, distances, num_peaks=20)
    angle = np.mean(np.rad2deg(angles_peaks))
    if 0 <= angle <= 90:
        rot_angle = angle - 90
    elif -45 <= angle < 0:
        rot_angle = angle - 90
    elif -90 <= angle < -45:
        rot_angle = 90 + angle
    else:
        rot_angle = 0
    if abs(rot_angle) > 20:
        rot_angle = 0
    rotated = rotate(img, rot_angle, resize=True) * 255
    rotated = rotated.astype(np.uint8)
    lab = cv2.cvtColor(rotated, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    grayfinal = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    tfinal = cv2.threshold(grayfinal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    images1 = [img / 255, final / 255, cv2.cvtColor(tfinal, cv2.COLOR_GRAY2BGR) / 255]
    images1 = [cv2.resize(i, (img_width, img_height)) for i in images1]
    img2 = [np.expand_dims(i, axis=0) for i in images1]
    img2 = np.concatenate(img2, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    preds = prediction_model.predict(img2)
    pred_texts = decode_batch_predictions(preds)
    log.info(pred_texts)
    return pred_texts[0]

