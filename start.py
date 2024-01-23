from os import walk, remove, listdir
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from detect import load_model, run


model = load_model(weights='bestR5.pt')

print(run(model=model, source='test_image_part1/', save_crop=True, nosave=True))


