import cv2
import numpy as np

from adaptive_publisher.models.base import BaseModel



class CVDiffModel(BaseModel):
    def __init__(self):
        super().__init__()
    def predict(self, new_image_frame, last_key_frame=None):
        perc_difference = 1.0
        if last_key_frame is not None:
            abs_diff = cv2.absdiff(last_key_frame, new_image_frame)

            abs_diff_int = abs_diff.astype(np.uint8)

            perc_difference = np.count_nonzero(abs_diff_int)/ abs_diff_int.size
        return perc_difference

