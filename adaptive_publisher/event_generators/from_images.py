import glob
import os
import re
import shutil

import cv2
import requests


from adaptive_publisher.conf import TEMP_IMG_PATH, TMP_IGNORE_N_FRAMES
from adaptive_publisher.event_generators.base import OCVEventGenerator



class LocalOCVEventGenerator(OCVEventGenerator):
    def __init__(self, service, ef_pipeline_name, file_storage_cli, publisher_id, input_source, fps, width, height, thresholds):
        super().__init__(service, ef_pipeline_name, file_storage_cli, publisher_id, input_source, fps, width, height, thresholds)
        source_dirname = os.path.dirname(input_source)
        self.source_uri = f'genosis://{publisher_id}/{source_dirname}'
        self.images_paths = []
        self._is_open = True

    def setup(self):
        if 'http' in self.input_source:
            ret = requests.get(self.input_source)

            rg = re.compile(r'href="(.*)"')
            files_names = rg.findall(ret.text)
            self.images_paths = sorted([os.path.join(self.input_source, f) for f in files_names], key=lambda s: int(s.split('frame_')[1].split('.png')[0]))
        else:
            self.images_paths = sorted(glob.glob(os.path.join(self.input_source, '*.png')), key=lambda s: int(os.path.basename(s).split('frame_')[1].split('.png')[0]))
            self.images_paths = self.images_paths[TMP_IGNORE_N_FRAMES:]
        self.expected_total_frames = len(self.images_paths)

    def read_next_frame(self):
        # print('reading next frame')
        if len(self.images_paths) != 0:
            next_frame_index = self.current_frame_index + 1
            if next_frame_index < self.expected_total_frames:
                image_path = self.images_paths[next_frame_index]
                if 'http' in image_path:
                    image_path = self.dl_temp_image(image_path)
                frame = cv2.imread(image_path)
                self.current_frame_index = next_frame_index
                # print(f'read next frame: {self.current_frame_index}')
                return frame
            else:
                self._is_open = False

    def dl_temp_image(self, image_path_url):
        response = requests.get(image_path_url, stream=True)
        # image_name = os.path.basename(image_path)
        # image_path = os.path.join(TEMP_IMG_PATH, image_name)
        if os.path.exists(TEMP_IMG_PATH):
            os.remove(TEMP_IMG_PATH)
        with open(TEMP_IMG_PATH, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        return TEMP_IMG_PATH

    def is_open(self):
        return self._is_open
    def close(self):
        self._is_open = False