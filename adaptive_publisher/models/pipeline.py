import math
import time

import cv2
import torch

from adaptive_publisher.models.transforms.transf_ocv import get_transforms

from adaptive_publisher.models.pixel_diff import CVDiffModel
from adaptive_publisher.models.oi_cls import OIClsModel
from adaptive_publisher.models.oi_obj import OIObjModel




class ModelPipeline():
    def __init__(self, target_fps, thresholds, oi_label_list=None):
        self.target_fps = target_fps
        self.target_processing_time = 1 / target_fps
        self.cached_result_count = 0
        self.last_key_frame = None
        self.last_key_frame_prediction = None
        self.thresholds = thresholds
        # self.thresholds = {
        #     'diff': 0.05,
        #     'oi_cls': (0.3, 0.7),
        #     'oi_obj': 0.5
        # }

        self.oi_label_list = oi_label_list
        self.setup_models()
        self.setup_transforms()
        self.processing_times = {
            'diff': [],
            'oi_cls_transf': [],
            'oi_cls': [],
            'oi_obj_transf': [],
            'oi_obj': [],
            'pipeline': [],
            'predict': [],
        }

    def register_func_time(self, name, func, *args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        self.processing_times[name].append(run_time)
        return value

    def setup_models(self):
        self.pixel_diff = CVDiffModel()
        self.oi_cls = OIClsModel()
        self.oi_obj = OIObjModel(oi_label_list=self.oi_label_list)

    def setup_transforms(self):
        self.obj_res_transf = get_transforms('OBJ')
        self.cls_res_transf = get_transforms('CLS')

    def oi_transforms(self, new_image_frame):
        obj_img = self.register_func_time('oi_obj_transf', self.obj_res_transf, new_image_frame)
        cls_img = self.register_func_time('oi_cls_transf', self.cls_res_transf, obj_img)
        return obj_img, cls_img

    def replace_key_frame(self, new_image_frame, prediction):
        del self.last_key_frame
        self.last_key_frame = new_image_frame
        self.last_key_frame_prediction = prediction

    def global_transform(self, new_image_frame):
        return cv2.cvtColor(new_image_frame, cv2.COLOR_BGR2RGB)

    def calculate_cached_result_count(self):
        total_cached_results_frames = 0
        try:
            last_processing_time = self.processing_times['pipeline'][-1]
        except IndexError:
            pass
        else:
            processing_time_left = self.target_processing_time - last_processing_time
            if processing_time_left < 0:
                missing_time = processing_time_left * -1
                total_cached_results_frames = math.ceil(missing_time / self.target_processing_time)
        return total_cached_results_frames

    def run_pipeline(self, new_image_frame):
        has_oi = False
        with torch.no_grad():
            diff_perc = 1.0
            new_image_frame = self.global_transform(new_image_frame)
            if self.last_key_frame is not None:
                diff_perc = self.register_func_time('diff', self.pixel_diff.predict, new_image_frame, last_key_frame=self.last_key_frame)
            if diff_perc > self.thresholds['diff']:
                oi_obj_img, oi_cls_img  = self.oi_transforms(new_image_frame)
                oi_cls_conf = self.register_func_time('oi_cls', self.oi_cls.predict, oi_cls_img)
                if oi_cls_conf < self.thresholds['oi_cls'][0]:
                    has_oi = False
                elif oi_cls_conf > self.thresholds['oi_cls'][1]:
                    has_oi = True
                else:
                    has_oi = self.register_func_time('oi_obj', self.oi_obj.predict, oi_obj_img, threshold=self.thresholds['oi_obj'])

                self.replace_key_frame(new_image_frame, has_oi)
            else:
                has_oi = self.last_key_frame_prediction
            return has_oi

    def run_pipeline_or_cached(self, new_image_frame):
        if self.cached_result_count > 0:
            self.cached_result_count -= 1
            return self.last_key_frame_prediction

        ret = self.register_func_time('pipeline', self.run_pipeline, new_image_frame)
        self.cached_result_count = self.calculate_cached_result_count()
        return ret

    def predict(self, new_image_frame):
        return self.register_func_time('predict', self.run_pipeline_or_cached, new_image_frame)

