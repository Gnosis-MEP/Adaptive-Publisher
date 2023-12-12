import os
import math
import time

import cv2
import torch


from adaptive_publisher.conf import USE_OCV_TRANSFORMS, EXAMPLE_IMAGES_PATH, DEFAULT_CACHED_FRAME_RATE_MULTIPLIER
if USE_OCV_TRANSFORMS:
    from adaptive_publisher.models.transforms.transf_ocv import get_transforms
else:
    from adaptive_publisher.models.transforms.transf_torch import get_transforms

from adaptive_publisher.models.pixel_diff import CVDiffModel
from adaptive_publisher.models.oi_cls import OIClsModel
from adaptive_publisher.models.oi_obj import OIObjModel
from adaptive_publisher.models.base_pipeline import BaseModelPipeline



class ModelPipeline(BaseModelPipeline):
    def __init__(self, target_fps, thresholds, oi_label_list=None):
        self.cached_result_count = 0
        self.last_key_frame = None
        self.last_key_frame_prediction = None

        super().__init__(target_fps, thresholds, oi_label_list=oi_label_list)
        # self.thresholds = {
        #     'diff': 0.05,
        #     'oi_cls': (0.3, 0.7),
        #     'oi_obj': 0.5
        # }
        self.processing_times.update({
            'diff': [],
            'oi_cls_transf': [],
            'oi_cls': [],
            'oi_obj_transf': [],
            'oi_obj': [],
            'pipeline': [],
            'predict': [],
        })

    def setup_models(self):
        self.pixel_diff = CVDiffModel()
        self.oi_cls = OIClsModel()
        self.oi_obj = OIObjModel(oi_label_list=self.oi_label_list)

    def setup_transforms(self):
        self.obj_res_transf = get_transforms('OBJ')
        self.cls_res_transf = get_transforms('CLS')

    def update_oi_ids(self, oi_label_list):
        super().update_oi_ids(oi_label_list)
        self.oi_cls.update_oi_ids(oi_label_list)

    def oi_transforms(self, new_image_frame):
        obj_img = self.register_func_time('oi_obj_transf', self.obj_res_transf, new_image_frame)
        cls_img = self.register_func_time('oi_cls_transf', self.cls_res_transf, obj_img)
        return obj_img, cls_img

    def replace_key_frame(self, new_image_frame, prediction):
        # if self.last_key_frame is not None:
        #     cv2.imwrite(os.path.join(EXAMPLE_IMAGES_PATH,'last_kf.jpg'), self.last_key_frame)
        del self.last_key_frame
        self.last_key_frame = new_image_frame
        self.last_key_frame_prediction = prediction
        # cv2.imwrite(os.path.join(EXAMPLE_IMAGES_PATH,'new_kf.jpg'), self.last_key_frame)


    def calculate_cached_result_count(self):
        total_cached_results_frames = 0
        try:
            last_processing_time = self.processing_times['pipeline'][-1]
        except IndexError:
            pass
        else:
            # processing_time_ratio = last_processing_time / self.target_processing_time
            processing_time_left = self.target_processing_time - last_processing_time
            if processing_time_left < 0:
            # if processing_time_ratio > 1:
                missing_time = processing_time_left * -1
                cached_float_frames = DEFAULT_CACHED_FRAME_RATE_MULTIPLIER * missing_time / self.target_processing_time
                total_cached_results_frames = math.ceil(cached_float_frames)
                # total_cached_results_frames = math.ceil(self.target_fps * ((processing_time_ratio) - 1))
                # print(f'bIG P TIME: {self.target_fps} {processing_time_ratio} {total_cached_results_frames}')
        return total_cached_results_frames

    def run_pipeline(self, new_image_frame):
        has_oi = False
        with torch.no_grad():
            diff_perc = 1.0
            origin = new_image_frame
            if self.last_key_frame is not None:
                diff_perc = self.register_func_time('diff', self.pixel_diff.predict, origin, last_key_frame=self.last_key_frame)
            if diff_perc > self.thresholds['diff']:
                new_image_frame = self.global_transform(new_image_frame)
                oi_obj_img, oi_cls_img  = self.oi_transforms(new_image_frame)
                oi_cls_conf = self.register_func_time('oi_cls', self.oi_cls.predict, oi_cls_img)
                if oi_cls_conf < self.thresholds['oi_cls'][0]:
                    has_oi = False
                elif oi_cls_conf > self.thresholds['oi_cls'][1]:
                    has_oi = True
                else:
                    has_oi = self.register_func_time('oi_obj', self.oi_obj.predict, oi_obj_img, threshold=self.thresholds['oi_obj'])

                print('= replaced KF')
                self.replace_key_frame(origin, has_oi)
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


class BLSingleModelPipeline(BaseModelPipeline):

    def __init__(self, target_fps, thresholds, model_name, oi_label_list=None):
        self.model_name = model_name
        self.model = None
        self.model_predict_funct = None
        self.model_transf_funct = None
        super().__init__(target_fps, thresholds, oi_label_list=oi_label_list)
        self.processing_times.update({
            'model_transf': [],
            'model': [],
        })

    def setup_models(self):
        if self.model_name == 'CVDiffModel':
            self.model = CVDiffModel()
            self.model_predict_funct = lambda x: self.model.predict(x) > self.thresholds['diff']
        elif self.model_name == 'OIClsModel':
            self.model = OIClsModel()
            self.model_predict_funct = lambda x: self.model.predict(x) > self.thresholds['oi_cls'][0]
        elif self.model_name == 'OIObjModel':
            self.model = OIObjModel(oi_label_list=self.oi_label_list)
            self.model_predict_funct = self.model.predict

    def setup_transforms(self):
        self.obj_res_transf = get_transforms('OBJ')
        self.cls_res_transf = get_transforms('CLS')

        if self.model_name == 'CVDiffModel':
            self.model_transf_funct = self.global_transform

        elif self.model_name == 'OIClsModel':
            self.cls_transf = get_transforms('CLS')

            if USE_OCV_TRANSFORMS:
                self.model_transf_funct = lambda new_image_frame: self.cls_transf(self.global_transform(new_image_frame))
            else:
                from torchvision import transforms
                self.model_transf_funct = lambda new_image_frame: self.cls_transf(transforms.ToPILImage()(self.global_transform(new_image_frame)))

        elif self.model_name == 'OIObjModel':
            self.obj_transf = get_transforms('OBJ')
            self.model_transf_funct = lambda new_image_frame: self.obj_transf(self.global_transform(new_image_frame))

    def run_pipeline_or_cached(self, new_image_frame):
        new_image_frame = self.register_func_time('model_transf', self.model_transf_funct, new_image_frame)
        return self.register_func_time('model', self.model_predict_funct, new_image_frame)

    def update_oi_ids(self, oi_label_list):
        super().update_oi_ids(oi_label_list)
        if self.model_name == 'OIObjModel':
            self.model.update_oi_ids(oi_label_list)