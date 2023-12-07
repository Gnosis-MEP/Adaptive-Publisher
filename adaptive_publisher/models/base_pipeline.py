import time

import cv2


class BaseModelPipeline():
    def __init__(self, target_fps, thresholds, oi_label_list=None):
        self.oi_label_list = oi_label_list
        self.target_fps = target_fps
        self.target_processing_time = 1 / target_fps
        self.thresholds = thresholds
        # self.thresholds = {
        #     'diff': 0.05,
        #     'oi_cls': (0.3, 0.7),
        #     'oi_obj': 0.5
        # }

        self.setup_models()
        self.setup_transforms()
        self.processing_times = {
            'predict': [],
        }

    def register_func_time(self, name, func, *args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        self.processing_times[name].append(run_time)
        return value

    def update_oi_ids(self, oi_label_list):
        self.oi_label_list = oi_label_list

    def setup_models(self):
        pass

    def setup_transforms(self):
        pass

    def global_transform(self, new_image_frame):
        return cv2.cvtColor(new_image_frame, cv2.COLOR_BGR2RGB)

    def run_pipeline_or_cached(self, new_image_frame):
        raise NotImplemented()

    def predict(self, new_image_frame):
        return self.register_func_time('predict', self.run_pipeline_or_cached, new_image_frame)



class MockedPipeline(BaseModelPipeline):

    def run_pipeline_or_cached(self, new_image_frame):
        return True