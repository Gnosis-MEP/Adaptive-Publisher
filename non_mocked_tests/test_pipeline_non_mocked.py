import os
from unittest import TestCase

import numpy as np
import cv2
from torchvision import transforms


from adaptive_publisher.models.pipelines import ModelPipeline, OIObjModel
from adaptive_publisher.models.transforms.transf_ocv import (
    get_transforms_ocv
)
from adaptive_publisher.models.transforms.transf_torch import get_transforms_torch
from adaptive_publisher.conf import EXAMPLE_IMAGES_PATH


class TestModelPipelineActualModels(TestCase):

    def setUp(self):
        self.thresholds = {
            'diff': 0.05,
            'oi_cls': (0.3, 0.8),
            'oi_obj': 0.5
        }
        self.oi_label_list = ['car']
        self.pipeline = ModelPipeline(target_fps=60, thresholds=self.thresholds, oi_label_list=self.oi_label_list)

    def tearDown(self):
        pass

    def mocked_img(self, height, width, blank=True):
        if blank:
            bgr_color=(0, 0, 0)
            image = np.zeros((height, width, 3), np.uint8)
            # Fill image with color
            image[:] = bgr_color
        else:
            image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        return image

    def test_image_cls_model_with_ocv_transform(self):
        mocked_img1 = self.mocked_img(1080, 1920)

        transf_image_ocv = get_transforms_ocv('CLS')(mocked_img1)

        ret = self.pipeline.oi_cls.predict(transf_image_ocv)
        self.assertAlmostEqual(0.817254, ret, places=6)

    def test_image_cls_model_with_torch_transform(self):
        mocked_img1 = self.mocked_img(1080, 1920)

        pil_img = transforms.ToPILImage()(mocked_img1)
        transf_image_torch = get_transforms_torch('CLS')(pil_img)

        ret = self.pipeline.oi_cls.predict(transf_image_torch)
        self.assertAlmostEqual(0.817254, ret, places=6)

    def test_image_obj_model_with_ocv_transform_with_real_image(self):
        img_bgr = cv2.imread(os.path.join(EXAMPLE_IMAGES_PATH, 'dog_bike_car.jpg'))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        transf_image_ocv = get_transforms_ocv('OBJ')(img_rgb)

        oi_obj = OIObjModel(oi_label_list=['dog', 'car'])

        ret = oi_obj.predict(transf_image_ocv, threshold=0.5)
        self.assertTrue(ret)

    def test_image_obj_model_with_ocv_transform_with_real_image_not_positive(self):
        img_bgr = cv2.imread(os.path.join(EXAMPLE_IMAGES_PATH, 'dog_bike_car.jpg'))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        transf_image_ocv = get_transforms_ocv('OBJ')(img_rgb)

        oi_obj = OIObjModel(oi_label_list=['dog', 'bicycle'])

        ret = oi_obj.predict(transf_image_ocv, threshold=0.5)
        self.assertFalse(ret)

    def test_full_pipeline_with_real_image(self):
        img_bgr = cv2.imread(os.path.join(EXAMPLE_IMAGES_PATH, 'dog_bike_car.jpg'))
        self.pipeline.last_key_frame = self.mocked_img(576, 768)
        ret = self.pipeline.predict(img_bgr)
        self.assertTrue(ret)

    def test_showimg_pipeline_with_real_image(self):
        img_bgr = cv2.imread(os.path.join(EXAMPLE_IMAGES_PATH, 'dog_bike_car.jpg'))
        import matplotlib
        from matplotlib import pyplot as plt
        plt.imshow(img_bgr)
        plt.show()
        import ipdb; ipdb.set_trace()
        pass