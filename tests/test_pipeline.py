from unittest.mock import patch, MagicMock
from unittest import TestCase

import numpy as np

import torch

from torchvision import transforms

from adaptive_publisher.models.pipeline import ModelPipeline

from adaptive_publisher.models.transforms.transf_ocv import (
    get_transforms_ocv
)
from adaptive_publisher.models.transforms.transf_torch import get_transforms_torch
from adaptive_publisher.conf import EXAMPLE_IMAGES_PATH



class TestModelPipelineWithMockedModels(TestCase):

    def setUp(self):
        self.thresholds = {
            'diff': 0.05,
            'oi_cls': (0.3, 0.7),
            'oi_obj': 0.5
        }

        with patch('adaptive_publisher.models.pipeline.ModelPipeline.setup_models') as mocked_setup:
            with patch('adaptive_publisher.models.pipeline.ModelPipeline.setup_transforms') as mocked_transforms:
                self.pipeline = ModelPipeline(thresholds=self.thresholds)
                self.pipeline.pixel_diff = MagicMock()
                self.pipeline.oi_cls = MagicMock()
                self.pipeline.oi_obj = MagicMock()

                self.pipeline.obj_res_transf = lambda x: x
                self.pipeline.cls_res_transf = lambda x: x

    def tearDown(self):
        pass

    def mocked_img(self, height, width):
        bgr_color=(0, 0, 0)
        image = np.zeros((height, width, 3), np.uint8)
        # Fill image with color
        image[:] = bgr_color
        return image

    def test_pipeline_aways_ignore_diff_predict_and_set_frame_as_key_on_first_frame(self):
        mocked_img = self.mocked_img(1080, 1920)

        self.pipeline.pixel_diff.predict.return_value = 0.5
        self.pipeline.oi_cls.predict.return_value = 0.1
        self.pipeline.oi_obj.predict.return_value = 0.5
        self.pipeline.predict(mocked_img)

        self.assertFalse(self.pipeline.pixel_diff.predict.called)
        np.testing.assert_array_equal(self.pipeline.last_key_frame, mocked_img)

    def test_pipeline_should_run_diff_predict_if_not_first_frame(self):
        mocked_img1 = self.mocked_img(1080, 1920)
        self.pipeline.last_key_frame = mocked_img1
        self.pipeline.last_key_frame_prediction = False

        mocked_img2 = self.mocked_img(1080, 1920)

        self.pipeline.pixel_diff.predict.return_value = 0.06
        self.pipeline.oi_cls.predict.return_value = 0.1
        self.pipeline.oi_obj.predict.return_value = 0.5

        self.pipeline.predict(mocked_img2)
        self.assertTrue(self.pipeline.pixel_diff.predict.called)

    def test_pipeline_should_run_and_stop_diff_if_less_then_threshold(self):
        mocked_img1 = self.mocked_img(1080, 1920)
        self.pipeline.last_key_frame = mocked_img1
        self.pipeline.last_key_frame_prediction = False

        mocked_img2 = self.mocked_img(1080, 1920)

        self.pipeline.pixel_diff.predict.return_value = 0.049

        self.pipeline.predict(mocked_img2)
        self.assertTrue(self.pipeline.pixel_diff.predict.called)
        self.assertFalse(self.pipeline.oi_cls.predict.called)
        self.assertFalse(self.pipeline.oi_obj.predict.called)

    def test_pipeline_should_run_and_stop_on_cls_predict_if_cls_conf_low_and_diff_more_then_threshold(self):
        mocked_img1 = self.mocked_img(1080, 1920)
        self.pipeline.last_key_frame = mocked_img1
        self.pipeline.last_key_frame_prediction = False

        mocked_img2 = self.mocked_img(1080, 1920)

        self.pipeline.pixel_diff.predict.return_value = 0.06
        self.pipeline.oi_cls.predict.return_value = 0.1

        self.pipeline.predict(mocked_img2)
        self.assertTrue(self.pipeline.pixel_diff.predict.called)
        self.assertTrue(self.pipeline.oi_cls.predict.called)
        self.assertFalse(self.pipeline.oi_obj.predict.called)

    def test_pipeline_should_run_and_stop_on_cls_predict_if_cls_conf_high_and_diff_more_then_threshold(self):
        mocked_img1 = self.mocked_img(1080, 1920)
        self.pipeline.last_key_frame = mocked_img1
        self.pipeline.last_key_frame_prediction = False

        mocked_img2 = self.mocked_img(1080, 1920)

        self.pipeline.pixel_diff.predict.return_value = 0.06
        self.pipeline.oi_cls.predict.return_value = 0.71

        self.pipeline.predict(mocked_img2)
        self.assertTrue(self.pipeline.pixel_diff.predict.called)
        self.assertTrue(self.pipeline.oi_cls.predict.called)
        self.assertFalse(self.pipeline.oi_obj.predict.called)

    def test_pipeline_should_run_and_stop_on_obj_predict_if_cls_conf_medium_and_diff_more_then_threshold(self):
        mocked_img1 = self.mocked_img(1080, 1920)
        self.pipeline.last_key_frame = mocked_img1
        self.pipeline.last_key_frame_prediction = False

        mocked_img2 = self.mocked_img(1080, 1920)

        self.pipeline.pixel_diff.predict.return_value = 0.06
        self.pipeline.oi_cls.predict.return_value = 0.69
        self.pipeline.oi_obj.predict.return_value = 0.49

        self.pipeline.predict(mocked_img2)
        self.assertTrue(self.pipeline.pixel_diff.predict.called)
        self.assertTrue(self.pipeline.oi_cls.predict.called)
        self.assertTrue(self.pipeline.oi_obj.predict.called)

    def test_pipeline_should_return_true_if_cls_conf_high(self):
        mocked_img = self.mocked_img(1080, 1920)
        self.pipeline.oi_cls.predict.return_value = 0.71

        ret = self.pipeline.predict(mocked_img)
        self.assertTrue(ret)

    def test_pipeline_should_return_true_if_obj_conf_more_than_threshold(self):
        mocked_img = self.mocked_img(1080, 1920)
        self.pipeline.oi_cls.predict.return_value = 0.5
        self.pipeline.oi_obj.predict.return_value = 0.51

        ret = self.pipeline.predict(mocked_img)
        self.assertTrue(ret)

    def test_pipeline_should_return_false_if_cls_conf_low(self):
        mocked_img = self.mocked_img(1080, 1920)
        self.pipeline.oi_cls.predict.return_value = 0.29

        ret = self.pipeline.predict(mocked_img)
        self.assertFalse(ret)

    def test_pipeline_should_ret_false_as_last_predict_if_diff_less_than_threshold(self):
        mocked_img1 = self.mocked_img(1080, 1920)
        self.pipeline.last_key_frame = mocked_img1
        mocked_img2 = self.mocked_img(1080, 1920)
        self.pipeline.pixel_diff.predict.return_value = 0.04

        self.pipeline.last_key_frame_prediction = False
        ret = self.pipeline.predict(mocked_img2)
        self.assertFalse(ret)

    def test_pipeline_should_ret_true_as_last_predict_if_diff_less_than_threshold(self):
        mocked_img1 = self.mocked_img(1080, 1920)
        self.pipeline.last_key_frame = mocked_img1

        mocked_img2 = self.mocked_img(1080, 1920)
        self.pipeline.pixel_diff.predict.return_value = 0.04

        self.pipeline.last_key_frame_prediction = True
        ret = self.pipeline.predict(mocked_img2)
        self.assertTrue(ret)


class TestModelPipelineActualModels(TestCase):

    def setUp(self):
        self.thresholds = {
            'diff': 0.05,
            'oi_cls': (0.3, 0.7),
            'oi_obj': 0.5
        }
        self.pipeline = ModelPipeline(thresholds=self.thresholds)

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


    def test_image_obj_model_with_ocv_transform(self):
        mocked_img1 = self.mocked_img(1080, 1920)

        transf_image_ocv = get_transforms_ocv('OBJ')(mocked_img1)
        ret = self.pipeline.oi_obj.predict(transf_image_ocv)
        # self.assertAlmostEqual(0.817254, ret, places=6)

    def test_image_obj_model_with_ocv_transform_with_real_image(self):
        import cv2
        import os
        img_bgr = cv2.imread(os.path.join(EXAMPLE_IMAGES_PATH, 'dog_bike_car.jpg'))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


        transf_image_ocv = get_transforms_ocv('OBJ')(img_rgb)
        ret = self.pipeline.oi_obj.predict(transf_image_ocv)
        # self.assertAlmostEqual(0.817254, ret, places=6)


    # def test_image_obj_model_with_torch_transform(self):
    #     mocked_img1 = self.mocked_img(1080, 1920)

    #     pil_img = transforms.ToPILImage()(mocked_img1)
    #     transf_image_torch = get_transforms_torch('OBJ')(pil_img)

    #     ret = self.pipeline.oi_obj.predict(transf_image_torch)
    #     self.assertAlmostEqual(0.817254, ret, places=6)