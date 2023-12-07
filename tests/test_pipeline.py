import os
from unittest.mock import patch, MagicMock
from unittest import TestCase

import numpy as np
import cv2
from torchvision import transforms


from adaptive_publisher.models.pipeline import ModelPipeline, OIObjModel
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
                self.pipeline = ModelPipeline(target_fps=60, thresholds=self.thresholds)
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

    def test_pipeline_should_ret_true_as_last_predict_if_diff_less_than_threshold(self):
        mocked_img1 = self.mocked_img(1080, 1920)
        self.pipeline.last_key_frame = mocked_img1

        mocked_img2 = self.mocked_img(1080, 1920)
        self.pipeline.pixel_diff.predict.return_value = 0.04

        self.pipeline.last_key_frame_prediction = True
        ret = self.pipeline.predict(mocked_img2)
        self.assertTrue(ret)



class TestModelPipelineWithMockedModelsWithCachedResults(TestCase):

    def setUp(self):
        self.thresholds = {
            'diff': 0.05,
            'oi_cls': (0.3, 0.7),
            'oi_obj': 0.5
        }
        self.target_fps = 30
        self.target_time = 1 / self.target_fps

        with patch('adaptive_publisher.models.pipeline.ModelPipeline.setup_models') as mocked_setup:
            with patch('adaptive_publisher.models.pipeline.ModelPipeline.setup_transforms') as mocked_transforms:
                self.pipeline = ModelPipeline(target_fps=self.target_fps, thresholds=self.thresholds)
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

    def test_calculate_cached_result_count_should_return_zero_if_no_prev_execution(self):
        ret = self.pipeline.calculate_cached_result_count()
        self.assertEqual(0, ret)

    def test_calculate_cached_result_count_should_return_zero_if_prev_time_less_then_target(self):
        self.pipeline.processing_times['pipeline'] = [self.target_time - 0.001]
        ret = self.pipeline.calculate_cached_result_count()
        self.assertEqual(0, ret)

    def test_calculate_cached_result_count_should_return_1_if_prev_time_1_frame_more_then_target(self):
        self.pipeline.processing_times['pipeline'] = [self.target_time + 0.001]
        ret = self.pipeline.calculate_cached_result_count()
        self.assertEqual(1, ret)

    def test_calculate_cached_result_count_should_round_frames_up_if_prev_time_is_more_then_target(self):
        self.pipeline.processing_times['pipeline'] = [(self.target_time * 2) + 0.001]
        ret = self.pipeline.calculate_cached_result_count()
        self.assertEqual(2, ret)

    @patch('adaptive_publisher.models.pipeline.ModelPipeline.calculate_cached_result_count')
    def test_cached_result_count_should_be_correctly_set_to_calculated_res_after_first_prediction(self, mocked_cached):
        mocked_img1 = self.mocked_img(1080, 1920)

        self.pipeline.pixel_diff.predict.return_value = 0.06
        self.pipeline.oi_cls.predict.return_value = 0.69
        self.pipeline.oi_obj.predict.return_value = 0.49

        mocked_cached.return_value = 5
        self.pipeline.predict(mocked_img1)
        mocked_cached.assert_called()
        self.assertEqual(5, self.pipeline.cached_result_count)

    @patch('adaptive_publisher.models.pipeline.ModelPipeline.run_pipeline')
    @patch('adaptive_publisher.models.pipeline.ModelPipeline.calculate_cached_result_count')
    def test_cached_result_count_should_be_correctly_reduced_after_cached_result_until_zero(
                                                                                self, mocked_cached, m_run_pipe):
        mocked_img1 = self.mocked_img(1080, 1920)
        self.pipeline.cached_result_count = 3

        self.pipeline.predict(mocked_img1)
        self.assertEqual(self.pipeline.cached_result_count, 2)
        self.pipeline.predict(mocked_img1)
        self.assertEqual(self.pipeline.cached_result_count, 1)
        self.pipeline.predict(mocked_img1)
        self.assertEqual(self.pipeline.cached_result_count, 0)

    @patch('adaptive_publisher.models.pipeline.ModelPipeline.run_pipeline')
    @patch('adaptive_publisher.models.pipeline.ModelPipeline.calculate_cached_result_count')
    def test_prediction_should_not_call_calc_cached_res_nor_run_pipeline_if_cached_result_count_more_then_zero(
                                                                                self, mocked_cached, m_run_pipe):
        mocked_img1 = self.mocked_img(1080, 1920)
        self.pipeline.cached_result_count = 1

        self.pipeline.predict(mocked_img1)
        mocked_cached.assert_not_called()
        m_run_pipe.assert_not_called()


    @patch('adaptive_publisher.models.pipeline.ModelPipeline.run_pipeline')
    @patch('adaptive_publisher.models.pipeline.ModelPipeline.calculate_cached_result_count')
    def test_prediction_should_return_last_key_frame_prediction_if_cached_result_count_more_then_zero(
                                                                                self, mocked_cached, m_run_pipe):
        mocked_img1 = self.mocked_img(1080, 1920)
        self.pipeline.last_key_frame_prediction = True
        self.pipeline.cached_result_count = 2

        self.assertTrue(self.pipeline.predict(mocked_img1))

        self.pipeline.last_key_frame_prediction = False
        self.assertFalse(self.pipeline.predict(mocked_img1))

