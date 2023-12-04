import statistics
import time
from unittest.mock import patch, MagicMock
from unittest import TestCase

import numpy as np

from adaptive_publisher.models.pipeline import ModelPipeline
from adaptive_publisher.models.transforms.transf_ocv import (
    get_transforms_ocv,
    to_pytorch_tensor_format,
    to_tensor,
    normalize_image,
    crop_center,
)
from adaptive_publisher.models.transforms.transf_torch import get_transforms_torch


from torchvision import transforms
import numpy as np


class TestModelPipelineActualModels(TestCase):

    def setUp(self):
        pass
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

    def test_ocv_to_tensor(self):

        opencv_image = self.mocked_img(1080, 1920, blank=False)
        pil_img = transforms.ToPILImage()(opencv_image)
        np.testing.assert_array_equal(transforms.ToTensor()(pil_img), to_pytorch_tensor_format(to_tensor(opencv_image)))

    def test_ocv_normalise_to_tensor(self):
        opencv_image = self.mocked_img(1080, 1920, blank=False)
        pil_img = transforms.ToPILImage()(opencv_image)

        ttest = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        np.testing.assert_array_almost_equal(ttest(pil_img), to_pytorch_tensor_format(normalize_image(to_tensor(opencv_image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))

    def test_ocv_crop(self):
        opencv_image = self.mocked_img(1080, 1920, blank=False)
        pil_img = transforms.ToPILImage()(opencv_image)
        np.testing.assert_array_equal(
            transforms.CenterCrop(224)(pil_img),
            crop_center(opencv_image, 224))

    def test_ocv_crop_tensor_normalize(self):
        opencv_image = self.mocked_img(1080, 1920, blank=False)
        pil_img = transforms.ToPILImage()(opencv_image)
        ttest = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        ocv_res = to_pytorch_tensor_format(
            normalize_image(
                to_tensor(
                    crop_center(opencv_image, 224)
                ),
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        )
        np.testing.assert_array_almost_equal(
            ttest(pil_img),
            ocv_res,
            decimal=6
        )

    def test_perf_cv2_consecutive_res(self):
        import json
        times = {
            'total': [],
            'obj': [],
            'cls': []
        }
        # shape = (640, 1137)
        obj_transf = get_transforms_ocv('OBJ')
        cls_transf = get_transforms_ocv('CLS')
        for i in range(1000):
            image = self.mocked_img(1080, 1920, blank=False)
            start_time = time.perf_counter()

            obj_img = obj_transf(image)
            obj_end_time = time.perf_counter()

            cls_start_time = time.perf_counter()
            cls_img = cls_transf(obj_img)
            cls_end_time = time.perf_counter()

            end_time = time.perf_counter()

            obj_run_time = obj_end_time - start_time
            cls_run_time = cls_end_time - cls_start_time
            run_time = obj_run_time + cls_run_time
            times['total'].append(run_time)
            times['obj'].append(obj_run_time)
            times['cls'].append(cls_run_time)

        latencies = {
            f'{k}_avg': statistics.mean(v) for k, v in times.items()
        }

        latencies.update({
            f'{k}_std': statistics.stdev(v) for k, v in times.items()
        })
        latencies.update({
            f'{k}_fps': 1/statistics.mean(v)for k, v in times.items()
        })
        print(f'cv2 (consecutive_res): {json.dumps(latencies, indent=4)}')

    def test_perf_cv2_non_consecutive_res(self):
        import json
        times = {
            'total': [],
            'obj': [],
            'cls': []
        }
        # shape = (640, 1137)
        obj_transf = get_transforms_ocv('OBJ')
        cls_transf = get_transforms_ocv('CLS')
        for i in range(1000):
            image = self.mocked_img(1080, 1920, blank=False)
            start_time = time.perf_counter()

            obj_img = obj_transf(image)
            obj_end_time = time.perf_counter()

            cls_start_time = time.perf_counter()
            cls_img = cls_transf(image)
            cls_end_time = time.perf_counter()

            end_time = time.perf_counter()

            obj_run_time = obj_end_time - start_time
            cls_run_time = cls_end_time - cls_start_time
            run_time = obj_run_time + cls_run_time
            times['total'].append(run_time)
            times['obj'].append(obj_run_time)
            times['cls'].append(cls_run_time)

        latencies = {
            f'{k}_avg': statistics.mean(v) for k, v in times.items()
        }

        latencies.update({
            f'{k}_std': statistics.stdev(v) for k, v in times.items()
        })
        latencies.update({
            f'{k}_fps': 1/statistics.mean(v)for k, v in times.items()
        })
        print(f'cv2 (non_consecutive res): {json.dumps(latencies, indent=4)}')

    def test_perf_torch_consecutive_res(self):
        import json
        times = {
            'total': [],
            'obj': [],
            'cls': []
        }
        # shape = (640, 1137)
        obj_transf = get_transforms_torch('OBJ')
        cls_transf = get_transforms_torch('CLS')
        for i in range(1000):
            image = self.mocked_img(1080, 1920, blank=False)
            start_time = time.perf_counter()

            obj_img = obj_transf(image)
            obj_end_time = time.perf_counter()

            cls_start_time = time.perf_counter()
            cls_img = cls_transf(obj_img)
            cls_end_time = time.perf_counter()

            end_time = time.perf_counter()

            obj_run_time = obj_end_time - start_time
            cls_run_time = cls_end_time - cls_start_time
            run_time = obj_run_time + cls_run_time
            times['total'].append(run_time)
            times['obj'].append(obj_run_time)
            times['cls'].append(cls_run_time)

        latencies = {
            f'{k}_avg': statistics.mean(v) for k, v in times.items()
        }

        latencies.update({
            f'{k}_std': statistics.stdev(v) for k, v in times.items()
        })
        latencies.update({
            f'{k}_fps': 1/statistics.mean(v)for k, v in times.items()
        })
        print(f'torch (consecutive res): {json.dumps(latencies, indent=4)}')

    def test_perf_torch_non_consecutive_res(self):
        import json
        times = {
            'total': [],
            'obj': [],
            'cls': []
        }
        # shape = (640, 1137)
        obj_transf = get_transforms_torch('OBJ')
        cls_transf = get_transforms_torch('CLS')
        for i in range(1000):
            image = self.mocked_img(1080, 1920, blank=False)
            obj_start_time = time.perf_counter()
            # requires convert to PIL

            obj_img = obj_transf(image)
            obj_end_time = time.perf_counter()

            cls_start_time = time.perf_counter()
            pil_img = transforms.ToPILImage()(image)
            cls_img = cls_transf(pil_img)
            cls_end_time = time.perf_counter()


            obj_run_time = obj_end_time - obj_start_time
            cls_run_time = cls_end_time - cls_start_time
            run_time = obj_run_time + cls_run_time
            times['total'].append(run_time)
            times['obj'].append(obj_run_time)
            times['cls'].append(cls_run_time)

        latencies = {
            f'{k}_avg': statistics.mean(v) for k, v in times.items()
        }

        latencies.update({
            f'{k}_std': statistics.stdev(v) for k, v in times.items()
        })
        latencies.update({
            f'{k}_fps': 1/statistics.mean(v)for k, v in times.items()
        })
        print(f'torch (non_consecutive res): {json.dumps(latencies, indent=4)}')

    # def test_resizes_pil(self):
    #     times = []
    #     shape = (640, 1137)
    #     for i in range(1000):
    #         image = self.mocked_img(1080, 1920, blank=False)
    #         image = transforms.ToPILImage()(image)
    #         tranf = transforms.Resize(shape[0], interpolation=InterpolationMode.BILINEAR)
    #         start_time = time.perf_counter()    # 1
    #         tranf(image)
    #         end_time = time.perf_counter()      # 2
    #         run_time = end_time - start_time    # 3
    #         times.append(run_time)

    #     avg_latency = statistics.mean(times)
    #     std_latency = statistics.stdev(times)
    #     print(f'torch(PIL): {avg_latency} ({std_latency}) / {1/avg_latency}')