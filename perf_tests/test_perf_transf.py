import statistics
import time
from unittest import TestCase

import numpy as np

from adaptive_publisher.models.transforms.transf_ocv import (
    get_transforms_ocv
)
from adaptive_publisher.models.transforms.transf_torch import get_transforms_torch


from torchvision import transforms
import numpy as np



class TestPerfTransfModels(TestCase):

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

    # def test_perf_cv2_consecutive_res(self):
    #     import json
    #     times = {
    #         'total': [],
    #         'obj': [],
    #         'cls': []
    #     }
    #     # shape = (640, 1137)
    #     obj_transf = get_transforms_ocv('OBJ')
    #     cls_transf = get_transforms_ocv('CLS')
    #     for i in range(1000):
    #         image = self.mocked_img(1080, 1920, blank=False)
    #         start_time = time.perf_counter()

    #         obj_img = obj_transf(image)
    #         obj_end_time = time.perf_counter()

    #         cls_start_time = time.perf_counter()
    #         cls_img = cls_transf(obj_img)
    #         cls_end_time = time.perf_counter()

    #         end_time = time.perf_counter()

    #         obj_run_time = obj_end_time - start_time
    #         cls_run_time = cls_end_time - cls_start_time
    #         run_time = obj_run_time + cls_run_time
    #         times['total'].append(run_time)
    #         times['obj'].append(obj_run_time)
    #         times['cls'].append(cls_run_time)

    #     latencies = {
    #         f'{k}_avg': statistics.mean(v) for k, v in times.items()
    #     }

    #     latencies.update({
    #         f'{k}_std': statistics.stdev(v) for k, v in times.items()
    #     })
    #     latencies.update({
    #         f'{k}_fps': 1/statistics.mean(v)for k, v in times.items()
    #     })
    #     print(f'cv2 (consecutive_res): {json.dumps(latencies, indent=4)}')

    # def test_perf_cv2_non_consecutive_res(self):
    #     import json
    #     times = {
    #         'total': [],
    #         'obj': [],
    #         'cls': []
    #     }
    #     # shape = (640, 1137)
    #     obj_transf = get_transforms_ocv('OBJ')
    #     cls_transf = get_transforms_ocv('CLS')
    #     for i in range(1000):
    #         image = self.mocked_img(1080, 1920, blank=False)
    #         start_time = time.perf_counter()

    #         obj_img = obj_transf(image)
    #         obj_end_time = time.perf_counter()

    #         cls_start_time = time.perf_counter()
    #         cls_img = cls_transf(image)
    #         cls_end_time = time.perf_counter()

    #         end_time = time.perf_counter()

    #         obj_run_time = obj_end_time - start_time
    #         cls_run_time = cls_end_time - cls_start_time
    #         run_time = obj_run_time + cls_run_time
    #         times['total'].append(run_time)
    #         times['obj'].append(obj_run_time)
    #         times['cls'].append(cls_run_time)

    #     latencies = {
    #         f'{k}_avg': statistics.mean(v) for k, v in times.items()
    #     }

    #     latencies.update({
    #         f'{k}_std': statistics.stdev(v) for k, v in times.items()
    #     })
    #     latencies.update({
    #         f'{k}_fps': 1/statistics.mean(v)for k, v in times.items()
    #     })
    #     print(f'cv2 (non_consecutive res): {json.dumps(latencies, indent=4)}')

    # def test_perf_torch_consecutive_res(self):
    #     import json
    #     times = {
    #         'total': [],
    #         'obj': [],
    #         'cls': []
    #     }
    #     # shape = (640, 1137)
    #     obj_transf = get_transforms_torch('OBJ')
    #     cls_transf = get_transforms_torch('CLS')
    #     for i in range(1000):
    #         image = self.mocked_img(1080, 1920, blank=False)
    #         start_time = time.perf_counter()

    #         obj_img = obj_transf(image)
    #         obj_end_time = time.perf_counter()

    #         cls_start_time = time.perf_counter()
    #         cls_img = cls_transf(obj_img)
    #         cls_end_time = time.perf_counter()

    #         end_time = time.perf_counter()

    #         obj_run_time = obj_end_time - start_time
    #         cls_run_time = cls_end_time - cls_start_time
    #         run_time = obj_run_time + cls_run_time
    #         times['total'].append(run_time)
    #         times['obj'].append(obj_run_time)
    #         times['cls'].append(cls_run_time)

    #     latencies = {
    #         f'{k}_avg': statistics.mean(v) for k, v in times.items()
    #     }

    #     latencies.update({
    #         f'{k}_std': statistics.stdev(v) for k, v in times.items()
    #     })
    #     latencies.update({
    #         f'{k}_fps': 1/statistics.mean(v)for k, v in times.items()
    #     })
    #     print(f'torch (consecutive res): {json.dumps(latencies, indent=4)}')

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

            cls_start_time = time.perf_counter()
            # requires convert to PIL
            pil_img = transforms.ToPILImage()(image)
            cls_img = cls_transf(pil_img)
            cls_end_time = time.perf_counter()

            obj_start_time = time.perf_counter()

            obj_img = obj_transf(image)
            obj_end_time = time.perf_counter()



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

    def test_perf_gpu_torch_consecutive_res(self):
        import torch
        import json
        if torch.cuda.device_count() == 0:
            'dont run this test if there is no cuda'
            return
        gpu_device = torch.device('cuda')
        times = {
            'total': [],
            'obj': [],
            'cls': []
        }
        # shape = (640, 1137)
        obj_transf = get_transforms_torch('OBJ')
        cls_transf = get_transforms_torch('CLS')
        to_tensor = transforms.ToTensor()
        for i in range(1000):
            image = self.mocked_img(1080, 1920, blank=False)
            tensor_image = to_tensor(image)
            tensor_image.to(gpu_device)
            start_time = time.perf_counter()

            obj_img = obj_transf(tensor_image)
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
        print(f'GPU: torch (consecutive res): {json.dumps(latencies, indent=4)}')