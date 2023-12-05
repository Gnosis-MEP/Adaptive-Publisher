from unittest.mock import patch, MagicMock
from unittest import TestCase

import numpy as np

from adaptive_publisher.models.oi_cls import OIClsModel
from adaptive_publisher.models.transforms.transf_ocv import (
    to_pytorch_tensor_format,
    to_tensor,
    normalize_image,
    crop_center,
)


from torchvision import transforms
import numpy as np


class TestOCV2Transf(TestCase):

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

