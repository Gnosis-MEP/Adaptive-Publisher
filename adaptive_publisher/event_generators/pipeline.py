from adaptive_publisher.event_generators.from_images import LocalOCVEventGenerator

from adaptive_publisher.models import *

class LocalPipelineEarlyFiltering(LocalOCVEventGenerator):

    def __init__(self, file_storage_cli, publisher_id, input_source, fps, width, height, thresholds):
        super().__init__(file_storage_cli, publisher_id, input_source, fps, width, height)
        self.last_key_frame = None
        self.thresholds = thresholds
        # self.thresholds = {
        #     'diff': 0.05,
        #     'oi_cls': (0.3, 0.7),
        #     'oi_obj': 0.5
        # }

    def check_drop_frame(self, frame):
        return False
