from adaptive_publisher.event_generators.from_images import LocalOCVEventGenerator

from adaptive_publisher.models.pipeline import ModelPipeline


class LocalPipelineEarlyFiltering(LocalOCVEventGenerator):

    def __init__(self, file_storage_cli, publisher_id, input_source, fps, width, height, thresholds):
        super().__init__(file_storage_cli, publisher_id, input_source, fps, width, height)
        self.thresholds = thresholds
        self.pipeline = ModelPipeline(self.thresholds)
        # self.thresholds = {
        #     'diff': 0.05,
        #     'oi_cls': (0.3, 0.7),
        #     'oi_obj': 0.5
        # }

    def check_drop_frame(self, frame):
        return False
