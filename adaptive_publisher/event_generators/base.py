import datetime
import os
import statistics
import uuid

import numpy as np
import cv2

from adaptive_publisher.models.base_pipeline import MockedPipeline
from adaptive_publisher.models.pipelines import BLSingleModelPipeline, ModelPipeline
from adaptive_publisher.conf import DEFAULT_OI_LIST, EXAMPLE_IMAGES_PATH, REGISTER_EVAL_DATA


class OCVEventGenerator():
    def __init__(self, service, ef_pipeline_name, file_storage_cli, publisher_id, input_source, fps, width, height, thresholds):
        self.service = service
        self.thresholds = thresholds
        self.ef_pipeline_name = ef_pipeline_name
        self.file_storage_cli = file_storage_cli
        self.publisher_id = publisher_id
        self.input_source = input_source
        self.fps = fps
        self.width = width
        self.height = height
        self.query_ids = []
        self.current_frame_index = -1
        self.color_channels = 'BGR'
        self.source_uri = f'genosis://{publisher_id}/{input_source}'
        self.cap = None
        self.ef_pipeline = None
        self.setup_ef_pipeline()
        self.exp_eval_data = {'results': {}}

    def _get_experiment_eval_data(self):
        self.exp_eval_data['stats'] = self.get_stats_dict()
        return self.exp_eval_data

    def setup_ef_pipeline(self):
        if self.ef_pipeline_name == 'ModelPipeline':
            self.ef_pipeline =  ModelPipeline(
                self.fps, self.thresholds, oi_label_list=DEFAULT_OI_LIST)
        else:
            self.ef_pipeline = BLSingleModelPipeline(
                self.fps, self.thresholds, self.ef_pipeline_name, oi_label_list=DEFAULT_OI_LIST)

    def _clean_input_source(self):
        try:
            clean_source = int(self.input_source)
        except Exception as e:
            clean_source = self.input_source
        return clean_source

    def setup(self):
        clean_source = self._clean_input_source()
        self.cap = cv2.VideoCapture(clean_source)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise Exception(f'Capture for input source {clean_source} is not oppening, wont be able to read data from source')

    def is_open(self):
        return (self.cap and self.cap.isOpened())

    def add_query_id(self, new_query_id):
        self.query_ids = list(set(self.query_ids).add(new_query_id))

    def read_next_frame(self):
        # print('reading next frame')
        # raise NotImplementedError()
        # ovc read stuff
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_index += 1
                # print(f'read next frame: {self.current_frame_index}')
                return frame

    def check_drop_frame(self, frame):
        if self.ef_pipeline is None:
            # print(f'NO EF PIPELINE AVAILABLE!!!')
            has_oi = True
        else:
            has_oi = self.ef_pipeline.predict(frame)
            # if not has_oi:
            #     print('DROP!')
        if REGISTER_EVAL_DATA:
            self.exp_eval_data['results'][self.current_frame_index] = has_oi
        return not has_oi

    def generate_event_from_frame(self, frame):
        # Get current UTC timestamp
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')

        event_id = f'{self.publisher_id}-{str(uuid.uuid4())}'

        # cv2.imwrite(os.path.join(EXAMPLE_IMAGES_PATH, 'testing.jpg'), frame)
        # nd_shape = (self.height, self.width, 3)
        img_uri = self.file_storage_cli.upload_inmemory_to_storage(frame)
        # reimage_nd_array = self.file_storage_cli.get_image_ndarray_by_key_and_shape(img_uri, nd_shape)
        # cv2.imwrite('testing2.jpg', frame)

        event_data = {
            'id': event_id,
            'publisher_id': self.publisher_id,
            'source': self.source_uri,
            'image_url': img_uri,
            'vekg': {},
            'query_ids': self.query_ids,
            'width': self.width,
            'height': self.height,
            'color_channels': self.color_channels,
            'frame_index': self.current_frame_index,
            'timestamp': timestamp,
        }
        return event_data

    def next_event(self):
        next_frame = self.read_next_frame()
        if self.current_frame_index % 100 == 0:
            self.service.logger.info(f'Current Frame: {self.current_frame_index}')
        if next_frame is not None:
            should_drop_frame = self.check_drop_frame(next_frame)
            if not should_drop_frame:
                return self.generate_event_from_frame(next_frame)


    def close(self):
        if self.is_open():
            self.cap.release()

    def get_stats_dict(self):
        times_copy = self.ef_pipeline.processing_times.copy()

        stats_dict = {}
        for k, v in times_copy.items():
            stats_dict[f'{k}_avg'] = statistics.mean(v) if len(v) > 0 else 0
            stats_dict[f'{k}_std'] = statistics.stdev(v) if len(v) > 1 else 0
            stats_dict[f'FPS_{k}_avg'] = 1 / statistics.mean(v) if len(v) > 0 else 0
            stats_dict[f'FPS_{k}_std'] = 1 / statistics.stdev(v) if len(v) > 1 else 0
            stats_dict[f'FPS_{k}_50_75_90_perc'] = [float(f) for f in 1 / np.percentile(v, [50, 75, 90])] if len(v) > 0 else 0
            stats_dict[f'{k}_SIZE'] = len(v)
            if 'predict' in k:
                stats_dict['processing_times'] = v

        return stats_dict


class MockedEventGenerator(OCVEventGenerator):

    def setup_ef_pipeline(self):
        self.ef_pipeline =  MockedPipeline(self.fps, self.thresholds, DEFAULT_OI_LIST)

    def setup(self):
        pass

    def is_open(self):
        return False


