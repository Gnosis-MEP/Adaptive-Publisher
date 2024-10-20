import datetime
import os
# from sys import getsizeof
import multiprocessing
import threading
import statistics
import time
import uuid

import numpy as np
import cv2

from opentracing.ext import tags
from opentracing.propagation import Format
from event_service_utils.services.tracer import EVENT_ID_TAG

from adaptive_publisher.models.base_pipeline import MockedPipeline
from adaptive_publisher.models.pipelines import BLSingleModelPipeline, ModelPipeline, MockedPipeline
from adaptive_publisher.conf import DEFAULT_OI_LIST, EXAMPLE_IMAGES_PATH, REGISTER_EVAL_DATA, TMP_IGNORE_N_FRAMES, IGNORE_SEND_IMAGE


class OCVEventGenerator():
    def __init__(self, service, ef_pipeline_name, publisher_id, input_source, fps, width, height, thresholds):
        self.service = service
        self.thresholds = thresholds
        self.ef_pipeline_name = ef_pipeline_name
        self.publisher_id = publisher_id
        self.input_source = input_source
        self.fps = fps
        self.frame_delay = 1 / fps
        self.width = width
        self.height = height
        self.query_ids = []
        self.current_frame_index = -1
        self.color_channels = 'BGR'
        self.source_uri = f'gnosis://{publisher_id}/{input_source.split("/")[-1]}'
        self.cap = None
        self.ef_pipeline = None
        self.setup_ef_pipeline()
        self.publisher_details = {
            'publisher_id': self.publisher_id,
            'source': self.source_uri,
            'meta':{
                'color': 'True',
                'fps': str(self.fps),
                'resolution': f'{width}x{height}'
            }
        }
        self.exp_eval_data = {
            'results': {},
            # 'storage': []
        }
        self.last_event_time = time.perf_counter()

    def _get_experiment_eval_data(self):
        self.exp_eval_data['stats'] = self.get_stats_dict()
        return self.exp_eval_data

    def setup_ef_pipeline(self):
        if self.ef_pipeline_name == 'ModelPipeline':
            self.ef_pipeline =  ModelPipeline(
                self.fps, self.thresholds, oi_label_list=DEFAULT_OI_LIST)
        elif self.ef_pipeline_name == 'SingleModelPipeline':
            self.ef_pipeline = BLSingleModelPipeline(
                self.fps, self.thresholds, self.ef_pipeline_name, oi_label_list=DEFAULT_OI_LIST)
        else:
            self.ef_pipeline = MockedPipeline(
                self.fps, self.thresholds, oi_label_list=DEFAULT_OI_LIST)

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

        # for demo
        # cv2.namedWindow("OriginalFrame", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("OriginalFrame", 1280, 720)
        # can_go = input('start?')

        if not self.cap.isOpened():
            raise Exception(f'Capture for input source {clean_source} is not oppening, wont be able to read data from source')

    def is_open(self):
        return (self.cap and self.cap.isOpened())

    def add_query_id(self, new_query_id):
        query_set = set(self.query_ids)
        query_set.add(new_query_id)
        self.query_ids = list(query_set)

    def read_next_frame(self):
        with self.service.tracer.start_active_span('read_next_frame') as scope:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame_index += 1
                    # cv2.imshow('OriginalFrame', frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'): # wait for 1 millisecond
                    #     pass

                    # print(f'read next frame: {self.current_frame_index}')
                    return frame
                else:
                    self.close()

    def check_drop_frame(self, frame):
        with self.service.tracer.start_active_span('early_filtering') as scope:
            if self.ef_pipeline is None:
                # print(f'NO EF PIPELINE AVAILABLE!!!')
                has_oi = True
            else:
                has_oi = self.ef_pipeline.predict(frame)
                # if not has_oi:
                #     print('DROP!')

            scope.span.set_tag('filter_exit_on', self.ef_pipeline.last_frame_step_exit)

            if REGISTER_EVAL_DATA:
                self.exp_eval_data['results'][f'frame_{TMP_IGNORE_N_FRAMES + self.current_frame_index + 1}'] = has_oi
            return not has_oi

    def read_next_frame_or_drop(self):
        next_frame = self.read_next_frame()

        if self.current_frame_index % 100 == 0:
            self.service.logger.info(f'Current Frame: {self.current_frame_index}')

        if next_frame is not None:
            should_drop_frame = self.check_drop_frame(next_frame)
            if should_drop_frame:
                next_frame = None
        return next_frame

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


