import threading
import datetime
import logging
import json

import logzero
import opentracing
from opentracing.ext import tags
from opentracing.propagation import Format

from jaeger_client import Config
from event_service_utils.services.tracer import EVENT_ID_TAG
import uuid

from event_service_utils.img_serialization.cv2 import nd_array_from_ndarray_bytes, DEFAULT_DTYPE
from event_service_utils.streams.redis import RedisStreamFactory
from event_service_utils.img_serialization.redis import RedisImageCache
from event_service_utils.tracing.jaeger import init_tracer

from adaptive_publisher.conf import (
    PUBLISHER_FPS,
    PUBLISHER_HEIGHT,
    PUBLISHER_ID,
    PUBLISHER_INPUT_SOURCE,
    PUBLISHER_WIDTH,
    REDIS_ADDRESS,
    REDIS_PORT,
    REDIS_EXPIRATION_TIME,
    PUB_EVENT_LIST,
    SERVICE_STREAM_KEY,
    SERVICE_CMD_KEY_LIST,
    EVENT_GENERATOR_TYPE,
    EARLY_FILTERING_PIPELINE_NAME,
    LOGGING_LEVEL,
    TRACER_REPORTING_HOST,
    TRACER_REPORTING_PORT,
    SERVICE_DETAILS,
)







class EventPublisher():
    def __init__(self, pub_queue, tracer, publisher_details, query_ids, buffer_stream_key, logging_level):
        self.pub_queue = pub_queue
        self.tracer = tracer
        self.logging_level = logging_level
        self.publisher_details = publisher_details
        self.name = self.publisher_details['publisher_id']
        self.logger = self._setup_logging()

        self.width, self.height = self.publisher_details['meta']['resolution'].split('x')
        self.width = int(self.width)
        self.height = int(self.height)
        self.color_channels = 'BGR'
        self.query_ids = query_ids
        self.tracer_configs = {
            'reporting_host': TRACER_REPORTING_HOST,
            'reporting_port': TRACER_REPORTING_PORT,
            'logging_level': logging.DEBUG
        }

        self.redis_fs_cli_config = {
            'host': REDIS_ADDRESS,
            'port': REDIS_PORT,
            'db': 0,
        }
        self.buffer_stream_key = buffer_stream_key
        self.logger.error('Loggin is ready for Event Publisher!')

    def _setup_logging(self):
        log_format = (
            '%(color)s[%(levelname)1.1s %(name)s %(asctime)s:%(msecs)d '
            '%(module)s:%(funcName)s:%(lineno)d]%(end_color)s %(message)s'
        )
        formatter = logzero.LogFormatter(fmt=log_format)
        return logzero.setup_logger(name=self.name, level=logging.getLevelName(self.logging_level), formatter=formatter)

    def setup_connections(self):
        self.file_storage_cli = RedisImageCache()
        self.file_storage_cli.file_storage_cli_config = self.redis_fs_cli_config
        self.file_storage_cli.expiration_time = REDIS_EXPIRATION_TIME
        self.file_storage_cli.initialize_file_storage_client()
        self.stream_factory = RedisStreamFactory(host=REDIS_ADDRESS, port=REDIS_PORT)

        opentracing._reset_global_tracer()
        Config._initialized_lock = threading.Lock()
        Config._initialized = False

        self.tracer = init_tracer('AdaptivePublisherPublisher', **self.tracer_configs)

        self.logger.error(f"tracer is this: {self.tracer}")
        self.bufferstream = self.stream_factory.create(self.buffer_stream_key, stype='streamOnly')

    def run_forever(self):
        self.setup_connections()
        while True:
            # frame_bytes, frame_index, trace_id = self.service_child_conn.recv()
            frame_bytes, frame_index, trace_id = self.pub_queue.get()
            self.logger.error(f'Received frame: {frame_index}')
            n_channels = 3
            shape = (self.height, self.width, n_channels)
            frame = nd_array_from_ndarray_bytes(frame_bytes, shape, dtype=DEFAULT_DTYPE)
            thread = threading.Thread(target=self.generate_and_send_event, args=(frame, frame_index, trace_id))
            thread.start()

    def default_event_serializer(self, event_data):
        event_msg = {'event': json.dumps(event_data)}
        return event_msg

    def inject_current_tracer_into_event_data(self, event_data):
        tracer_data = event_data.setdefault('tracer', {})
        tracer_headers = tracer_data.setdefault('headers', {})
        with self.tracer.start_active_span('tracer_injection') as scope:
            scope.span.set_tag(EVENT_ID_TAG, event_data['id'])
            self.tracer.inject(scope.span, Format.HTTP_HEADERS, tracer_headers)
        return event_data

    def serialize_and_write_event_with_trace(self, event_data, serializer, destination_stream):
        event_data = self.inject_current_tracer_into_event_data(event_data)
        event_msg = serializer(event_data)
        return destination_stream.write_events(event_msg)

    def event_trace_for_method_with_event_data(
            self, method, method_args, method_kwargs, get_event_tracer=False, tracer_tags=None):
        span_name = method.__name__
        if tracer_tags is None:
            tracer_tags = {}

        tracer_kwargs = {}
        if get_event_tracer:
            event_data = method_kwargs['event_data']
            tracer_kwargs = self.get_event_tracer_kwargs(event_data)
        with self.tracer.start_active_span(span_name, **tracer_kwargs) as scope:
            for tag, value in tracer_tags.items():
                scope.span.set_tag(tag, value)
            method(*method_args, **method_kwargs)

    def write_event_with_trace(self, event_data, destination_stream, serializer=None):
        if serializer is None:
            serializer = self.default_event_serializer
        self.event_trace_for_method_with_event_data(
            method=self.serialize_and_write_event_with_trace,
            method_args=(),
            method_kwargs={
                'event_data': event_data,
                'serializer': serializer,
                'destination_stream': destination_stream
            },
            get_event_tracer=False,
            tracer_tags={
                tags.MESSAGE_BUS_DESTINATION: destination_stream.key,
                tags.SPAN_KIND: tags.SPAN_KIND_PRODUCER,
                EVENT_ID_TAG: event_data['id'],
            }
        )

    def generate_and_send_event(self, frame, frame_index, trace_id):
        span_ctx = self.tracer.extract(Format.HTTP_HEADERS, {'uber-trace-id': trace_id})
        with self.tracer.start_active_span('generate_and_send_event', child_of=span_ctx) as scope:
            event_data = self.generate_event_from_frame(frame, frame_index)

            self.write_event_with_trace(event_data, self.bufferstream)
            self.logger.error(f'sending event_data "{event_data}", to buffer stream: "{self.buffer_stream_key}"')

    def generate_event_from_frame(self, frame, frame_index):
        # Get current UTC timestamp
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')

        event_id = f'{self.publisher_details["publisher_id"]}-{str(uuid.uuid4())}'

        img_uri = self.file_storage_cli.upload_inmemory_to_storage(frame)


        # store_size = getsizeof(frame.tobytes(order='C'))

        # store_size = self.file_storage_cli.client.execute_command(f'MEMORY USAGE {img_uri}')
        # self.service.logger.info('>>> store size: ' + str(store_size) + ' <> ' + str(frame.nbytes))
        event_data = {
            'id': event_id,
            'publisher_id': self.publisher_details['publisher_id'],
            'source': self.publisher_details['source'],
            'image_url': img_uri,
            'vekg': {},
            'query_ids': self.query_ids,
            'width': self.width,
            'height': self.height,
            'color_channels': self.color_channels,
            'frame_index': frame_index,
            'timestamp': timestamp,
        }
        return event_data
