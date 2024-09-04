import threading
import datetime

import cv2
from opentracing.propagation import Format

from event_service_utils.services.tracer import EVENT_ID_TAG
import uuid


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
    REDUCE_SCALE,
)




class EventPublisher():
    def __init__(self, parent_service, publisher_details, query_ids, buffer_stream_key):
        self.parent_service = parent_service
        self.file_storage_cli = self.parent_service.file_storage_cli
        self.tracer = self.parent_service.tracer
        self.logger = self.parent_service.logger
        self.publisher_details = publisher_details

        self.width, self.height = self.publisher_details['meta']['resolution'].split('x')
        self.width = int(self.width)
        self.height = int(self.height)
        self.color_channels = 'BGR'
        self.query_ids = query_ids

        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.frame_ready = False
        self.frame_sent = True

        self.buffer_stream_key = buffer_stream_key
        self.bufferstream = self.parent_service.stream_factory.create(self.buffer_stream_key, stype='streamOnly')

    # def run_forever(self):
    #     while True:
    #         with self.condition:
    #             while not self.frame_ready:
    #                 self.condition.wait()

    #             frame, frame_index, trace_id = self.frame_data
    #             self.frame_ready = False

    #         self.generate_and_send_event(frame, frame_index, trace_id)
    #         with self.condition:
    #             self.frame_sent = True
    #             self.condition.notify()

    def run(self):
        with self.condition:
            while not self.frame_ready:
                self.condition.wait()

            frame, frame_index, trace_id = self.frame_data
            self.frame_ready = False

        self.generate_and_send_event(frame, frame_index, trace_id)
        with self.condition:
            self.frame_sent = True
            self.condition.notify()

    def generate_and_send_event(self, frame, frame_index, trace_id):
        span_ctx = self.tracer.extract(Format.HTTP_HEADERS, {'uber-trace-id': trace_id})
        with self.tracer.start_active_span('generate_and_send_event', child_of=span_ctx) as scope:
            event_data = self.generate_event_from_frame(frame, frame_index)

            self.parent_service.write_event_with_trace(event_data, self.bufferstream)
            self.logger.info(f'sending event_data "{event_data}", to buffer stream: "{self.buffer_stream_key}"')

    def generate_event_from_frame(self, frame, frame_index):
        # Get current UTC timestamp
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')

        event_id = f'{self.publisher_details["publisher_id"]}-{str(uuid.uuid4())}'

        if REDUCE_SCALE > 1:
            frame = cv2.resize(frame, (self.width // REDUCE_SCALE, self.height // REDUCE_SCALE), interpolation=cv2.INTER_LINEAR)

        img_uri = self.file_storage_cli.upload_inmemory_to_storage(frame)


        # store_size = getsizeof(frame.tobytes(order='C'))

        # store_size = self.file_storage_cli.client.execute_command(f'MEMORY USAGE {img_uri}')
        # self.parent_service.logger.info('>>> store size: ' + str(store_size) + ' <> ' + str(frame.nbytes))
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
