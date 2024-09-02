import datetime
import uuid

import cv2

from adaptive_publisher.models.base_pipeline import MockedPipeline
from adaptive_publisher.event_generators.base import OCVEventGenerator
from adaptive_publisher.conf import CLOUDSEG_SCALE, DEFAULT_OI_LIST


class CloudSegOCVEventGenerator(OCVEventGenerator):

    def setup_ef_pipeline(self):
        self.ef_pipeline = MockedPipeline(
            self.fps, self.thresholds, oi_label_list=DEFAULT_OI_LIST)

    def generate_event_from_frame(self, frame):
        with self.service.tracer.start_active_span('generate_event_from_frame') as scope:
            # Get current UTC timestamp
            timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')

            event_id = f'{self.publisher_id}-{str(uuid.uuid4())}'
            height, width  = frame.shape[:2]
            res_frame = cv2.resize(frame, (width // CLOUDSEG_SCALE, height // CLOUDSEG_SCALE), interpolation=cv2.INTER_LINEAR)

            img_uri = self.file_storage_cli.upload_inmemory_to_storage(res_frame)


            # store_size = getsizeof(frame.tobytes(order='C'))

            # store_size = self.file_storage_cli.client.execute_command(f'MEMORY USAGE {img_uri}')
            # self.exp_eval_data['storage'].append(store_size)
            # self.service.logger.info('>>> store size: ' + str(store_size) + ' <> ' + str(frame.nbytes))
            # img_uri = 'mocked_img_uri'

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
