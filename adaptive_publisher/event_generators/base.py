import datetime
import uuid

import cv2

class OCVEventGenerator():
    def __init__(self, file_storage_cli, publisher_id, input_source, fps, width, height):
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
        print('reading next frame')
        # raise NotImplementedError()
        # ovc read stuff
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_index += 1
                print(f'read next frame: {self.current_frame_index}')
                return frame


    def check_drop_frame(self, frame):
        raise NotImplementedError()

    def generate_event_from_frame(self, frame):
        # Get current UTC timestamp
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')

        event_id = f'{self.publisher_id}-{str(uuid.uuid4())}'
        # cv2.imwrite('testing.jpg', frame)
        img_uri = self.file_storage_cli.upload_inmemory_to_storage(frame)
        # nd_shape = (self.height, self.width, 3)
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
        if next_frame is not None:
            should_drop_frame = self.check_drop_frame(next_frame)
            if not should_drop_frame:
                return self.generate_event_from_frame(next_frame)

    def close(self):
        if self.is_open():
            self.cap.release()
