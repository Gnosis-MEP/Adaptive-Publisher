import time
import threading

from event_service_utils.logging.decorators import timer_logger
from event_service_utils.services.event_driven import BaseEventDrivenCMDService
from event_service_utils.tracing.jaeger import init_tracer

from adaptive_publisher.conf import LISTEN_EVENT_TYPE_EARLY_FILTERING_UPDATED
from adaptive_publisher.event_generators.rng import RngEarlyFiltering

class AdaptivePublisher(BaseEventDrivenCMDService):
    def __init__(self,
                 service_stream_key, service_cmd_key_list,
                 pub_event_list, service_details,
                 stream_factory,
                 file_storage_cli,
                 publisher_configs,
                 event_generator_type,
                 logging_level,
                 tracer_configs):
        tracer = init_tracer(self.__class__.__name__, **tracer_configs)
        super(AdaptivePublisher, self).__init__(
            name=self.__class__.__name__,
            service_stream_key=service_stream_key,
            service_cmd_key_list=service_cmd_key_list,
            pub_event_list=pub_event_list,
            service_details=service_details,
            stream_factory=stream_factory,
            logging_level=logging_level,
            tracer=tracer,
        )
        self.cmd_validation_fields = ['id']
        self.data_validation_fields = ['id']
        self.event_generator_type = event_generator_type
        self.available_event_generators = {
            'RngEarlyFiltering': RngEarlyFiltering,
        }
        self.file_storage_cli = file_storage_cli
        self.event_generator = None
        self.bufferstream_dict = {}
        self.early_filtering_rules = {}
        self.publisher_configs = publisher_configs
        self._fake_query_setup()
        self.setup_event_generator()

    def _fake_query_setup(self):
        self.bufferstream_dict['bufferstream'] = {
            'bufferstream': self.stream_factory.create('bufferstream', stype='streamOnly')
        }


    def setup_event_generator(self):
        self.event_generator = self.available_event_generators[self.event_generator_type](
            self.file_storage_cli,
            self.publisher_configs['id'],
            self.publisher_configs['input_source'],
            self.publisher_configs['fps'],
            self.publisher_configs['width'],
            self.publisher_configs['height'],
        )
        self.event_generator.setup()

    def process_data(self):
        self.logger.debug('Processing DATA..')
        buffer_stream_key_list = self.bufferstream_dict.keys()
        no_bufferstreams = len(buffer_stream_key_list) == 0
        generator_not_open = not self.event_generator.is_open()

        ignore_publishing = no_bufferstreams or generator_not_open
        if ignore_publishing:
            time.sleep(0.05)
            return
        event_data = None
        buffer_stream_key_list = None
        try:
            event_data = self.event_generator.next_event()
            buffer_stream_key_list = self.bufferstream_dict.keys()
            if event_data is not None and len(buffer_stream_key_list) > 0:
                for buffer_stream_key in buffer_stream_key_list:
                    bufferstream_data = self.bufferstream_dict[buffer_stream_key]
                    bufferstream = bufferstream_data['bufferstream']
                    self.write_event_with_trace(event_data, bufferstream)
        except KeyboardInterrupt as ke:
            self.event_generator.close()
        except Exception as e:
            self.logger.error(f'Error processing event_data "{event_data}", while sending to buffer streams: "{buffer_stream_key_list}"')
            self.logger.exception(e)
        finally:
            pass

    def process_early_filtering_updated(self, event_data):
        # if is early filtering for this buffer streams, than do something, otherwise, ignore.
        pass

    def process_event_type(self, event_type, event_data, json_msg):
        if not super(AdaptivePublisher, self).process_event_type(event_type, event_data, json_msg):
            return False
        if event_type == LISTEN_EVENT_TYPE_EARLY_FILTERING_UPDATED:
            self.process_early_filtering_updated(event_data=event_data)
            # do some processing
            pass

    def log_state(self):
        super(AdaptivePublisher, self).log_state()
        self.logger.info(f'Service name: {self.name}')
        # function for simple logging of python dictionary
        self._log_dict('Publishing to bufferstreams:', self.bufferstream_dict)
        self._log_dict('Early filtering rules:', self.bufferstream_dict)

    def run(self):
        super(AdaptivePublisher, self).run()
        self.log_state()

        self.cmd_thread = threading.Thread(target=self.run_forever, args=(self.process_cmd,))
        self.data_thread = threading.Thread(target=self.run_forever, args=(self.process_data,))
        self.cmd_thread.start()
        self.data_thread.start()
        self.cmd_thread.join()
        self.data_thread.join()

