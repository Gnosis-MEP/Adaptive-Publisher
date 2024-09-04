import json
import time
import multiprocessing
import threading

from opentracing.ext import tags
from opentracing.propagation import Format
from event_service_utils.logging.decorators import timer_logger
from event_service_utils.services.event_driven import BaseEventDrivenCMDService
from event_service_utils.tracing.jaeger import init_tracer

from adaptive_publisher.event_publishers.publisher import EventPublisher

from adaptive_publisher.conf import (
    LISTEN_EVENT_TYPE_EARLY_FILTERING_UPDATED,
    LISTEN_EVENT_TYPE_QUERY_CREATED,
    LISTEN_EVENT_TYPE_QUERY_REMOVED,
    PUB_EVENT_TYPE_PUBLISHER_CREATED,
    TMP_EXP_EVAL_DATA_JSON_PATH,
    DEFAULT_THRESHOLDS,
    DEFAULT_TARGET_FPS,
    PUB_QUEUE_MAX_SIZE,
    IGNORE_SEND_IMAGE,
)
from adaptive_publisher.event_generators import OCVEventGenerator, LocalOCVEventGenerator, CloudSegOCVEventGenerator, MockedEventGenerator

class AdaptivePublisher(BaseEventDrivenCMDService):
    def __init__(self,
                 service_stream_key, service_cmd_key_list,
                 pub_event_list, service_details,
                 stream_factory,
                 publisher_configs,
                 event_generator_type,
                 early_filtering_pipeline_name,
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
            'MockedEventGenerator': MockedEventGenerator,
            'OCVEventGenerator': OCVEventGenerator,
            'CloudSegOCVEventGenerator': CloudSegOCVEventGenerator,
            'LocalOCVEventGenerator': LocalOCVEventGenerator,
        }
        self.early_filtering_pipeline_name = early_filtering_pipeline_name
        self.event_generator = None
        self.bufferstream_dict = {}
        self.early_filtering_rules = {
            'pipeline': self.early_filtering_pipeline_name,
            'thresholds': DEFAULT_THRESHOLDS,
            'target_fps': DEFAULT_TARGET_FPS,
        }
        self.publisher_configs = publisher_configs
        # self._fake_query_setup()
        self.setup_event_generator()
        self.publisher_parent_conn = None
        self.publisher_child_conn = None
        self.publisher = None

    # def _fake_query_setup(self):
    #     # not connected yet with the rest of the system for this control
    #     # so hardcoding here the query details
    #     # in future should listen to query create in cmd thread and add
    #     # the buffer stream for the query in here
    #     self.bufferstream_dict['bufferstream'] = {
    #         'bufferstream': self.stream_factory.create('1dcc691eca747c0654c42232f7abf12b', stype='streamOnly')
    #     }


    def setup_event_generator(self):
        self.event_generator = self.available_event_generators[self.event_generator_type](
            self,
            self.early_filtering_pipeline_name,
            self.publisher_configs['id'],
            self.publisher_configs['input_source'],
            self.early_filtering_rules['target_fps'],
            self.publisher_configs['width'],
            self.publisher_configs['height'],
            self.early_filtering_rules['thresholds']
        )
        self.event_generator.setup()
        # hardcoding specific query
        # in future should add query from listed event

    def experiment_temporary_exit_data_gathering(self):
        "adding this method just to double check the results and have them saved for later"
        with open(TMP_EXP_EVAL_DATA_JSON_PATH, 'w') as f:
            json.dump(self.event_generator._get_experiment_eval_data(), f, indent=4)

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
            buffer_stream_key_list = self.bufferstream_dict.keys()
            if len(buffer_stream_key_list) > 0:
                with self.tracer.start_active_span('process_next_frame', child_of=None) as scope:
                    tracer_tags = {
                        tags.SPAN_KIND: tags.SPAN_KIND_PRODUCER,
                    }
                    for tag, value in tracer_tags.items():
                        scope.span.set_tag(tag, value)
                    init_time = time.perf_counter()
                    frame = self.event_generator.read_next_frame_or_drop()
                    if frame is not None:
                        if not IGNORE_SEND_IMAGE:
                            tracer_headers = {}
                            self.logger.info('will send data do proc>')
                            self.tracer.inject(scope.span, Format.HTTP_HEADERS, tracer_headers)
                            with self.publisher.condition:
                                self.logger.info('Wait for frame sent...')
                                while not self.publisher.frame_sent:
                                    self.publisher.condition.wait()
                                    
                                self.publisher.frame_data = (frame, self.event_generator.current_frame_index, tracer_headers['uber-trace-id'])
                                self.publisher.frame_ready = True
                                self.publisher.frame_sent = False
                                self.publisher.condition.notify()
                            #self.publisher_parent_conn.send((
                            #self.pub_queue.put((
                            #    frame.tobytes(order='C'),
                            #    self.event_generator.current_frame_index,
                            #    tracer_headers['uber-trace-id']
                            #))
                    else:
                        self.logger.info(f'Event filtered')

                    current_time = time.perf_counter()
                    elapsed_time = current_time - init_time
                    sleep_time = max(0, self.event_generator.frame_delay - elapsed_time)
                    # ensure correct FPS (e.g., avoid reading frames too fast from disk)
                    time.sleep(sleep_time)

        except KeyboardInterrupt as ke:
            self.event_generator.close()
            raise ke
        except Exception as e:
            self.logger.error(f'Error processing event_data "{event_data}", while sending to buffer streams: "{buffer_stream_key_list}"')
            self.logger.exception(e)
        finally:
            pass
        if not self.event_generator.is_open():
            raise KeyboardInterrupt()

    def process_early_filtering_updated(self, event_data):
        # if is early filtering for this buffer streams, than do something, otherwise, ignore.
        # but, not connected yet with the adaptation engine
        pass

    def process_query_created(self, event_data):
        buffer_stream = event_data['buffer_stream']
        publisher_id = buffer_stream['publisher_id']
        buffer_stream_key = buffer_stream['buffer_stream_key']
        query_id = event_data['query_id']
        if self.publisher_configs['id'] == publisher_id:
            self.event_generator.add_query_id(query_id)
            self.bufferstream_dict[buffer_stream_key] = {
                'bufferstream': buffer_stream_key,
                'query_ids': [query_id]
            }

    def process_event_type(self, event_type, event_data, json_msg):
        if not super(AdaptivePublisher, self).process_event_type(event_type, event_data, json_msg):
            return False
        if event_type == LISTEN_EVENT_TYPE_EARLY_FILTERING_UPDATED:
            self.process_early_filtering_updated(event_data=event_data)
        if event_type == LISTEN_EVENT_TYPE_QUERY_CREATED:
            self.process_query_created(event_data=event_data)
        # if event_type == LISTEN_EVENT_TYPE_QUERY_REMOVED:
        #     self.process_query_removed(event_data=event_data)

    def log_state(self):
        super(AdaptivePublisher, self).log_state()
        self.logger.info(f'Service name: {self.name}')
        # function for simple logging of python dictionary
        self._log_dict('Publishing to bufferstreams:', self.bufferstream_dict)
        self._log_dict('Early filtering rules:', self.early_filtering_rules)
        self._log_dict('Processing Times:', self.event_generator.get_stats_dict())

    def publish_publisher_created(self):
        new_event_data = {
            'id': self.service_based_random_event_id(),
        }
        new_event_data.update(self.event_generator.publisher_details)
        self.publish_event_type_to_stream(event_type=PUB_EVENT_TYPE_PUBLISHER_CREATED, new_event_data=new_event_data)


    def setup_publisher_process(self):
        #self.publisher_parent_conn, self.publisher_child_conn = multiprocessing.Pipe()

        #self.pub_queue = multiprocessing.Queue(maxsize=PUB_QUEUE_MAX_SIZE)
        
        bufferstream_key, first_bufferstream_data = list(self.bufferstream_dict.items())[0]
        self.publisher = EventPublisher(
            #self.pub_queue,
            #self.publisher_child_conn,
            self,
            self.tracer,
            self.event_generator.publisher_details,
            first_bufferstream_data['query_ids'],
            bufferstream_key,
            self.logging_level
        )
        #publisher_process = multiprocessing.Process(target=self.publisher.run_forever)
        publisher_process = None
        return publisher_process

    def run(self):
        super(AdaptivePublisher, self).run()
        self.log_state()
        self.publish_publisher_created()

        try:
            while True:
                self.process_cmd()
                if len(self.bufferstream_dict) != 0:
                    break
        except KeyboardInterrupt:
            self.logger.debug('No query to publish data to, will exit')
            return

        self.logger.debug('Will setup publisher')
        publisher_process = self.setup_publisher_process()
        self.logger.debug('setup publisher done')
        # Start the publishing subprocess
        #publisher_process.start()
        time.sleep(5)
        self.logger.debug('publisher_process.start() done')
        try:
            self.logger.debug('will self.run_forever(self.process_data)')
            rt = threading.Thread(target=self.run_forever, args=(self.process_data,))
            pt = threading.Thread(target=self.publisher.run_forever_thread, args=())
            rt.start()
            pt.start()
        except Exception as e:
            self.logger.exception(e)
        finally:
            rt.join()
            pt.join()
            self.log_state()
            self.experiment_temporary_exit_data_gathering()
            self.logger.debug('will finish publisher process')
            #publisher_process.terminate()
            #publisher_process.join()
            self.logger.debug('finished publisher process')

