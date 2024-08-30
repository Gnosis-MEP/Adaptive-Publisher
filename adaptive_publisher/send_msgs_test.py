#!/usr/bin/env python
import uuid
import json
from event_service_utils.streams.redis import RedisStreamFactory

from adaptive_publisher.conf import (
    REDIS_ADDRESS,
    REDIS_PORT,
    SERVICE_STREAM_KEY,
    LISTEN_EVENT_TYPE_QUERY_CREATED,
    PUBLISHER_ID,
)


def make_dict_key_bites(d):
    return {k.encode('utf-8'): v for k, v in d.items()}


def new_msg(event_data):
    event_data.update({'id': str(uuid.uuid4())})
    return {'event': json.dumps(event_data)}





def main():
    stream_factory = RedisStreamFactory(host=REDIS_ADDRESS, port=REDIS_PORT)
#     LISTEN_EVENT_TYPE_QUERY_CREATED
#    "QueryCreated" entity:

    # for checking published events output
    new_event_type_cmd = stream_factory.create(LISTEN_EVENT_TYPE_QUERY_CREATED, stype='streamOnly')

    # for testing sending msgs that the service listens to:
    # import ipdb; ipdb.set_trace()
    # some_event_type_cmd = stream_factory.create(LISTEN_EVENT_TYPE_SOME_EVENT_TYPE, stype='streamOnly')
    new_event_type_cmd.write_events(
        new_msg(
            {
                'subscriber_id': 'sub_id',
                'query_id': '1ca3c67d12721b8ea3bf234746348509',
                'parsed_query': 'doesntmatter',
                'query_received_event_id': 'doesntmatter',
                'buffer_stream': {
                    'publisher_id': PUBLISHER_ID,
                    'buffer_stream_key': 'somebufferstreamkey',
                    'source': 'doesntmatter',
                    'resolution': 'doesntmatter',
                    'fps': 'doesntmatter'
                },
                'service_chain': 'doesntmatter',
            }
        )
    )
    # import ipdb; ipdb.set_trace()

    # read published events output
    # events = new_event_type_cmd.read_events()
    # print(list(events))
    # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
