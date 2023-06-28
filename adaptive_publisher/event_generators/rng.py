from adaptive_publisher.event_generators.base import OCVEventGenerator

class RngEarlyFiltering(OCVEventGenerator):

    def check_drop_frame(self, frame):
        return False