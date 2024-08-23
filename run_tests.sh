#!/bin/bash
DEFAULT_CACHED_FRAME_RATE_MULTIPLIER=1.0 python -m unittest discover -s tests
CLS_MODEL_ID=TS-D-Q-1-10S_-300_car_person-bird-dog python -m unittest discover -s non_mocked_tests