FROM python:3.8

## install only stuff for ultralistic hard coupled requirements
WORKDIR /service

ADD ./requirements.txt /service/requirements.txt
ADD ./setup.py /service/setup.py
RUN mkdir -p /service/adaptive_publisher/ && \
    touch /service/adaptive_publisher/__init__.py

RUN pip install -r requirements.txt && \
    rm -rf /tmp/pip* /root/.cached

RUN wget -O yolov5n.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt

RUN apt-get update \
    && apt-get install -y \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

## add all the rest of the code and install the actual package
## this should keep the cached layer above if no change to the pipfile or setup.py was done.
ADD . /service
RUN pip install -e . && \
    rm -rf /tmp/pip* /root/.cache
