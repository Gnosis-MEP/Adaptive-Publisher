FROM registry.insight-centre.org/sit/mps/docker-images/base-services:3.7

## install only stuff for ultralistic hard coupled requirements
ADD ./requirements-ultra.txt /service/requirements-ultra.txt
WORKDIR /service
RUN pip install -r requirements-ultra.txt && \
    pip uninstall -y opencv-python  && \
    rm -rf /tmp/pip* /root/.cache

## install only the service requirements
ADD ./Pipfile /service/Pipfile
ADD ./setup.py /service/setup.py
RUN mkdir -p /service/adaptive_publisher/ && \
    touch /service/adaptive_publisher/__init__.py
WORKDIR /service
RUN rm -f Pipfile.lock && pipenv lock -vvv && pipenv --rm && \
    pipenv install --system  && \
    rm -rf /tmp/pip* /root/.cache

## add all the rest of the code and install the actual package
## this should keep the cached layer above if no change to the pipfile or setup.py was done.
ADD . /service
RUN pip install -e . && \
    rm -rf /tmp/pip* /root/.cache
