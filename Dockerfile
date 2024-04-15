FROM tensorflow/tensorflow:2.16.1-gpu

RUN set -x \
    && apt-get -y update \
    && apt-get -y install software-properties-common vim git libgl1-mesa-glx libxcb-xinerama0 libopenexr-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get install -y python3.11-tk \
    && apt-get -y clean

RUN set -x \
    && pip install -U matplotlib \
    && pip install -U opencv-python \
    && pip install -U tensorboard-plugin-profile \
    && pip install -U py-spy \
    && pip install -U joblib

ENTRYPOINT ["/bin/bash"]