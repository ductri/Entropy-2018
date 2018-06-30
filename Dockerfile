FROM tensorflow/tensorflow:nightly-gpu-py3
MAINTAINER Duc Tri trind@younetgroup.com
ENV REFRESHED_AT 2018-06-21
RUN apt-get -qq update

RUN apt-get -y install python3-tk

RUN pip install ruamel.yaml
RUN pip install nltk
RUN pip install pyvi

VOLUME /source/
VOLUME /all_dataset/

WORKDIR /source

