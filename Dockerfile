FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# set working directory
WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y python3-pip ffmpeg git

# update pip
RUN pip3 install --upgrade pip

RUN pip3 install git+https://github.com/openai/whisper.git

# add requirements
COPY ./requirements.txt /app/requirements.txt

# install requirements
RUN pip3 install -r requirements.txt

COPY src/*.py .

# force download of the hugging face models into the container
#RUN python3 download_hf_models_at_buildtime.py 

