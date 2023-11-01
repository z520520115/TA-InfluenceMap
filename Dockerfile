FROM python:3
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# CMD ["/bin/bash", "apt-get update --no-install-recommends"]
# CMD ["/bin/bash", "apt-get install libglib2.0-dev"]
COPY . .
CMD ["/bin/bash"]