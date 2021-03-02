FROM python:3.6

MAINTAINER YauFan <yaufan0625@gmail.com>

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

RUN mkdir -p /usr/src/deepknee
RUN mkdir -p /usr/src/deepknee/Data
RUN mkdir -p /usr/src/deepknee/QData
RUN mkdir -p /usr/src/deepknee/EData
RUN mkdir -p /usr/src/deepknee/Result

WORKDIR /usr/src/deepknee

# Installing python dependencies
COPY . /usr/src/deepknee

RUN pip install --upgrade pip setuptools
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-python-headless

# Running Python Application
CMD ["python", "Knee_and_classification_infer_v3.py"]
