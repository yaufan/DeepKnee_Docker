FROM ubuntu
MAINTAINER YauFan <yaufan0625@gmail.com>

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# check our python environment
RUN python3 --version
RUN pip3 --version


# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Running Python Application
CMD ["python", "Knee_and_classification_infer_v2.py"]