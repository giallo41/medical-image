FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update -y
RUN apt-get update
RUN apt-get install -y python
RUN apt-get install -y python-pip
RUN pip install --upgrade pip

WORKDIR /

COPY . .

RUN pip install -r requirements.txt
