FROM amd64/ubuntu:latest

ARG SOURCE_DIR="."

RUN apt-get update --fix-missing && \ 
	apt-get install wget && \
	apt-get clean


# ANACONDA

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -O /anaconda.sh && \
    /bin/bash /anaconda.sh -b -p /opt/conda && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/ && \
    rm /anaconda.sh

RUN /opt/conda/bin/conda init bash

ADD ${SOURCE_DIR}/ environment.yml /

RUN /opt/conda/bin/conda create -c conda-forge \
    --name abridge --file /environment.yml
RUN echo "conda activate abridge" >> ~/.bashrc

RUN mkdir /code

WORKDIR /code

SHELL ["/bin/bash", "-c"]