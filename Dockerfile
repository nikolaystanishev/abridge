FROM amd64/ubuntu:latest

ARG SOURCE_DIR="."

RUN apt-get update --fix-missing && \ 
	apt-get install -y wget curl && \
	apt-get clean

# PYTHON
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -O /anaconda.sh && \
    /bin/bash /anaconda.sh -b -p /opt/conda && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/ && \
    rm /anaconda.sh

RUN /opt/conda/bin/conda init bash

ADD ${SOURCE_DIR}/ environment.yml /

RUN /opt/conda/bin/conda env create --name abridge --file /environment.yml
RUN echo "conda activate abridge" >> ~/.bashrc

RUN mkdir /code

# JS
RUN curl https://deb.nodesource.com/setup_17.x | bash
RUN curl https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list

RUN apt-get update && apt-get install -y nodejs yarn
RUN yarn global add eslint

WORKDIR ${SOURCE_DIR}/web/frontend/templates/frontend/
RUN rm -rf node_modules
RUN yarn install

WORKDIR /code

SHELL ["/bin/bash", "-c"]