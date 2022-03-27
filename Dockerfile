FROM amd64/ubuntu:latest

ARG SOURCE_DIR="."

ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && \ 
    apt-get install -y wget curl unzip gnupg && \
    apt-get clean

RUN curl -sS https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list


RUN apt-get update --fix-missing && \ 
    apt-get install -y libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 \
    chromium-browser google-chrome-stable && \
    apt-get clean


RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip && \
    unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/

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

RUN yarn global add cobertura-merge

WORKDIR /code

SHELL ["/bin/bash", "-c"]
