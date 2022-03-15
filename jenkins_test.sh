#!/bin/bash

export PATH=/opt/conda/envs/abridge/bin:$PATH

py.test -v --cov-config ./coveragerc \
    --junitxml=./coverage/web-junit.xml \
    --cov-append \
    --cov-report=html:./coverage/coverage.html \
    --cov-report=xml:./coverage/coverage.xml \
    --cov web \
    web

chmod a+rwx -R coverage

find . -name "*.pyc" -delete

cd web/frontend/templates/frontend/
rm -rf coverage
yarn test -- --no-watch --no-progress --coverage --ci --reporters=jest-junit
cp coverage/cobertura-coverage.xml ../../../../coverage
cp coverage/js-junit.xml ../../../../coverage
cd -
