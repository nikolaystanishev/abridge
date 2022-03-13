#!/bin/bash

export PATH=/opt/conda/envs/abridge/bin:$PATH

flake8 --format=pylint \
    --exclude "--exclude \"*old*,*tmp*,*temp*,data-hg19*,gpf*\"" \
    /code > ./pyflakes.report || echo "pylint exited with $?"

cd web/frontend/templates/frontend/
yarn lint --format checkstyle > ts-lint-report.xml
sed '1,2d' ts-lint-report.xml > ts-lint-report.xml
sed '$d' ts-lint-report.xml > ts-lint-report.xml
cp ts-lint-report.xml ../../../../
cd -
