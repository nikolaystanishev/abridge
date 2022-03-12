/opt/conda/bin/conda activate abridge

rm -rf coverage/ && mkdir coverage

py.test -v --cov-config coveragerc \
    --junitxml=coverage/web-junit.xml \
    --cov-append \
    --cov-report=html:./coverage/coverage.html \
    --cov-report=xml:./coverage/coverage.xml \
    --cov web/ \
    web

chmod a+rwx -R coverage

find . -name "*.pyc" -delete

cd web/frontend/templates/frontend/
yarn test -- --no-watch --no-progress --code-coverage
cp coverage/cobertura-coverage.xml ../../../../coverage
cp coverage/junit/junit.xml ../../../../coverage/js-junit.xml
cd -
