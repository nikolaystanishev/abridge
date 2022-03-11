conda activate abridge

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
