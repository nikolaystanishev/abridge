flake8 --format=pylint \
    --exclude "--exclude \"*old*,*tmp*,*temp*,data-hg19*,gpf*\"" \
    /code > ./pyflakes.report || echo "pylint exited with $?"
