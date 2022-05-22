#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No input: quit."
    exit 1
fi

lfs_files=($(git lfs ls-files -n -I $@))
for file in "${lfs_files[@]}"; do
    git cat-file -e "HEAD:${file}" && git cat-file -p "HEAD:${file}" > "$file"
done
