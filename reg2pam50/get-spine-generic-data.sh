#!/bin/bash

sudo apt-get install git-annex -y

for ds in data-multi-subject data-single-subject; do
    # get data from github
    git clone https://github.com/spine-generic/$ds
    cd $ds
    # Remove unnecessary folders and files
    rm -rf code derivatives
    find . -mindepth 1 -maxdepth 1 -type f ! -name 'sub-*' -delete
    # Remove all non-anat directories inside sub-* folders
    find . -type d -regex '\./sub-[^/]+/.*' ! -regex '\./sub-[^/]+/anat' -exec rm -rf {} \; -prune
    # Leave only T1 and T2 files
    find . -type f -regex '\./sub-[^/]+/anat/.*' ! -regex '\./sub-[^/]+/anat/sub-[^_]+_\(T1\|T2\)w\.\(nii\.gz\|json\)' -delete

    # Initialize and get git-annex files
    git annex init
    git annex get

    cd ..
done
