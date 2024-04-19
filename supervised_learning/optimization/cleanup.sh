#!/bin/bash

# Replace these filenames with the ones you want to remove
file1="supervised_learning/classification/myenv/lib/python3.12/site-packages/clang/native/libclang.dylib"
file2="supervised_learning/classification/myenv/lib/python3.12/site-packages/tensorflow/libtensorflow_cc.2.dylib"

# Run BFG Repo-Cleaner
java -jar bfg-1.14.0.jar  --delete-files "$file1" "$file2"

# Force push changes
git push --force
