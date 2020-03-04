#!/bin/bash

# upload to machine: masudur-gpu6-56gb-4
rsync -av -e ssh --exclude='.git/' --exclude='train.csv' --exclude='test.csv' --exclude='*.sh' --exclude='__pycache__' --exclude='result/' `pwd` mnasim@data.cs.purdue.edu:/homes/mnasim/