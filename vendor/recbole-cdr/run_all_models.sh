#!/bin/bash

echo "Running CoNet model..."
python /app/run_recbole_cdr.py --model=CoNet

echo "Running EMCDR model..."
python /app/run_recbole_cdr.py --model=EMCDR

echo "Running SSCDR model..."
python /app/run_recbole_cdr.py --model=SSCDR

echo "All models have been run."