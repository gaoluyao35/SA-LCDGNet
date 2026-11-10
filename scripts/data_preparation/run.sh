
#!/bin/bash

# Data preparation for SA-LCDGNet
# Supported datasets: PEMS03, PEMS04, PEMS07, PEMS08

python scripts/data_preparation/PEMS03/generate_training_data.py
python scripts/data_preparation/PEMS04/generate_training_data.py
python scripts/data_preparation/PEMS07/generate_training_data.py
python scripts/data_preparation/PEMS08/generate_training_data.py
