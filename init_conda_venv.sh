#!/bin/bash

# Make sure to bypass a possible $PIP_REQUIRE_VIRTUALENV=true set by users
# This is because pip is unable to detect that it is inside conda's virtual environment - and would throw an error

if lspci | grep -i nvidia > /dev/null; then
    # Use environment file with cudatoolkit
    echo "Setting up GPU environment"
    environment="environment_gpu.yml"
else
    # Use environment file without cudatoolkit
    echo "Setting up CPU environment"
    environment="environment_cpu.yml"
fi

# Create a properly registered Conda environment
env_name="abb3"  # Change this to your preferred name
PIP_REQUIRE_VIRTUALENV=false conda env create -f ${environment} --name ${env_name}

# Initialize Conda in the shell
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate ${env_name}

# Confirm Python version
python --version

# Install pip dependencies
PIP_REQUIRE_VIRTUALENV=false pip install -e ".[dev]" --constraint pinned-versions.txt

