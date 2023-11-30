#!/bin/bash 

VENV_DIR="../venv"

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then


    echo 'Creating virtual environment with Python 3.11.5'
    python3.11 -m venv ../venv

    echo 'Starting virtual environment'
    source ../venv/bin/activate

    echo 'Loading libraries'
    pip install -r requirements.txt

else
    echo 'Virtual environment already exists'
    echo 'Starting virtual environment'
    source ../venv/bin/activate

fi

# run 'source STARTME.sh' instead of 'sh STARTME.sh'
# otherwise the virtual environment will not be activated
# in the terminal, and you'll have to do it manually
# that's because 'sh' runs the script in a subshell