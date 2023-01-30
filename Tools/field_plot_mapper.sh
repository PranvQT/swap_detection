#!/bin/bash

#Virtualenvwrapper settings:
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_VIRTUALENV=/home/quidich/.local/bin/virtualenv
source ~/.local/bin/virtualenvwrapper.sh
cd /home/quidich/Downloads/development_docker/code_v2/qt-deployment/Tools
workon cv
python3 field_plot_mapper.py
