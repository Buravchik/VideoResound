# Installing and running the project

## Install miniforge

curl -L -O "<https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh>"
bash Miniforge3-MacOSX-arm64.sh

"$(/Users/alexeyisachenko/miniforge3/bin/conda shell.zsh hook)"

## Create fresh environment

conda deactivate
conda create -n tts_env_new python=3.10
conda activate tts_env_new
conda config --env --set subdir osx-64

## Install numpy first with a version that satisfies most dependencies

pip install "numpy>=1.23.0,<2.0.0"

## Install torch and related packages

pip install torch torchvision torchaudio

## First install spacy separately with pre-built wheels

pip install "spacy[ja]"

## Then install TTS with an older version

pip install TTS==0.21.0# First remove any existing spacy installation
pip uninstall spacy -y

## Install a specific version of Cython known to work

pip install "Cython<3.0.0"

## Install a specific version of spacy with pre-built wheels

pip install "spacy[ja]==3.7.2"

## Then proceed with TTS installation

pip install TTS==0.21.0
