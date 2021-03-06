# jitpc

## Setup

### Anaconda

To recreate the environment; run in a terminal: 

#### Windows

    conda env create -f environment-windows.yml

#### Any OS

    conda env create -f environment.yml
    
or

    conda create --name <env-name> python=3.8
    
    conda activate <env-name>
    
    conda install -c conda-forge imbalanced-learn=0.9.1
    
    conda install pandas=1.2.5

Then copy and paste this project's [src, res] directories into your newly created 
anaconda environment.

## Usage

New ML classification models can be trained, and existing models can be used to predict 
labels for new samples.<br>
Pre-existing models are stored in the res/model directory, 
datasets are stored in the res/data directory, 
and classification reports are stored in the res/report directory.

### Train

To train a new model; run in a terminal:

    python src/train.py

To list all arguments:

    python src/train.py -h

### Classify    

To classify samples using an existing model; run in a terminal: 

    python src/classify.py
    
To list all arguments:

    python src/classify.py -h
    