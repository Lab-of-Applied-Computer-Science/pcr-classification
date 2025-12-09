#!/bin/bash
#SBATCH --job-name=Segmentation_Countour_Vinicius
#SBATCH --time=45:00:00
#SBATCH --nodes=1 

module avail
module load python3.10.12
module load cuda/12.3.2
# module load cuda/11.8.0 

# module list
python3 --version
# conda -V

# module load anaconda3.2023.09-0
# conda env create -f /home_cerberus/speed/daniel.ayala/breast-cancer-segmentation/environment.yaml
# conda activate breast-cancer-env
# cd /home_cerberus/speed/daniel.ayala/breast-cancer-segmentation/bcs-kedro/
# python3 -m pip install .

hostname

if [ -d "/home/all_home/daniel.ayala/" ] 
then
    echo "/home/all_home/daniel.ayala/ exists..."

    if [ -d "/home/all_home/daniel.ayala/exp_notebook_venv" ]
    then
        echo "exp_venv exists..."

        source /home/all_home/daniel.ayala/exp_notebook_venv/bin/activate
    else
        echo "exp_venv doesnt exists, creating it..."

        python3 -m venv /home/all_home/daniel.ayala/exp_notebook_venv/
        source /home/all_home/daniel.ayala/exp_notebook_venv/bin/activate

    fi

    set -x
    pwd 

    echo "installing packages..."
    python3 -m pip install scikit-image numpy opencv-python
    python3 /home_cerberus/speed/daniel.ayala/breast-cancer-segmentation/bcs-kedro/src/segmentation_model/countour.py

else
    echo "/home/all_home/daniel.ayala/ Doesnt exist, nothing happens..."

fi
