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

python3 /home_cerberus/speed/daniel.ayala/breast-cancer-segmentation/bcs-kedro/src/segmentation_model/countour.py

else
    echo "/home/all_home/daniel.ayala/ Doesnt exist, nothing happens..."

fi
