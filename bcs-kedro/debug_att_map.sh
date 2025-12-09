#!/bin/bash
#SBATCH --job-name=Attmap_henrique
#SBATCH --time=200:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=gorgona7
#SBATCH --output=/home_cerberus/speed/henrique.colonese/Virtual_Staining/slurm_logs/job_%j.out 
#SBATCH --error=/home_cerberus/speed/henrique.colonese/Virtual_Staining/slurm_logs/job_%j.err


module avail
module load python3.10.12
module load cuda/12.3.2 #gorgona7
# module load cuda/11.8.0  #gorgona5 e 6
module list
python3 --version

hostname

if [ -d "/home/all_home/henrique.colonese/" ] 
then
    echo "/home/all_home/henrique.colonese/ exists..."

    if [ -d "/home/all_home/henrique.colonese/exp_notebook_venv" ]
    then
        echo "exp_venv exists..."

        source /home/all_home/henrique.colonese/exp_notebook_venv/bin/activate
    else
        echo "exp_venv doesnt exists, creating it..."

        python3 -m venv /home/all_home/henrique.colonese/exp_notebook_venv/
        source /home/all_home/henrique.colonese/exp_notebook_venv/bin/activate

    fi

    set -x
    pwd 

    echo "installing packages..."
    python3 -m pip install matplotlib seaborn pandas girder_client girder-client pillow numpy scikit-image imageio rasterio scikit-learn torch torchaudio torchvision torchmetrics tqdm scipy rasterio opencv-python albumentations segmentation-models-pytorch pyarrow
    echo "starting .py"
    python3 /home_cerberus/speed/henrique.colonese/full_pipeline/breast-cancer-segmentation/bcs-kedro/debug_att_map.py
else
    echo "/home/all_home/henrique.colonese/ Doesnt exist, nothing happens..."

fi
