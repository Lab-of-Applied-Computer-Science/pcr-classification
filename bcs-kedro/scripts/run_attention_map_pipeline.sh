#!/bin/bash
#SBATCH --job-name=AttMapKI67_Normal
#SBATCH --time=200:00:00
#SBATCH --nodes=1 
#SBATCH --nodelist=gorgona4
#SBATCH --output=/home_cerberus/speed/henrique.colonese/breast-cancer-segmentation/slurm-logs/slurm-%j.log

module avail
module load python3.10.12
module load cuda/12.3.2

# module load cuda/11.8.0 
# module load anaconda3.2023.09-0

module list
python3 --version

# Configure debugging
set -x

# Configurable parameters
# username="henrique.colonese"
username=$(whoami)
update_deps="true"
kedro_pipeline="biomarker_attention_map"

# Constant parameters
base_dir="/home/all_home/$username/"
# base_dir="/snfs1/speed/$username/"
venv_name="breast-cancer-env"
proj_dir="/home_cerberus/speed/henrique.colonese/full_pipeline/breast-cancer-segmentation/bcs-kedro/"

###########################
#       Functions         #
###########################

export KERAS_HOME="$base_dir/data/06_models/.keras"
export CLEARML_CONFIG_FILE="$proj_dir/conf/local/clearml.conf"

create_and_activate_venv() {
    if [ -d "$base_dir/$venv_name" ]; then
        echo "$venv_name exists..."
    else
        echo "$venv_name does not exist, creating it..."
        python3 -m venv $base_dir/$venv_name/
    fi
    echo "Activating $venv_name..."
    source $base_dir/$venv_name/bin/activate
}

install_dependencies() {
    echo "Installing project dependencies..."
    python3 -m pip install .
}

run_pipeline() {
    echo "Running kedro pipeline..."
    python3 -m kedro run --pipeline=$kedro_pipeline
}

###########################
#       Main Script       #
###########################

mkdir -p $base_dir
cd $proj_dir
create_and_activate_venv
if [ "$update_deps" = "true" ]; then
    install_dependencies
fi
run_pipeline
