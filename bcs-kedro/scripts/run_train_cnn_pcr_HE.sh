#!/bin/bash
#SBATCH --job-name=trainCNNPCR_9_channels
#SBATCH --time=200:00:00
#SBATCH --nodes=1 
#SBATCH --nodelist=gorgona3
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/home_cerberus/speed/henrique.colonese/full_pipeline/breast-cancer-segmentation/slurm-logs/slurm-%j.log

module avail
module purge  # Clear all loaded modules first
module load python3.10.12
module load cuda/12.3.2
# module load cuda/11.8.0 
set -x

export CUDA_HOME=/usr/local/cuda-12.3.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "===================================="
echo $PATH
echo "===================================="
echo "=== CUDA Verification ==="
which -a nvcc                                           # Should show /usr/local/cuda-12.3/bin/nvcc
nvcc --version                                       # Must show 12.3
ls -l $(which nvcc)                                  # Check symlink resolution
module list                                          # Confirm loaded modules
echo "===================================="

# module load cuda/11.8.0 
# module load anaconda3.2023.09-0

module list
python3 --version

# Configure debugging


hostname

# Configurable parameters
# username="daniel.ayala"
username="henrique.colonese"
update_deps="true"
kedro_pipeline="train_HE"

# Constant parameters
base_dir="/home/all_home/$username/"
venv_name="breast-cancer-env"
# proj_dir="/home_cerberus/speed/$username/breast-cancer-segmentation/bcs-kedro/"
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
    ptxas --version
    nvcc --version
}

echo "=== CUDA/GPU Debug Info ==="
which nvcc
nvcc --version
nvidia-smi
python3 -c "import tensorflow as tf; print('TF Version:', tf.__version__); print('TF Built with CUDA:', tf.test.is_built_with_cuda()); print('Visible Devices:', tf.config.list_physical_devices('GPU'))"
# echo "=========================="
# echo "Copying snfs1 to gorgona scratch"

# SOURCE_DIR="/snfs1/speed/henrique.colonese/breast-cancer-segmentation/bcs-kedro/data/05_model_input/spatial-attention-pcr-attmaps-ihc-only-tensor"
# DEST_DIR="/scratch/henrique.colonese/breast-cancer-segmentation/data/05_model_input/spatial-attention-pcr-attmaps-tiles"
# mkdir -p "$DEST_DIR"
# cp -r "$SOURCE_DIR"/* "$DEST_DIR"/
# echo "All files have been copied from $SOURCE_DIR to $DEST_DIR."

# echo "=========================="


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