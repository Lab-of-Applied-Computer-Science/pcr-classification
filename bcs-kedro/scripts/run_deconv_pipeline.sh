#!/bin/bash
#SBATCH --job-name=ColorDeconv
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=gorgona10

module avail
module load python3.10.12
module load cuda/12.3.2
# module load cuda/11.8.0
# module load anaconda3.2023.09-0

# module list
python3 --version
# conda -V

# variables
base_dir="/home/all_home/gabriel.freddi"
venv_name="breast-cancer-env"
proj_dir="/home_cerberus/disk2/speed/gabriel.freddi/breast-cancer-segmentation/bcs-kedro"
kedro_pipeline="color_deconvolution"
src_dir="/scratch/gabriel.freddi/cache"
dst_dir="/snfs1/speed/gabriel.freddi/vs-deconvs-mi"

mkdir -p /scratch/gabriel.freddi/cache
rm -f /scratch/gabriel.freddi/cache/metadata.parquet
cp /snfs1/speed/gabriel.freddi/vs-mi/metadata.parquet /scratch/gabriel.freddi/cache/

if [ -d "$base_dir" ]; then
    echo "$base_dir exists..."

    if [ -d "$base_dir/$venv_name" ]; then
        echo "$venv_name exists..."
        source "$base_dir/$venv_name/bin/activate"
    else
        echo "$venv_name does not exist, creating it from zero..."
        set -x
        pwd
        echo "Installing project dependencies..."
        cd "$proj_dir"
        python3 -m venv "$base_dir/$venv_name"
        source "$base_dir/$venv_name/bin/activate"
        python3 -m pip install .
    fi

    echo "Running kedro pipeline..."
    python3 -m kedro run --pipeline="$kedro_pipeline"

else
    echo "$base_dir does not exist, creating it..."
    mkdir -p "$base_dir"
    python3 -m venv "$base_dir/$venv_name"
    source "$base_dir/$venv_name/bin/activate"
    python3 -m pip install .
    echo "Running kedro pipeline..."
    python3 -m kedro run --pipeline="$kedro_pipeline"
fi
