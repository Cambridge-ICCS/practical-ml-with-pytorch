#!/bin/bash
#SBATCH --job-name=mlvenv_dawn       # create a short name for your job
#SBATCH --output=%x.log              # job output file
#SBATCH --partition=pvc9             # cluster partition to be used
#SBATCH --nodes=1                    # number of nodes
#SBATCH --gres=gpu:1                 # number of allocated gpus per node
#SBATCH --time=01:00:00              # total run time limit (HH:MM:SS)

# Script for installing MLvenv environment for practical-ml-with-pytorch
# on Dawn supercomputer.

T0=${SECONDS}
ENV_NAME="MLvenv"
PROJ_NAME="practical-ml-with-pytorch"
echo "${ENV_NAME} installation for ${PROJ_NAME} started: $(date)"

# Create script for environment setup.
SETUP=${PROJ_NAME}-setup.sh
cat <<EOF >${SETUP}
# Setup script for activating ${ENV_NAME} on Dawn supercomputer.
# Generated: $(date)

module purge
module load rhel9/default-dawn

# Activate environment.
EOF

# Define installation environment.
source ${SETUP}
module load intelpython-conda/2025.0

# Initialise the virtual enviroment.
VENV_DIR=$(pwd)/${ENV_NAME}
rm -rf ${VENV_DIR}
python -m venv --system-site-packages ${VENV_DIR}
CMD="source ${VENV_DIR}/bin/activate"
echo ${CMD} >> ${SETUP}
${CMD}
pip install --upgrade pip

# Clone project, and install packages in virtual environment.
rm -rf ${PROJ_NAME}
git clone https://github.com/kh296/${PROJ_NAME}
cd ${PROJ_NAME}
git checkout xpu
pip install --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple . ipywidgets

# Create Jupyter kernel.
python -m ipykernel install --user --name=${PROJ_NAME}

echo ""
echo "${ENV_NAME} installation for ${PROJ_NAME} completed: $(date)"
echo "Installation time: $((${SECONDS}-${T0})) seconds"
