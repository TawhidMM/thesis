#!/bin/bash

# Exit on error


# Clean Conda cache
conda clean --all
# Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust path if different
conda activate MorphLink_env

# Uninstall existing PyTorch
conda remove pytorch torchvision torchaudio -y
pip uninstall torch torchvision torchaudio -y

# Install PyTorch 2.0.1 with CUDA 12.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install MorphLink dependencies
conda install -c conda-forge mkl mkl-include scanpy anndata numpy pandas imutils opencv scikit-image tifffile slideio texttable lazy-loader -y
pip install igraph==0.11.9 leidenalg==0.10.2

# Reinstall MorphLink
cd ~/Storage/morphlink/MorphLink/MorphLink_package
pip uninstall MorphLink -y
pip install .

# Verify installations
pip list | grep -E "MorphLink|torch"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import MorphLink; print('MorphLink imported successfully')" || echo "Import failed. Check for additional dependencies."

# Export environment
conda env export > ~/Storage/morphlink/environment.yml
echo "Environment exported to ~/Storage/morphlink/environment.yml"