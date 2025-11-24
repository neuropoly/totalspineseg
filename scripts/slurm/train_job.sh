#!/bin/bash
#SBATCH --account=
#SBATCH --job-name=train101and102     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/scratch/p/p118739/train_totalspineseg/logs/%x_%A_v2.out
#SBATCH --error=/scratch/p/p118739/train_totalspineseg/logs/%x_%A_v2.err
#SBATCH --mail-user=     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Create and activate venv
echo "Creating environment ..."
module load python/3.10
python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

# Install dependencies
echo "Installing dependencies ..."
TOTALSPINESEG=/project/6102268/p118739/code/totalspineseg
pip install "$TOTALSPINESEG"
pip install nnunetv2==2.6.0
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 --upgrade

# Copy nnUNetTrainerDAExt in nnUNet package under nnUNetTrainer section
nnunet_path=$(python -c "import nnunetv2;print(nnunetv2.__path__[0])")
cp "$TOTALSPINESEG"/totalspineseg/trainer/nnUNetTrainerDAExt.py "$nnunet_path"/training/nnUNetTrainer/

# Define paths used:
PATH_NNUNET_RAW_FOLDER="/project/6102268/p118739/data/nnUNet/raw"
PATH_NNUNET_PREPROCESSED_FOLDER="/project/6102268/p118739/data/nnUNet/preprocessed"
PATH_NNUNET_RESULTS_FOLDER="/scratch/p/p118739/train_totalspineseg/results"
PATH_LOGS="/scratch/p/p118739/train_totalspineseg/logs"

# Export nnUNet paths
export nnUNet_raw=${PATH_NNUNET_RAW_FOLDER}
export nnUNet_preprocessed=${PATH_NNUNET_PREPROCESSED_FOLDER}
export nnUNet_results=${PATH_NNUNET_RESULTS_FOLDER}

# Define variables
nnUNetTrainer="nnUNetTrainerDAExt"
nnUNetPlans="nnUNetPlans"
configuration="3d_fullres"

# Printing parameters
echo ""
echo "Running with the following parameters:"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "nnUNetTrainer=${nnUNetTrainer}"
echo "nnUNetPlans=${nnUNetPlans}"
echo "configuration=${configuration}"
echo ""

# Launch jobs
parallel --verbose --jobs 4 ::: \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S');  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 101  $configuration 0 -p $nnUNetPlans -tr $nnUNetTrainer --c 2>&1 | tee $PATH_LOGS/logfile_train101_fold0_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S');  CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 101  $configuration 1 -p $nnUNetPlans -tr $nnUNetTrainer --c 2>&1 | tee $PATH_LOGS/logfile_train101_fold1_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S');  CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 102  $configuration 0 -p $nnUNetPlans -tr $nnUNetTrainer --c 2>&1 | tee $PATH_LOGS/logfile_train102_fold0_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S');  CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 102  $configuration 1 -p $nnUNetPlans -tr $nnUNetTrainer --c 2>&1 | tee $PATH_LOGS/logfile_train102_fold1_\$ts.txt)" \