#!/bin/bash
#SBATCH --account=def-jcohen
#SBATCH --time=00-00:50:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=Yehuda.Warszawer@sheba.health.gov.il
#SBATCH --mail-type=ALL

# Get resources
# salloc --account=def-jcohen --time=0:50:00 --cpus-per-task=16 --mem=128G --gpus-per-node=a100:1

# Get params
########################################
PROJ_PATH=~/projects/def-jcohen/totalsegmri
INPUT_FOLDER=$1
OUTPUT_FOLDER=${INPUT_FOLDER}_output
ENVDIR=${PROJ_PATH}/venv

export nnUNet_def_n_proc=16
export nnUNet_n_proc_DA=$nnUNet_def_n_proc

# Set nnunet params
export nnUNet_raw=${PROJ_PATH}/nnUNet_raw
export nnUNet_preprocessed=${PROJ_PATH}/nnUNet_preprocessed
export nnUNet_results=${PROJ_PATH}/nnUNet_results
CONFIGURATION="3d_fullres"
#######################################

echo ""
echo "Running with the following parameters:"
echo "INPUT_FOLDER=${INPUT_FOLDER}"
echo "OUTPUT_FOLDER=${OUTPUT_FOLDER}"
echo "PROJ_PATH=${PROJ_PATH}"
echo ""

# Create and activate virtual env
if [ ! -d $ENVDIR ]; then
    module load python/3.10
    virtualenv --no-download $ENVDIR
    source $ENVDIR/bin/activate
    pip install --no-index torch torchvision torchaudio nnunetv2 nilearn
else
    source $ENVDIR/bin/activate
fi

nnUNetv2_predict -d 206 -i ${INPUT_FOLDER} -o ${OUTPUT_FOLDER}/206 -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans -npp $nnUNet_def_n_proc -nps $nnUNet_def_n_proc
python $PROJ_PATH/generate_seg_jpg.py -i ${INPUT_FOLDER} -s ${OUTPUT_FOLDER}/206 -o ${OUTPUT_FOLDER}/206_jpgs --image-suffix _0000 -r 1
nnUNetv2_apply_postprocessing -i ${OUTPUT_FOLDER}/206 -o ${OUTPUT_FOLDER}/206_pp -pp_pkl_file ${PROJ_PATH}/nnUNet_results/Dataset206_TotalSegMRI/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np $nnUNet_def_n_proc -plans_json ${PROJ_PATH}/nnUNet_results/Dataset206_TotalSegMRI/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json
python $PROJ_PATH/generate_seg_jpg.py -i ${INPUT_FOLDER} -s ${OUTPUT_FOLDER}/206_pp -o ${OUTPUT_FOLDER}/206_pp_jpgs --image-suffix _0000 -r 1

python $PROJ_PATH/generate_labels_sequential.py -i ${OUTPUT_FOLDER}/206_pp -o ${OUTPUT_FOLDER}/210_input --output-seg-suffix _0001 --disc-labels 1 2 3 4 --init-disc 2,1 3,6 4,23 --output-disc-step 1 --combine-before-label
python $PROJ_PATH/map_labels.py -i ${OUTPUT_FOLDER}/210_input -o ${OUTPUT_FOLDER}/210_input -m $PROJ_PATH/nnunet_210_0001_labels_map.json --seg-suffix _0001 --output-seg-suffix _0001

for i in ${OUTPUT_FOLDER}/210_input/*; do
    cp ${INPUT_FOLDER}/$(basename ${i//0001.nii.gz/0000.nii.gz}) ${i//0001.nii.gz/0000.nii.gz}
done

python $PROJ_PATH/generate_seg_jpg.py -i ${OUTPUT_FOLDER}/210_input -s ${OUTPUT_FOLDER}/210_input -o ${OUTPUT_FOLDER}/210_input_jpgs --image-suffix _0000 --seg-suffix _0001 -r 1

nnUNetv2_predict -d 210 -i ${OUTPUT_FOLDER}/210_input -o ${OUTPUT_FOLDER}/210 -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans -npp $nnUNet_def_n_proc -nps $nnUNet_def_n_proc
python $PROJ_PATH/generate_seg_jpg.py -i ${INPUT_FOLDER} -s ${OUTPUT_FOLDER}/210 -o ${OUTPUT_FOLDER}/210_jpgs --image-suffix _0000 -r 1
nnUNetv2_apply_postprocessing -i ${OUTPUT_FOLDER}/210 -o ${OUTPUT_FOLDER}/210_pp -pp_pkl_file ${PROJ_PATH}/nnUNet_results/Dataset210_TotalSegMRI/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np $nnUNet_def_n_proc -plans_json ${PROJ_PATH}/nnUNet_results/Dataset210_TotalSegMRI/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json
python $PROJ_PATH/generate_seg_jpg.py -i ${INPUT_FOLDER} -s ${OUTPUT_FOLDER}/210_pp -o ${OUTPUT_FOLDER}/210_pp_jpgs --image-suffix _0000 -r 1

python $PROJ_PATH/generate_labels_sequential.py -i ${OUTPUT_FOLDER}/210_pp -o ${OUTPUT_FOLDER}/final  --disc-labels 2 3 4 5 6 --vertebrea-labels 8 9 10 11 12 --init-disc 4,224 5,219 6,202 --init-vertebrae 10,41 11,34 12,18
python $PROJ_PATH/map_labels.py -i ${OUTPUT_FOLDER}/206_pp -o ${OUTPUT_FOLDER}/final -m $PROJ_PATH/nnunet_206_210_canal_label_labels_map.json --seg-suffix "" --output-seg-suffix "" --add-output
python $PROJ_PATH/generate_seg_jpg.py -i ${INPUT_FOLDER} -s ${OUTPUT_FOLDER}/final -o ${OUTPUT_FOLDER}/final_jpgs --image-suffix _0000 -r 1
