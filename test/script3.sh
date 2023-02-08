command="python /opt/totalsegmentator-mri/scripts/combine_masks.py -d /opt/TotalSegmentatorMRI_SynthSeg/Totalsegmentator_dataset -o /opt/TotalSegmentatorMRI_SynthSeg/output/TotalSegmentator_Masks_Combined -m /opt/totalsegmentator-mri/resources/labels.json --subject-prefix s --subject-subdir segmentations --seg-suffix _ct_seg"

#run on docker

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v "$PWD":/mnt -v /mounts/auto:/mounts/auto:shared -v /data:/data:shared -v /mounts/auto/arman7/totalSegmentatorFolder/Totalsegmentator_dataset/:/Totalsegmentator_dataset/  -v /mounts/auto/ipmsa-results:/data/ipmsa-results:shared -v /mounts/auto/arman7/totalSegmentatorFolder/TotalSegmentatorMRI_SynthSeg:/opt/TotalSegmentatorMRI_SynthSeg -u $(id -u ${USER}):$(id -g ${USER}) --network=host --workdir /mnt --rm -ti armaneshaghi/totalsegmentator-mri:latest ${command} 

