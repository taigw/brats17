#$ -P gpu
#$ -l gpu=1
#$ -l gpu_pascal=1
#$ -l h_rt=10:0:0
#$ -l tmem=10.0G
##$ -l h=tesla5
#$ -N test_all
#$ -S /bin/bash
#$ -wd /home/guotwang/tf_project/brats17
#!/bin/bash
# The lines above are resource requests. This script has requested 1 Titan X (Pascal) GPU for 24 hours, and 11.5 GB of memory to be started with the BASH Shell.
# More information about resource requests can be found at http://hpc.cs.ucl.ac.uk/job_submission_sge/

# This line ensures that you only use the 1 GPU requested.
nvidia-smi
export CUDA_VISIBLE_DEVICES=$(( `nvidia-smi | grep " / .....MiB"|grep -n " 0MiB / [0-9]....MiB"|cut -d : -f 1|head -n 1` - 1 ))
echo $CUDA_VISIBLE_DEVICES
if [ $CUDA_VISIBLE_DEVICES -lt 0 ];then
exit 1
fi
# These lines runs your NiftyNet task with the correct library paths for tensorflow
TF_LD_LIBRARY_PATH=/share/apps/libc6_2.17/lib/x86_64-linux-gnu/:/share/apps/libc6_2.17/usr/lib64/:/share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib:/share/apps/python-3.6.0-shared/lib:/share/apps/cuda-8.0/lib64:/share/apps/cuda-8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PYTHONPATH="${PYTHONPATH}:/home/guotwang/tf_project/niftynet_src/NiftyNet"
/share/apps/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so --library-path $TF_LD_LIBRARY_PATH $(command -v /home/guotwang/miniconda3/bin/python3) /home/guotwang/tf_project/brats17/test.py  /home/guotwang/tf_project/brats17/config17/test_all_class.txt
