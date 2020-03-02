module load gcc/5.3.0
module load cmake/3.7.1
if [ -n "$(command -v conda)" ]
then
    conda activate ecoevolity-model-prior-project
fi
