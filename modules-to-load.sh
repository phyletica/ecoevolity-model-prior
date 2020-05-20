# Needed for compiling ecoevolity. Also needed to subsequently run ecoevolity,
# because gcc is dynamically linked
module load gcc/5.3.0
module load cmake/3.7.1

# Needed for Python plotting scripts, because matplotlib is set to use LaTeX to
# generate in-plot text
module load perl/5.26.1
module load glibc/2.14
module load texlive/2019
