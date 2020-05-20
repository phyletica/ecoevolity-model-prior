FROM ubuntu:bionic

# Install dependencies for ecoevolity and pycoevolity
RUN apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
        git \
        cmake \
        g++ \
        curl \
        texlive-latex-extra \ 
        texlive-bibtex-extra

# Install R for pycoevolity plotting
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
        r-base

# Install R packages used by pycoevolity
RUN R -e 'install.packages(c("ggplot2", "ggridges"), repos = "http://cloud.r-project.org/")'

# Download, install, and setup miniconda
RUN curl -L -o "install-miniconda3.sh" "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
    bash install-miniconda3.sh -b -p "${HOME}/miniconda3"  && \
    rm install-miniconda3.sh && \
    ln -s "${HOME}/miniconda3/etc/profile.d/conda.sh" /etc/profile.d/conda.sh && \
    echo "source \"${HOME}/miniconda3/etc/profile.d/conda.sh\"" >> "${HOME}/.bashrc"

# Update conda
RUN "${HOME}/miniconda3/bin/conda" update -y conda

# Download the project repo
RUN git clone https://github.com/phyletica/ecoevolity-model-prior.git

# Move into project directory and run setup script
RUN cd ecoevolity-model-prior && \
    bash setup_project_env.sh && \
    "${HOME}/miniconda3/bin/conda" env create -f conda-environment.yml && \
    echo "conda activate ecoevolity-model-prior-project" >>  "${HOME}/.bashrc"
