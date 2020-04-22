.. _getting-started:

###############
Getting started
###############

If you haven't already, you should familiarize yourself with the
:ref:`prerequisites` and :ref:`background` sections of documentation.
If you are a |phyleticalab|_ member, make sure you have done everything in the
:ref:`setting-up` section. 


.. _clone-project:

Cloning the project repo
========================

Open up a terminal and navigate to where you want to work on this project.
For |phyleticalab|_, this means logging in to Hopper::

    ssh YOUR-AU-USERNAME@hopper.auburn.edu

or::

    ssh hopper

and navigating to your scratch directory::

    cd /scratch/YOUR-AU-USERNAME

Now, clone the |git|_ repository for this project::

    git clone git@github.com:phyletica/ecoevolity-model-prior.git

If this command is successful, when you::

    ls
    
you should see a directory called ``ecoevolity-model-prior`` listed.
Go ahead and ``cd`` into this directory::

    cd ecoevolity-model-prior

To get oriented to the contents of the project repository, please checkout the
:ref:`tour` section of the documentation.


.. _setup-project:

Setting up the project
======================

The first thing we need to do is run the ``setup_project_env.sh`` Bash script
which is located at the base of the project directory.
This script will download and build a specific version of |eco|_ and will
install all of the |eco|_ tools in the ``bin`` directory within our project
directory. It will also create the ``conda`` environment for the project::

    bash setup_project_env.sh

.. note:: On AU's Hopper cluster the script sometimes hangs at this point in
    setting up the conda environment::

        Collecting package metadata (repodata.json): \

    If this happens, you can safely kill the script (Control + C).
    |Eco|_ should already be successfully installed, so killing the script
    during the conda setup will not affect any other aspect of the setup
    script.
    Now, you can setup the conda environment manually::

        conda env create -f conda-environment.yml

After running the ``setup_project_env.sh`` script, activate the
new Python environment using ``conda``::

    conda activate ecoevolity-model-prior-project

and (if you are working on this project on the AU Hopper cluster) source the
file that loads all the necessary modules::

    source modules-to-load.sh

You will want to get in the habit of running these ``conda activate
ecoevolity-model-prior-project`` and ``source modules-to-load.sh`` (the latter
only if you are on AU's Hopper cluster) whenever you return to work on this
project.

If the setup proecess was successful, you should be able to call up the help menu
of ``ecoevolity`` by entering (from the base directory of the project)::

    bin/ecoevolity -h

This should display the help menu that begins with something like::

   ======================================================================
                                 Ecoevolity
                     Estimating evolutionary coevality
         Version 0.3.2 (testing 932c358: 2020-02-17T15:27:08-06:00)
   ======================================================================
   
   Usage: ecoevolity [OPTIONS] YAML-CONFIG-FILE
   
   Ecoevolity: Estimating evolutionary coevality
   
   Options:
     --version             show program's version number and exit
     -h, --help            show this help message and exit
    
Also, the |pyco|_ tools should be available in your path. You can confirm this
by trying::

    pyco-sumtimes -h

which should display the help menu of the ``pyco-sumtimes`` tool, the beginning of which should look something like::

    ========================================================================
                                  Pycoevolity                               
                       Summarizing evolutionary coevality                   
    
            A Python package for analyzing the output of Ecoevolity         
    
                        Version 0.2.4 (None None: None)                     
    
                 License: GNU General Public License Version 3              
    ========================================================================
    usage: pyco-sumtimes [-h] [-b BURNIN] [-p PREFIX] [-f]
                         [-l COMPARISON-LABEL REPLACEMENT-LABEL]
                         [-i COMPARISON-LABEL] [--violin] [--include-map-model]
                         [-z] [--x-limits LOWER-LIMIT UPPER-LIMIT] [-x X_LABEL]
                         [-y Y_LABEL] [-w WIDTH] [--base-font-size BASE_FONT_SIZE]
                         [--colors [COLORS [COLORS ...]]] [--no-plot]
                         ECOEVOLITY-STATE-LOG-PATH [ECOEVOLITY-STATE-LOG-PATH ...]
    
Once everything is setup, we have no need for the ``ecoevolity`` directory that
was cloned by the ``setup_project_env.sh`` script, so you can go ahead and
remove it::

    rm -rf ../ecoevolity-model-prior/ecoevolity

.. note:: The extra long path in the above command is to help ensure you are
    where you think you are on the file system and don't blow away anything you
    didn't intend to.

Congrats! You are all set and ready to begin working on the project.

Why all the trouble?
--------------------

If the setup process seemed onerous, you might be wondering, "why all the
trouble?" Well, the goal is to maximize transparency and reproducibility.
Everyone reaching this point should have the *exact* same version of |eco|_
installed to simulate datasets and analyze them, and a very similar Python
environment for running the ancillary scripts to parse, summarize, and plot the
results of these analyses.
This helps ensure that all of the details of the project are open, clear, and
can be repeated.


Next, let's go to the :ref:`sample-priors` section to get started
with anayses!
