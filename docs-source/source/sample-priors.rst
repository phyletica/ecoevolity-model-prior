.. _sample-priors:

#########################
Sampling the model priors
#########################

.. important:: The most important part of the project is the contents of the
    |eco|_ configuration files in the ``ecoevolity-configs`` directory.  These
    files specify the models we will be using to simulate and analyze genomic
    datasets.

    Read the :ref:`background` and :ref:`eco-configs` sections to become
    familiar with these models.

As described in the
:ref:`background` and :ref:`eco-configs` sections,
we will be analyzing simulated data under three different priors
on divergence models:

1.  Dirichlet-process (DP) prior
2.  Pitman-Yor process (PYP) prior
3.  Uniform distribution with a split-weight parameter (SW)

The settings for these models are contained in three |eco|_ configuration files
in the ``ecoevolity-configs`` directory of the project. These
are, respectively:

1.  ``pairs-10-dpp-conc-2_0-2_71-time-1_0-0_05.yml``
2.  ``pairs-10-pyp-conc-2_0-1_79-disc-1_0-4_0-time-1_0-0_05.yml``
3.  ``pairs-10-unif-sw-0_55-7_32-time-1_0-0_05.yml``

Before we do anything else, we should validate that these modes are what we
think we are.
To do this, we will analyze them with ``ecoevolity``, but tell the program to
ignore the data.
What should we get if we perform a Bayesian analysis while ignoring the data
(i.e., the likelihood is always 1)?
The prior!

The config files define the model prior we wish to use.
By analyzing it while ignoring the data, the distribution that ``ecoevolity``
samples from should be the very one we specified in the configs.
Hence, this is a nice sanity check to make sure everything is working as we
expect.


Setup our environment
=====================

Before anything else, navigate to the project directory (if you are not already
there).
For me, this is::

    cd /scratch/jro0014/ecoevolity-model-prior

Your path will be different.
If you haven't already, let's activate the Python environment for this project::

    conda activate ecoevolity-model-prior-project

and (if you are working on this project on the AU Hopper cluster) source the
file that loads all the necessary modules::

    source source modules-to-load.sh


Run the analyses
================

Now, lets ``cd`` into the project's ``scripts`` directory::

    cd scripts

Next, we need to run the ``prior_sampling_tests.sh`` script.
If you are working on Auburn University's Hopper cluster, submit this script to
the cluster's queue by entering::

    ../bin/psub prior_sampling_tests.sh 2

.. note:: If you are not on the Hopper cluster, you can simply run this
    directly::

        bash prior_sampling_tests.sh 2

In case you're curious the "2" argument tells the script how many extra zeros to
add to the MCMC chain length specified in the config files.
The ``chain_length`` in the config files is 75,000 generations.
So, the "2" will extend this to 7,500,000.
Because, sampling from the prior is fast, we might as well collect a large
sample.

What this script will do, is for each of the three model configs listed above,
it will:

1.  Use ``simcoevolity`` to simulate one dataset from that config.
2.  Use ``ecoevolity`` to analyze that simulated dataset while ignoring the
    data.
3.  Run ``sumcoevolity`` to summarize the results.
4.  Run ``pyco-sumevents`` to plot the results.
5.  Run the custom plotting script ``scripts/plot_prior_samples.py`` to do some
    additional plotting of the results.

Why run ``simcoevolity``? I.e., why not just run ``ecoevolity`` straightaway?
Well the extra step helps to validate that the whole workflow of simulating
data and then analyzing them is working as we expect it should.
Since we will be doing this thousands of times for this project, it's a nice
extra step for our sanity check.


Checkout the results
====================

Once the ``prior_sampling_tests.sh`` script finishes, you should find all of
the output for each config file (model) in a ``prior-sampling-tests``
directory (assuming you are still in the ``scripts`` directory::

    ls ../prior-sampling-tests

should reveal::

    pairs-10-dpp-conc-2_0-2_71-time-1_0-0_05
    pairs-10-pyp-conc-2_0-1_79-disc-1_0-4_0-time-1_0-0_05
    pairs-10-unif-sw-0_55-7_32-time-1_0-0_05

Inside of these you will find the dataset simulated by ``simcoevolity`` (10
data files; 1 for each pair of populations), the output of ``ecoevolity`` (the
.log files) and a bunch of plots (.pdf files).

Let's take a look at the plots from each model and make sure everything looks
as we expect.

DPP results
-----------

.. figure:: /images/prior-sampling-dpp-pycoevolity-nevents.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: DP number of events prior

    The expected and sampled prior probabilities of the number of divergence
    events for the DP model.


.. figure:: /images/prior-sampling-dpp-prior-concentration.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: DP concentration prior

    The expected (orange line) and sampled (histogram) prior distribution on
    the concentration parameter of the DP model.


.. figure:: /images/prior-sampling-dpp-prior-event_time.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: DP event-time prior

    The expected (orange line) and sampled (histogram) prior distribution on
    divergence times for the DP model.


.. figure:: /images/prior-sampling-dpp-prior-leaf_population_size.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: DP descendant pop size prior

    The expected (orange line) and sampled (histogram) prior distribution on
    the effective size of the descendant populations for the DP model.


.. figure:: /images/prior-sampling-dpp-prior-root_relative_population_size.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: DP relative ancestral pop size prior

    The expected (orange line) and sampled (histogram) prior distribution on
    the relative effective size of the ancestral population for the DP model.


PYP results
-----------

.. figure:: /images/prior-sampling-pyp-pycoevolity-nevents.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: PYP number of events prior

    The expected and sampled prior probabilities of the number of divergence
    events for the PYP model.


.. figure:: /images/prior-sampling-pyp-prior-concentration.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: PYP concentration prior

    The expected (orange line) and sampled (histogram) prior distribution on
    the concentration parameter of the PYP model.


.. figure:: /images/prior-sampling-pyp-prior-discount.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: PYP discount prior

    The expected (orange line) and sampled (histogram) prior distribution on
    the discount parameter of the PYP model.


.. figure:: /images/prior-sampling-pyp-prior-event_time.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: PYP event-time prior

    The expected (orange line) and sampled (histogram) prior distribution on
    divergence times for the PYP model.


.. figure:: /images/prior-sampling-pyp-prior-leaf_population_size.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: PYP descendant pop size prior

    The expected (orange line) and sampled (histogram) prior distribution on
    the effective size of the descendant populations for the PYP model.


.. figure:: /images/prior-sampling-pyp-prior-root_relative_population_size.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: PYP relative ancestral pop size prior

    The expected (orange line) and sampled (histogram) prior distribution on
    the relative effective size of the ancestral population for the PYP model.


SW results
----------

.. figure:: /images/prior-sampling-unif-pycoevolity-nevents.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: SW number of events prior

    The expected and sampled prior probabilities of the number of divergence
    events for the SW model.


.. figure:: /images/prior-sampling-unif-prior-split_weight.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: SW split-weight prior

    The expected (orange line) and sampled (histogram) prior distribution on
    the split-weight parameter of the SW model.


.. figure:: /images/prior-sampling-unif-prior-event_time.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: SW event-time prior

    The expected (orange line) and sampled (histogram) prior distribution on
    divergence times for the SW model.


.. figure:: /images/prior-sampling-unif-prior-leaf_population_size.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: SW descendant pop size prior

    The expected (orange line) and sampled (histogram) prior distribution on
    the effective size of the descendant populations for the SW model.


.. figure:: /images/prior-sampling-unif-prior-root_relative_population_size.png
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: SW relative ancestral pop size prior

    The expected (orange line) and sampled (histogram) prior distribution on
    the relative effective size of the ancestral population for the SW model.


Cleanup
=======

Given how quickly we can generate these results, and the fact that we should
regenerate them any time we change/add models to the project, there is no need
to add these results to the repository or keep them around.
So, once you are done checking out the results, go ahead and remove all of
the output::

    rm -r ../prior-sampling-tests
