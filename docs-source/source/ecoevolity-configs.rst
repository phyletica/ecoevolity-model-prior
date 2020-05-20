.. _eco-configs:

######################
The ecoevolity configs
######################

.. note:: For a detailed treatment of |eco|_ config files, please see
    `the config-file section of the ecoevolity documentation <http://phyletica.org/ecoevolity/yaml-config.html>`_.

The |eco|_ configuration files in the ``ecoevolity-configs`` directory are core
to this project.
They specify the models that we will be using to simulate and analyze
genome-scale datasets.

By design, all of these configs are identical except for the prior on the
divergence events (``event_model_prior``).
Remember, the goal of the project is to compare different ways of modeling
shared divergence events.
Please see the :ref:`background` section for some background information about
shared divergence events and the three different ways modelling them that we
will be comparing for this project.

Before we dig into the different models of divergence events, let's first
look at some of the features that all of these configs have in common.

1.  In the ``comparisons`` section, they all point to the same datasets for 10
    pairs of populations.
    Each of these datasets comprises 500,000 biallelic characters from 4
    genomes (2 diploid individuals) sampled from 2 populations.
    These are "dummy" datasets that live in the ``data`` directory of the
    project.
    I call them "dummy," because they are intended to serve as a template from
    which |simco| will simulate datasets of the same size.

2.  The prior on divergence times (``event_time_prior``) is an exponential
    distribution with mean of 0.05 (i.e., a mean of 5% divergent).

Let's take a look at the ``event_model_prior`` settings in each of the
configuration files.

.. _dp_config:

``pairs-10-dpp-conc-2_0-2_71-time-1_0-0_05.yml``
------------------------------------------------

This config specifies a Dirichlet-process (DP) prior over all the ways that the
10 pairs of populations can share divergence times.
See the :ref:`dp-prior-on-divergence-models` section for some background
about the DP.

The config file specifies a gamma distribution on the prior for the
concentration parameter of the DP, with a shape of 2 and a scale of 2.71.
So, rather than assume a specific value of the concentration parameter of the
DP, we are putting a prior on it (a hyperprior) to integrate over uncertainty
about the value of this parameter of the DP prior.

We can use the ``dpprobs`` tool of |eco|_ to get a sense of the prior this
creates over the myriad ways that the 10 pairs of populations can share (or not
share) divergence times.

From the base directory of the project, enter the following command::

    bin/dpprobs -n 1000000 --shape 2.0 --scale 2.71 1 10

Assuming the |eco|_ tools were successfully installed to the projects ``bin``
directory when we setup the project (see the :ref:`setup-project` section), you
should see output similar to::

    ======================================================================
                                   DPprobs
                  Simulating Dirichlet process probabilities
    
                                   Part of:
                                  Ecoevolity
          Version 0.3.2 (testing 932c358: 2020-02-17T15:27:08-06:00)
    ======================================================================
    
    Process = Dirichlet
    Seed = 1020136828
    Number of samples = 1000000
    Number of elements = 10
    Concentration ~ gamma(shape = 2, scale = 2.71)
    Discount = 0
    Mean number of categories = 6.00467
    Sample mean number of categories = 5.50189
    
    Estimated probabilities of the number of categories:
    ----------------------------------------------------------------------
    p(ncats =  1) = 0.020063     (n = 1)
    p(ncats =  2) = 0.053345     (n = 511)
    p(ncats =  3) = 0.096057     (n = 9330)
    p(ncats =  4) = 0.142066     (n = 34105)
    p(ncats =  5) = 0.17664      (n = 42525)
    p(ncats =  6) = 0.185396     (n = 22827)
    p(ncats =  7) = 0.16003      (n = 5880)
    p(ncats =  8) = 0.106323     (n = 750)
    p(ncats =  9) = 0.048465     (n = 45)
    p(ncats = 10) = 0.011615     (n = 1)
    ----------------------------------------------------------------------
    
    Runtime: 0 seconds.

This shows us that the prior mean number of divergence events across the 10
pairs of populations is approximately 5.5.
Under "Estimated probabilities of the number of categories", it shows us the
approximate prior probabilities of there being
:math:`1, 2, 3, \ldots, 10`
divergence events among the pairs.
To the right of those probabilities, it also shows the number of different ways
the 10 pairs of populations can share each number of divergence events (e.g.,
there are 511 different ways we can assign the 10 pairs to 2 events).
So, there are many more possible divergence models for intermediate numbers of
events.

While this DP prior is favoring 6 divergence events *a priori*, it's important
to keep in mind that it is *not* favoring any one of the divergence models with
6 events.
The prior probability of 6 events is approximately 0.185, but this is
distributed over 22,827 different models with 6 divergence events (i.e., 6
divergence-time parameters).


.. _pyp_config:

``pairs-10-pyp-conc-2_0-1_79-disc-1_0-4_0-time-1_0-0_05.yml``
-------------------------------------------------------------
    
This config specifies a Pitman-Yor process (PYP) prior over all the ways that
the 10 pairs of populations can share divergence times.
See the :ref:`pyp-prior-on-divergence-models` section for some background about
the PYP.

The config file specifies a gamma distribution on the prior for the
concentration parameter of the PYP, with a shape of 2 and a scale of 1.79.
For the prior on the discount parameter of the PYP it specifies a beta
distribution with an alpha of 1 and beta of 4.
Again, rather than assume values of these parameters we are putting hyperpriors
on them to integrate over uncertainty.

As for the DP, we can use the ``dpprobs`` tool of |eco|_ to get a sense of the
prior this PYP creates over all the ways that the 10 pairs of populations can
share (or not share) divergence times.

From the base directory of the project, enter the following command::

    bin/dpprobs -n 1000000 --shape 2.0 --scale 1.79 --discount-alpha 1.0 --discount-beta 4.0 1 10

You should see output similar to::

    ======================================================================
                                   DPprobs
                  Simulating Dirichlet process probabilities
    
                                   Part of:
                                  Ecoevolity
          Version 0.3.2 (testing 932c358: 2020-02-17T15:27:08-06:00)
    ======================================================================
    
    Process = Pitman-Yor
    Seed = 1068578483
    Number of samples = 1000000
    Number of elements = 10
    Concentration ~ gamma(shape = 2, scale = 1.79)
    Discount ~ beta(1, 4)
    Mean number of categories = 5.87019
    Sample mean number of categories = 5.49826
    
    Estimated probabilities of the number of categories:
    ----------------------------------------------------------------------
    p(ncats =  1) = 0.023656     (n = 1)
    p(ncats =  2) = 0.058365     (n = 511)
    p(ncats =  3) = 0.101317     (n = 9330)
    p(ncats =  4) = 0.142269     (n = 34105)
    p(ncats =  5) = 0.170535     (n = 42525)
    p(ncats =  6) = 0.172826     (n = 22827)
    p(ncats =  7) = 0.149305     (n = 5880)
    p(ncats =  8) = 0.105087     (n = 750)
    p(ncats =  9) = 0.057015     (n = 45)
    p(ncats = 10) = 0.019625     (n = 1)
    ----------------------------------------------------------------------
    
    Runtime: 1 seconds.
    
By comparing the outputs for the DP and PYP, you will notice that the prior
probability distributions over the number of divergence events is quite
similar.
For example, they both have a prior mean of 5.5.
The hyperprior settings for the DP and PYP models were specifically chosen to
make them very similar.
Remember, our goal is to compare the performance of these models in estimating
shared divergence events.
If we compared DP and PYP models that placed wildly different priors over
divergence models, it would be difficult to interpret any differences in their
performance.


.. _uniform_config:

``pairs-10-unif-sw-0_55-7_32-time-1_0-0_05.yml``
------------------------------------------------

This config specifies a uniform(-ish) prior with a "split weight" parameter
over all the ways that the 10 pairs of populations can share divergence times.
We will use SW to abbreviate this distribution.
See the :ref:`uniform-prior-on-divergence-models` section for some background
about the SW.

The config file specifies a gamma distribution on the prior for the
split-weight parameter of uniform distribution, with a shape of 0.55 and a
scale of 7.32.
Once again, rather than assume a specific value, we are using a hyperprior to
integrate over uncertainty.

We can use the ``swprobs`` tool of |eco|_ to get a sense of the prior this SW
prior creates over all the ways that the 10 pairs of populations can share (or
not share) divergence times.

From the base directory of the project, enter the following command::

    bin/swprobs -n 1000000 --shape 0.55 --scale 7.32 1 10

This should produce output similar to::

    ======================================================================
                                   swprobs
         Calculating probabilities under the split-weight model prior
    
                                   Part of:
                                  Ecoevolity
          Version 0.3.2 (testing 932c358: 2020-02-17T15:27:08-06:00)
    ======================================================================
    
    Prior = split-weighted uniform
    Number of elements = 10
    Split weight ~ gamma(shape = 0.55, scale = 7.32)
    Seed = 2080907633
    Number of samples = 1000000
    Mean number of categories = 5.50017
    
    Approximated probabilities of the number of categories:
    ----------------------------------------------------------------------
    p(ncats =  1) = 0.0172824    (n = 1)
    p(ncats =  2) = 0.06953      (n = 511)
    p(ncats =  3) = 0.108496     (n = 9330)
    p(ncats =  4) = 0.138439     (n = 34105)
    p(ncats =  5) = 0.159231     (n = 42525)
    p(ncats =  6) = 0.165719     (n = 22827)
    p(ncats =  7) = 0.151055     (n = 5880)
    p(ncats =  8) = 0.112674     (n = 750)
    p(ncats =  9) = 0.0604041    (n = 45)
    p(ncats = 10) = 0.0171697    (n = 1)
    ----------------------------------------------------------------------
    
    Runtime: 1 seconds.
    
This distribution also has a prior mean number of divergence events of
approximately 5.5.
By comparing this to the outputs for the DP and PYP, you will notice that the
prior probability distributions over the number of divergence events is quite
similar across all three.
Once again, this similarity is "by design," because our goal is to compare
the utility of these different models.

The three models (DP, PYP, and SW) specified in the configs described above are
the ones that we will be using to analyze simulated datasets.
We are also interested in knowing how these three models perform when they are
all wrong.
To get at that, we have 2 more config files that we will use only to simulate
datasets.
These are described next.


.. _independent_config:

``fixed-pairs-10-independent-time-1_0-0_05.yml``
------------------------------------------------

A major goal of this project is to compare the performance of the DP, PYP, and
SW models for inferring shared divergence events.
When biogeographers use these sorts of methods, they are often interested in
testing for synchronous divergences.
So, it is very important to know how these models behave when there are *no*
shared divergences; all the taxa diverged independently.
When all divergences are random, we don't want our models to infer false
positives of shared events.

This config is the same in all other respects to the DP, PYP, and SW configs,
but it specifies that all 10 pairs of populations diverged independently (i.e.,
there were 10 divergence events).
We will use this config to simulate datasets for which we know there were no
shared divergence events.
We will analyze these data under the DP, PYP, and SW models to see which is
least likely to infer false positives.


.. _shared_config:

``fixed-pairs-10-simultaneous-time-1_0-0_05.yml``
-------------------------------------------------

This config is the same in all other respects to the DP, PYP, and SW configs,
but it specifies that all 10 pairs of populations diverged simultaneously
(i.e., there were 1 shared divergence events).
We will use this config to simulate datasets for which we know all 10
taxa co-diverged.
We will analyze these data under the DP, PYP, and SW models to see which is
most likely to correctly infer a single event.
