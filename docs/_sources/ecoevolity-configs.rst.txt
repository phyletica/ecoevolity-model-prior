.. _eco-configs:

######################
The ecoevolity configs
######################

The |eco|_ configuration files in the ``ecoevolity-configs`` directory are core
to this project.
They specify the models that we will be using to simulate and analyze
genome-scale datasets.

By design, all of these configs are identical except for the prior on the event
(divergence) model.
Remember, the goal of the project is to compare different ways of modeling
shared divergence events.

DP model::

    bin/dpprobs -n 1000000 --shape 2.0 --scale 2.71 1 10
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
    
PYP model::

    bin/dpprobs -n 1000000 --shape 2.0 --scale 1.79 --discount-alpha 1.0 --discount-beta 4.0 1 10
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
    
Uniform model::

    bin/swprobs -n 1000000 --shape 0.55 --scale 7.32 1 10
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
