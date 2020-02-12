# Table of Contents

-   [Overview](#overview)
-   [Acknowledgements](#acknowledgements)
-   [License](#license)


# Overview

This repository serves as an [open-science
notebook](http://en.wikipedia.org/wiki/Open_notebook_science) for research
conducted by the [Phyletica Lab](http://phyletica.org) to assess the
performance new model priors implemented in the software package,
[Ecoevolity](https://github.com/phyletica/ecoevolity).

[Ecoevolity](https://github.com/phyletica/ecoevolity)
is a software package for full-likelihood Bayesian comparative phylogeographic
analyses.
It compares the timing of evolutionary events across an arbitrary number of
taxa, or "comparisons," to infer whether such events were temporally clustered,
which might suggest a shared biogeographic process.
These evolutionary events can comprise the divergence between two populations
or species, or the change in the effective size of a population.

To do this, 
[ecoevolity](https://github.com/phyletica/ecoevolity)
uses Bayesian model choice to approximate the posterior probability of all
possible ways the taxa can partitioned into event time categories.
Each possible partitioning of the taxa is a unique model of evolutionary
events, or "event model."
To do this, we must specify the probability of each possible event model
*a priori*.

Initially, 
[ecoevolity](https://github.com/phyletica/ecoevolity)
used a Dirichlet process (DP) as a mathematically convenient and flexible way to
assign the prior probability of every possible event model.
We have since implemented a
Pitman-Yor process (PYP) prior, which is a generalization of the Dirichlet process.
We also implemented a prior that by default assigns equal prior probability to
all possible event models, which we'll refer to as a uniform prior.
However, this "uniform" prior has a "split weight" (SW) parameter to allow
classes of event models with more event time categories to be favored (or
disfavored) *a priori*.

The goal of this project is to compare the performance of these three
priors on event models (DP, PYP, and SW) using simulations.


# Acknowledgements

This work was made possible by funding provided to [Jamie
Oaks](http://phyletica.org) from the National Science Foundation (grant numbers
DBI 1308885 and DEB 1656004).


# License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/deed.en_US"><img alt="Creative Commons License" style="border-width:0" src="http://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/deed.en_US">Creative Commons Attribution 4.0 International License</a>.
