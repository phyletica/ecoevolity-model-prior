.. _background:

##########
Background
##########


.. _prior_on_divergence_models:

Prior on divergence models
==========================

|Eco| treats the divergence model (number of divergence events, and the
assignment of the taxa to the events) as a random variable under a Dirichlet
process :cite:`Ferguson1973,Antoniak1974`.
The basic idea of the Dirichlet process is quite simple; we assign species
pairs to divergence events one at a time following a very simple rule.
When assigning the :math:`n^{th}` pair, we assign it to its own
event (i.e., a new divergence event) with probability

.. math::
    :label: dppnewcat

    \frac{\alpha}{\alpha + n -1}

or you assign it to an existing event :math:`x` with probability

.. math::
    :label: dppexistingcat

    \frac{n_x}{\alpha + n -1}

where :math:`n_x` is the number of pairs already assigned to
event :math:`x`.
Let's walk through an example using our three pairs of lizard species.
First, we have to assign our first pair ("A") to a
divergence event with probability 1.0;
let's call this the "blue" divergence event.
Next we assign the second pair ("B") to either a new ("red") divergence
event with probability :math:`\alpha/\alpha + 1` or to the same "blue"
divergence event as the first pair with probability :math:`1/\alpha + 1`.
For this example, let's say it gets assigned to the "blue" event.
Lastly, we assign the third pair ("C") to either a new ("red") divergence
event with probability :math:`\alpha/\alpha + 2` or to the same "blue"
divergence event as the first two pairs with probability :math:`2/\alpha +
2`.

The GIF below illustrates how the these simple rules determine the
prior probability of all five possible models.
Notice toward the end of the animation, as the concentration parameter
increases we place more probability on the divergence models with more
independent divergence events (less shared divergences).

.. _dpp_tree:

.. figure:: /images/dpp-3-example.gif
    :align: center
    :width: 600 px
    :figwidth: 90 %
    :alt: DPP example

    An example of the Dirichlet process.

Notice that the Dirichlet process prior (DPP) is not motivated by any
biological processes.
Rather, we use it because it is flexible (we can adjust or estimate the
concentration parameter), and mathematically convenient; it allows us to use
Gibbs sampling :cite:`Neal2000` to sample across divergence models.
