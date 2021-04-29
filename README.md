# Combinatorial Generalization

These simulations test the ability to learn a pure combinatorial mapping, where N pools of input patterns map fully systematically to a corresponding set of output pools, but a fully connected hidden layer must learn this structure.

This capacity for "mix-and-match" combinatorics is essential for generativity and systematicity.

# O'Reilly, 2001

* O'Reilly, R.C. (2001). Generalization in interactive networks: The benefits of inhibitory competition and Hebbian learning. *Neural Computation, 13*, 1199-1242.

This paper demonstrates that bidirectionally-connected networks exhibit more "nonlinear" attractor-like behavior that significantly impairs combinatorial generalization.  It then shows that the Leabra model can overcome these impairments through the addition of hebbian learning and inhibitory competition.  However, in retrospect, it now seems that this was a bit of a special case (as it seems the reviewers had suspected).  When using random bit patterns instead of lines for each pool, the network does not generalize nearly as systematically.

