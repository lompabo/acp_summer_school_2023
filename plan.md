---
title: Notes about the lecture
author: Michele Lombardi <michele.lombardi2@unibo.it>
---

* DFL recap (just to establish notation)
* Advantages and drawbacks
  - A more accurate model can get close
  - Training time
* A few possible fixes:
  - Weight initialization
  - Cache
* Still the impact of the approach looks somewhat diminished
  - ...But is that all we can do with it?
* One stage stochastic programming
  - Stochastic process
  - This is actually a one-stage stochastic program
  - Classical approach
  - DFL, from this perspective
  - Main advantage: limited need to make the correct assumptions on the distribution
  * Can we push it further?
* Two-stage stochastic programming
  - Classical formulation
  - The corresponding two-stage approach
  - Can we tackle that with DFL?
  - Possible formulation and main issue
  - Back to loss function design
  - SPO as a form of smoothing
  - Stochastic smoothing
  - SFGE Formulation
* Dealing with complex functions or constraints
  - Variant with a complex linear function
  - BB optimization could be used, but support for constraints is limited
  - Replace with a simple function
  - Still use the true objective for evaluation
* Sequential decision making
  - Classical formulation
  - Link between SFGE and REINFORCE
  - Classical RL formulation
  - Policy factorization (UNIFY)
  - Mapping between the two
  - Example/advantage: support for constraints in RL
  - Example/advantage: scalable multi-stage stochastic programming
  - Why only REINFORCE? Multiple RL methods can be used
  - Adjusting the optimization problem formulation to reach a good trade-off








