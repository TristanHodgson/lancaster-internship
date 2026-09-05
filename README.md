# Lancaster University STOR-i Summer Research Project

This repository contains code relating to a project investigating the structure of policies for redundant systems. You can find a research [poster](write-up/poster.pdf) and [presentation](write-up/presentation.pdf) in the `write-up` folder. They contain a formal description of the mode, the proved results, and the conjectures we tested numerically.

## To Run

1. Clone the repository
2. Add a Gurobi key to the `secret/gurobi.lic` file (if you want to use the linear programing solver)
3. Create a Python virtual environment and install the dependencies in `requirements.txt`
4. Run the `main.py` file. You can change settings as well as uncomment sections to run different experiments.

## What is Here?

| File                          | What it does                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `main.py`                     | Runs all experiments                                                                                    |
| `modules/mdp.py`              | Builds the model: the state space, the available actions, the costs and the transitions between states. |
| `modules/policy_iteration.py` | Policy iteration, for both the discounted and the long-run average case.                                |
| `modules/value_iteration.py`  | Value iteration,, for both the discounted and the long-run average case.                                |
| `modules/lp.py`               | Linear program using Gurobi, with an optional constraint forcing a minimum uptime.                      |
| `modules/algorithm.py`        | A brute force search over the reduced policy space suggested by the proofs and conjectures.             |
| `modules/conjectures.py`      | Conjecture testing                                                                                      |
| `modules/helper.py`           | Helper functions for graphs and evaluation                                                              |
| `modules/utils.py`            | Shared functions for calculating common quantities                                                      |


## Features

- **Multiple component types** Failure rates, repair rates, repair costs and the number of components needed to stay online can all be set per type.
- **Optional repair cancellation** Repairs already in progress can be called off, which is the extended model discussed in the write-up.
- **Both optimality criteria** Every solver handles discounted and long-run average reward, so the two can be compared on the same problem.
- **Four independent solvers** Policy iteration, value iteration, linear programming and brute force all solve the same model, which makes it cheap to check that a structural pattern is not an artefact of numerical instability in one algorithm.
- **Uptime tooling** We can constrain uptime in the linear program or binary search for it in other models.
- **Conjecture testing** Each conjecture has a test that sweeps a grid of problem instances and reports every state where it fails. The most recent run found no violations; see `conjecture-report.md`.
- **Figures** Policy heat-maps with transient states blacked out and randomised states annotated, a grid of heat-maps for the two-type model, uptime against the downtime penalty, and solver timings.

## Citations

* Fairley, Luke. ‘A Dynamic Approach to Optimal Maintenance of Critical Network Infrastructure’. Master’s thesis, Lancaster University, 2022.
* Fairley, Luke. Bi-Objective Strategic and Operational Decision-Making in Redundancy Allocation Problems with Dynamic Maintenance. 2026, 2147208 B, 326 pages. Application/pdf, 2147208 B, 326 pages. https://doi.org/10.17635/LANCASTER/THESIS/3148.
* Puterman, Martin L. Markov Decision Processes: Discrete Stochastic Dynamic Programming. Wiley Online Library. Wiley-Interscience, 2005. https://doi.org/10.1002/9780470316887.
* Serfozo, Richard F. ‘Technical Note—An Equivalence Between Continuous and Discrete Time Markov Decision Processes’. Operations Research 27, no. 3 (1979): 616–20. https://doi.org/10.1287/opre.27.3.616.
* Sutton, Richard S., and Andrew Barto. Reinforcement Learning: An Introduction. Second edition. Adaptive Computation and Machine Learning. The MIT Press, 2020.
