# Robust Control of Markov Jump Linear Systems with Cluster Observations

This repository provides the Python implementation accompanying the paper:

> **Control of Markov Jump Linear Systems with Lumpable Cluster Observations**  
> Carlos A. F. Persiani, Ram Padmanabhan, Melkior Ornik, and Marco H. Terra.

The code implements the coupled robust Riccati recursions developed in the paper for the control of discrete-time Markov jump linear systems (MJLS) when the true Markov mode is not directly observed. Instead, the controller has access only to a cluster containing the current mode.

The proposed formulation accounts for uncertainty associated with both the active system dynamics and the transition probabilities between observed clusters.

## Main Implementation

The main implementation is provided in:

```text
coupled_robust_riccati_v3.py
```

The `CoupledRobustRiccati` class implements the two main control formulations presented in the paper.

### Theorem IV.3

The method

```python
solve_TheoremIV3()
```

implements the coupled robust Riccati recursion presented in Theorem IV.3.

The method considers finite penalty parameters and iteratively computes the coupled Riccati matrices and the corresponding cluster-dependent feedback gains.

### Corollary V.1

The method

```python
solve_CorollaryVI()
```

implements the limiting formulation presented in Corollary V.1, corresponding to the exact-dynamics limit used in the convergence and stability analysis.

The Riccati equations associated with different observed clusters are solved simultaneously because their solutions are coupled through the uncertain cluster transition probabilities.

## Requirements

The implementation requires:

- Python 3
- NumPy

Install the required dependency using:

```bash
pip install numpy
```

## Usage

A problem is defined by specifying:

- the lifted system matrices;
- the observed clusters;
- the state, control, and lifted-state weighting matrices;
- the nominal cluster transition probabilities;
- the transition probability uncertainty bounds;
- the nominal selection matrices;
- the structured uncertainty matrices;
- the regularization parameters.

A typical workflow is:

```python
solver = CoupledRobustRiccati(
    F=F,
    G=G,
    clusters=clusters,
    Q=Q,
    R=R,
    S=S,
    qbar=qbar,
    alpha=alpha,
    Cbar=Cbar,
    M_C=M_C,
    E_C=E_C,
    mu1=mu1,
    mu2=mu2,
    beta=beta,
)
```

To solve the finite-penalty formulation from Theorem IV.3:

```python
P, K = solver.solve_TheoremIV3()
```

To solve the limiting recursion from Corollary V.1:

```python
P, K = solver.solve_CorollaryVI()
```

The returned dictionaries contain the Riccati matrices and cluster-dependent feedback gains:

```python
P[l]   # Riccati matrix associated with cluster l
K[l]   # Feedback gain associated with cluster l
```

## Numerical Example

The example included in the code considers the fault-susceptible lateral-directional aircraft dynamics used in the accompanying paper.

The example demonstrates how to:

1. construct the mode-lifted system;
2. define the observed clusters;
3. specify uncertain cluster transition probabilities;
4. construct the structured mode-selection uncertainty;
5. solve the coupled robust Riccati equations; and
6. obtain a feedback gain for each observed cluster.

## Reference

If you use this implementation in academic work, please cite:

```bibtex
@inproceedings{persiani2027cluster,
  title  = {Control of Markov Jump Linear Systems with Lumpable Cluster Observations},
  author = {Persiani, Carlos A. F. and Padmanabhan, Ram and Ornik, Melkior and Terra, Marco H.},
  year   = {2027}
}
```

The citation information will be updated following publication.

## Notes

This repository contains research code associated with the theoretical developments of the accompanying paper. It is primarily intended to reproduce the proposed Riccati-based controller and the numerical examples presented in the manuscript.

For the mathematical derivation, assumptions, uncertainty formulation, convergence analysis, and stability conditions, please refer to the accompanying paper.
