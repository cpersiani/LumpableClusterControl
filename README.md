# Coupled Robust Riccati Solver for Cluster-Observed Markov Jump Linear Systems

Python implementation of a coupled robust Riccati recursion for discrete-time Markov jump linear systems with uncertain cluster observations.

The implementation considers the case in which the true Markov mode is not directly available to the controller. Instead, the controller observes a cluster containing multiple possible modes. Uncertainty associated with both the system selection within each cluster and the cluster transition probabilities is incorporated into the Riccati recursion.

The solver computes cluster-dependent stationary Riccati matrices and feedback gains and provides a numerical verification of sufficient stability conditions for the resulting closed-loop system.

## Features

- Coupled Riccati iteration over observable Markov clusters
- Cluster-dependent state-feedback gains
- Structured uncertainty in the system-selection matrices
- Uncertainty bounds for cluster transition probabilities
- Automatic construction of the robust auxiliary matrices
- Convergence monitoring for the coupled Riccati recursion
- Numerical verification of sufficient stability conditions
- Stability margins reported independently for each cluster

## Requirements

The implementation requires Python 3 and NumPy.

```bash
pip install numpy
```

## Usage

Import the solver:

```python
import numpy as np
from coupled_robust_riccati_v4 import CoupledRobustRiccati
```

Consider a simple problem with two observable clusters:

```python
clusters = ["A", "B"]

F = {
    "A": np.array([
        [1.0, 0.1],
        [0.0, 1.0]
    ]),
    "B": np.array([
        [1.0, 0.1],
        [0.0, 0.9]
    ]),
}

G = {
    "A": np.array([
        [0.0],
        [1.0]
    ]),
    "B": np.array([
        [0.0],
        [1.0]
    ]),
}

Q = {
    "A": np.eye(2),
    "B": np.eye(2),
}

R = {
    "A": np.eye(1),
    "B": np.eye(1),
}
```

Define the cluster-dependent system-selection matrices and their structured uncertainty:

```python
Cbar = {
    "A": np.eye(2),
    "B": np.eye(2),
}

M_C = {
    "A": np.eye(2),
    "B": np.eye(2),
}

E_C = {
    "A": 0.1 * np.eye(2),
    "B": 0.05 * np.eye(2),
}

S = {
    "A": 10.0 * np.eye(2),
    "B": 10.0 * np.eye(2),
}
```

Define the nominal cluster transition probabilities and their uncertainty bounds:

```python
qbar = {
    "A": {
        "A": 0.90,
        "B": 0.10,
    },
    "B": {
        "A": 0.20,
        "B": 0.80,
    },
}

alpha = {
    "A": {
        "A": 0.05,
        "B": 0.05,
    },
    "B": {
        "A": 0.05,
        "B": 0.05,
    },
}
```

Create the solver:

```python
solver = CoupledRobustRiccati(
    F=F,
    G=G,
    Q=Q,
    R=R,
    S=S,
    Cbar=Cbar,
    M_C=M_C,
    E_C=E_C,
    qbar=qbar,
    alpha=alpha,
    mu2=10.0,
    beta=1.01,
    tol=1e-10,
    max_iter=10_000,
)
```

Solve the coupled robust Riccati equations:

```python
result = solver.solve_theorem(verbose=True)

print("Converged:", result.converged)
print("Iterations:", result.iterations)
print("Final error:", result.error)

for l in clusters:
    print(f"\nP[{l}] =")
    print(result.P[l])

    print(f"\nK[{l}] =")
    print(result.K[l])
```

The resulting controller is cluster dependent. At each time step, the feedback gain associated with the currently observed cluster is applied.

## Stability Check

After convergence of the Riccati recursion, the sufficient stability conditions can be checked numerically:

```python
stability = solver.check_stability_sufficient_conditions(result)

print(
    "All sufficient stability conditions satisfied:",
    stability.all_satisfied
)

for l in clusters:
    print(f"\nCluster {l}")
    print("Lambda condition:", stability.condition_lambda[l])
    print("Lambda margin:", stability.lambda_margin[l])
    print("S condition:", stability.condition_S[l])
    print("S minimum-eigenvalue margin:", stability.S_margin_min_eig[l])
    print("Conditions satisfied:", stability.satisfied[l])
```

A positive `lambda_margin` indicates that the corresponding scalar robustness condition is satisfied.

A nonnegative `S_margin_min_eig` indicates that the corresponding matrix inequality is positive semidefinite, up to the numerical tolerance used by the implementation.

The flag

```python
stability.all_satisfied
```

is `True` only when the sufficient stability conditions are satisfied for every observable cluster.

## Returned Results

The Riccati solver returns a `RiccatiResult` object containing:

| Variable | Description |
| --- | --- |
| `P` | Cluster-dependent Riccati matrices |
| `K` | Cluster-dependent feedback gains |
| `Psi` | Coupled future-cost matrices |
| `W_C` | Robust auxiliary matrices |
| `Omega` | Effective matrices used in the Riccati recursion |
| `lam` | Cluster-dependent robustness parameters |
| `iterations` | Number of Riccati iterations |
| `converged` | Convergence flag |
| `error` | Final iteration error |

The stability routine returns:

| Variable | Description |
| --- | --- |
| `condition_lambda` | Result of the scalar stability condition for each cluster |
| `condition_S` | Result of the matrix stability condition for each cluster |
| `lambda_margin` | Numerical margin of the scalar condition |
| `S_margin_min_eig` | Minimum eigenvalue margin of the matrix condition |
| `satisfied` | Whether both conditions hold for each cluster |
| `all_satisfied` | Whether the sufficient conditions hold for every cluster |

## Notes

All cluster-dependent quantities are represented using Python dictionaries. The same cluster keys must therefore be used consistently across the system, cost, uncertainty, and transition-probability matrices.

The implementation is intended primarily as a research and numerical-validation tool for robust control of Markov jump linear systems under uncertain cluster observations.
