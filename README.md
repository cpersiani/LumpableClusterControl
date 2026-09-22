# Robust Control of Markov Jump Linear Systems with Cluster Observations

This repository provides the Python implementation accompanying the paper:

> **Control of Markov Jump Linear Systems with Lumpable Cluster Observations**  
> Carlos A. F. Persiani, Ram Padmanabhan, Melkior Ornik, and Marco H. Terra.

The code implements the coupled robust Riccati recursions developed in the paper for the control of discrete-time Markov jump linear systems (MJLS) when the true Markov mode is not directly observed. Instead, the controller has access only to a cluster containing the current mode.

Uncertainty associated with both the active system dynamics and the transition probabilities between observed clusters is incorporated through structured norm-bounded uncertainty models.

## Main Implementation

The main implementation is provided in:

```text
coupled_robust_riccati_v3.py
```

The class `CoupledRobustRiccati` implements two versions of the proposed Riccati recursion.

### Theorem IV.3 — Finite-Penalty Robust Riccati Recursion

The method

```python
solve_TheoremIV3()
```

implements the recursion presented in Theorem IV.3 of the paper.

For each observed cluster $l$, the algorithm constructs

$$
\Psi_{l,k+1}
=
\sum_{d\in\hat{\Theta}}
\left(
\bar q_{ld}+\alpha_{ld}
\right)P_{d,k+1},
$$

and defines

$$
\lambda_l
=
\beta\mu_2
\left\|
M_{C_l}^{T}M_{C_l}
\right\|,
$$

$$
\Pi_l
=
\left(
\mu_2^{-1}I_n
-
\lambda_l^{-1}M_{C_l}M_{C_l}^{T}
\right)^{-1},
$$

and

$$
\mathcal{Z}_{l,k+1}
=
S_l
+
\lambda_l E_{C_l}^{T}E_{C_l}
+
\bar C_l^{T}
\left[
\Pi_l
-
\Pi_l
\left(
\Psi_{l,k+1}+\Pi_l
\right)^{-1}
\Pi_l
\right]
\bar C_l.
$$

For finite $\mu_1$, the effective weighting matrix is

$$
\Omega_{l,k+1}
=
\mu_1 I_{ns}
-
\mu_1^2
\left(
\mu_1 I_{ns}
+
\mathcal{Z}_{l,k+1}
\right)^{-1}.
$$

The Riccati recursion is

$$
P_{l,k}
=
Q_l
+
F_k^{T}\Omega_{l,k+1}F_k
-
F_k^{T}\Omega_{l,k+1}G_k
\left(
R_l
+
G_k^{T}\Omega_{l,k+1}G_k
\right)^{-1}
G_k^{T}\Omega_{l,k+1}F_k.
$$

The corresponding cluster-dependent feedback law is

$$
u_k = K_{l,k}x_k,
$$

where

$$
K_{l,k}
=
-
\left(
R_l
+
G_k^{T}\Omega_{l,k+1}G_k
\right)^{-1}
G_k^{T}\Omega_{l,k+1}F_k.
$$

### Corollary V.1 — Exact-Dynamics Limit

The method

```python
solve_CorollaryVI()
```

implements the limiting recursion used in the convergence and stability analysis.

Taking

$$
\mu_1 \rightarrow \infty
$$

gives

$$
\Omega_{l,i}
\rightarrow
\mathcal{Z}_{l,i}.
$$

The resulting coupled Riccati recursion is

$$
P_{l,i+1}
=
Q_l
+
F^{T}\mathcal{Z}_{l,i}F
-
F^{T}\mathcal{Z}_{l,i}G
\left(
R_l
+
G^{T}\mathcal{Z}_{l,i}G
\right)^{-1}
G^{T}\mathcal{Z}_{l,i}F,
$$

with cluster-dependent feedback gain

$$
K_{l,i}
=
-
\left(
R_l
+
G^{T}\mathcal{Z}_{l,i}G
\right)^{-1}
G^{T}\mathcal{Z}_{l,i}F.
$$

The iterations are performed simultaneously for all observed clusters because the Riccati equations are coupled through $\Psi_{l,i}$.

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

- the lifted system matrices $F$ and $G$;
- the observed clusters;
- the weighting matrices $Q_l$, $R_l$, and $S_l$;
- the nominal cluster transition probabilities $\bar q_{ld}$;
- the transition uncertainty bounds $\alpha_{ld}$;
- the nominal selection matrices $\bar C_l$;
- the uncertainty matrices $M_{C_l}$ and $E_{C_l}$;
- the regularization parameters $\mu_1$, $\mu_2$, and $\beta$.

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

To solve the finite-$\mu_1$ recursion from Theorem IV.3:

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

The numerical example included in the code corresponds to the fault-susceptible lateral-directional aircraft dynamics considered in the accompanying paper.

The example illustrates how to:

1. construct the mode-lifted matrices $F$ and $G$;
2. define the observed clusters;
3. specify the uncertain cluster transition probabilities;
4. construct the selection-matrix uncertainty;
5. solve the coupled robust Riccati equations; and
6. obtain one feedback gain for each observed cluster.

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

This repository contains research code associated with the theoretical developments of the accompanying paper. The implementation is primarily intended to reproduce the proposed Riccati-based controller and the numerical examples presented in the manuscript.

For the mathematical assumptions, uncertainty construction, convergence conditions, and stability guarantees, please refer to the accompanying paper.
