# Robust Control of Markov Jump Linear Systems with Cluster Observations

This repository provides the Python implementation accompanying the paper:

> **Control of Markov Jump Linear Systems with Lumpable Cluster Observations**  
> Carlos A. F. Persiani, Ram Padmanabhan, Melkior Ornik, and Marco H. Terra.

The code implements the coupled robust Riccati recursions developed in the paper for the control of discrete-time Markov jump linear systems (MJLS) when the true Markov mode is not directly observed. Instead, the controller has access only to a cluster containing the current mode.

Uncertainty associated with both the active system dynamics and the transition probabilities between observed clusters is incorporated through structured norm-bounded uncertainty models.

## Main Implementation

The main implementation is provided in:

`coupled_robust_riccati_v3.py`

The class `CoupledRobustRiccati` implements two versions of the proposed Riccati recursion.

### Theorem IV.3 — Finite-Penalty Robust Riccati Recursion

The method

`solve_TheoremIV3()`

implements the recursion presented in Theorem IV.3 of the paper.

For each observed cluster \(l\), the algorithm constructs

\[
\Psi_{l}
=
\sum_{d}
(\bar q_{ld}+\alpha_{ld})P_d,
\]

together with the auxiliary matrices

\[
\Pi_l
=
\left(
\mu_2^{-1}I
-
\lambda_l^{-1}M_{C_l}M_{C_l}^{T}
\right)^{-1},
\]

and

\[
\mathcal Z_l
=
S_l
+
\lambda_l E_{C_l}^{T}E_{C_l}
+
\bar C_l^{T}
\left[
\Pi_l
-
\Pi_l(\Psi_l+\Pi_l)^{-1}\Pi_l
\right]
\bar C_l.
\]

For finite \(\mu_1\), the effective weighting matrix is

\[
\Omega_l
=
\mu_1 I
-
\mu_1^2
(\mu_1 I+\mathcal Z_l)^{-1}.
\]

The corresponding cluster-dependent controller is

\[
u_k = K_l x_k,
\]

with

\[
K_l
=
-
\left(
R_l+G^T\Omega_lG
\right)^{-1}
G^T\Omega_lF.
\]

### Corollary V.1 — Exact-Dynamics Limit

The method

`solve_CorollaryVI()`

implements the limiting recursion used in the convergence and stability analysis of the paper.

Taking

\[
\mu_1 \rightarrow \infty
\]

gives

\[
\Omega_l \rightarrow \mathcal Z_l.
\]

The resulting coupled Riccati recursion is

\[
P_{l,i+1}
=
Q_l
+
F^T\mathcal Z_{l,i}F
-
F^T\mathcal Z_{l,i}G
\left(
R_l+G^T\mathcal Z_{l,i}G
\right)^{-1}
G^T\mathcal Z_{l,i}F,
\]

with feedback gain

\[
K_{l,i}
=
-
\left(
R_l+G^T\mathcal Z_{l,i}G
\right)^{-1}
G^T\mathcal Z_{l,i}F.
\]

The iterations are performed simultaneously for all observed clusters because the Riccati equations are coupled through \(\Psi_l\).

## Requirements

The implementation requires:

- Python 3
- NumPy

Install the required dependency using:

```bash
pip install numpy
