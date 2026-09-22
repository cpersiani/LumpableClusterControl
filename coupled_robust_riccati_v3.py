from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Mapping, Tuple
import numpy as np

Array = np.ndarray
Key = Hashable


def _sym(A: Array) -> Array:
    return 0.5 * (A + A.T)


@dataclass
class RiccatiResult:
    P: Dict[Key, Array]
    K: Dict[Key, Array]
    Psi: Dict[Key, Array]
    Pi: Dict[Key, Array]
    Z: Dict[Key, Array]
    Omega: Dict[Key, Array]
    lam: Dict[Key, float]
    iterations: int
    converged: bool
    error: float


class CoupledRobustRiccati:

    def __init__(
        self,
        F: Mapping[Key, Array],
        G: Mapping[Key, Array],
        Q: Mapping[Key, Array],
        R: Mapping[Key, Array],
        S: Mapping[Key, Array],
        Cbar: Mapping[Key, Array],
        M_C: Mapping[Key, Array],
        E_C: Mapping[Key, Array],
        qbar: Mapping[Key, Mapping[Key, float]],
        alpha: Mapping[Key, Mapping[Key, float]],
        mu1: float,
        mu2: float,
        beta: float = 1.01,
        tol: float = 1e-10,
        max_iter: int = 10_000,
    ):
        self.F = {k: np.asarray(v, dtype=float) for k, v in F.items()}
        self.G = {k: np.asarray(v, dtype=float) for k, v in G.items()}
        self.Q = {k: np.asarray(v, dtype=float) for k, v in Q.items()}
        self.R = {k: np.asarray(v, dtype=float) for k, v in R.items()}
        self.S = {k: np.asarray(v, dtype=float) for k, v in S.items()}

        self.Cbar = {k: np.asarray(v, dtype=float) for k, v in Cbar.items()}

        self.M_C = {k: np.asarray(v, dtype=float) for k, v in M_C.items()}
        self.E_C = {k: np.asarray(v, dtype=float) for k, v in E_C.items()}

        self.qbar = {k: dict(v) for k, v in qbar.items()}
        self.alpha = {k: dict(v) for k, v in alpha.items()}

        self.mu1 = float(mu1)
        self.mu2 = float(mu2)
        self.beta = float(beta)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.clusters = tuple(self.F.keys())
        self._validate()

    def _validate(self) -> None:
        if self.mu1 <= 0 or self.mu2 <= 0:
            raise ValueError("mu1 and mu2 must be positive.")
        if self.beta <= 1.0:
            raise ValueError("beta must be greater than one.")

        required = (
            self.G, self.Q, self.R, self.S,
            self.Cbar,
            self.M_C, self.E_C,
            self.qbar, self.alpha,
        )

        for d in required:
            if set(d.keys()) != set(self.clusters):
                raise ValueError(
                    "All dictionaries must have exactly the same cluster keys."
                )

        for l in self.clusters:
            n = self.F[l].shape[1]
            nz = self.F[l].shape[0]
            m = self.G[l].shape[1]

            if self.G[l].shape[0] != nz:
                raise ValueError(f"G[{l!r}] has inconsistent row dimension.")

            if self.Q[l].shape != (n, n):
                raise ValueError(f"Q[{l!r}] must be ({n}, {n}).")

            if self.R[l].shape != (m, m):
                raise ValueError(f"R[{l!r}] must be ({m}, {m}).")

            if self.S[l].shape != (nz, nz):
                raise ValueError(f"S[{l!r}] must be ({nz}, {nz}).")

            if self.Cbar[l].shape != (n, nz):
                raise ValueError(f"Cbar[{l!r}] must be ({n}, {nz}).")

            # delta C_l = M_C_l Delta_C_l E_C_l
            if self.M_C[l].shape[0] != n:
                raise ValueError(
                    f"M_C[{l!r}] must have {n} rows."
                )

            if self.E_C[l].shape[1] != nz:
                raise ValueError(
                    f"E_C[{l!r}] must have {nz} columns."
                )

            p_sum = sum(self.qbar[l].values())
            if not np.isclose(p_sum, 1.0):
                raise ValueError(
                    f"qbar[{l!r}] must sum to one; got {p_sum}."
                )

            if set(self.qbar[l]) != set(self.clusters):
                raise ValueError(
                    f"qbar[{l!r}] must contain every destination cluster exactly once."
                )
            if set(self.alpha[l]) != set(self.clusters):
                raise ValueError(
                    f"alpha[{l!r}] must contain every destination cluster exactly once."
                )
            if any(a < 0 for a in self.alpha[l].values()):
                raise ValueError(f"alpha[{l!r}] must be elementwise nonnegative.")

    def _lambda(self, l: Key) -> float:
        """lambda_l = beta * mu2 * ||M_C_l.T M_C_l||_2."""
        gram = self.M_C[l].T @ self.M_C[l]
        return self.beta * self.mu2 * np.linalg.norm(gram, ord=2)

    def _psi(self, l: Key, P: Mapping[Key, Array]) -> Array:
        """Psi_l = sum_d (qbar[l,d] + alpha[l,d]) P_d."""
        n = self.Q[l].shape[0]
        Psi = np.zeros((n, n))
        for d in self.clusters:
            Psi += (self.qbar[l][d] + self.alpha[l][d]) * P[d]
        return _sym(Psi)

    def _pi_selection(self, l: Key) -> Tuple[Array, float]:
        """
        Pi_l = (mu2^-1 I - lambda_l^-1 M_C M_C.T)^-1.

        If M_C = 0, the selection uncertainty vanishes and the limiting
        expression is Pi_l = mu2 I.
        """
        n = self.Cbar[l].shape[0]
        lam = self._lambda(l)
        gram = self.M_C[l] @ self.M_C[l].T

        if np.linalg.norm(gram, ord=2) <= np.finfo(float).eps:
            raise np.linalg.LinAlgError(
                f"Robust admissibility failed in cluster {l!r}: "
                f"M_C M_C.T is (numerically) zero."
            )

        H = (1.0 / self.mu2) * np.eye(n) - (1.0 / lam) * gram
        H = _sym(H)
        eig_min = np.linalg.eigvalsh(H).min()
        if eig_min <= 0:
            raise np.linalg.LinAlgError(
                f"Robust admissibility failed in cluster {l!r}: "
                f"mu2^-1 I - lambda_l^-1 M_C M_C.T is not positive "
                f"definite (minimum eigenvalue = {eig_min:.3e})."
            )
        return _sym(np.linalg.solve(H, np.eye(n))), lam

    def _theorem_matrices(
        self, l: Key, P: Mapping[Key, Array]
    ) -> Tuple[Array, Array, Array, Array, float]:
        """Return (Psi_l, Pi_l, Z_l, Omega_l, lambda_l)."""
        nz = self.F[l].shape[0]
        Psi = self._psi(l, P)
        Pi, lam = self._pi_selection(l)

        middle = Pi - Pi @ np.linalg.solve(Psi + Pi, Pi)
        Z = (
            self.S[l]
            + lam * (self.E_C[l].T @ self.E_C[l])
            + self.Cbar[l].T @ middle @ self.Cbar[l]
        )
        Z = _sym(Z)

        eig_min = np.linalg.eigvalsh(Z).min()
        if eig_min <= 0:
            raise np.linalg.LinAlgError(
                f"Z[{l!r}] is not positive definite "
                f"(minimum eigenvalue = {eig_min:.3e})."
            )

        Omega = self.mu1 * np.eye(nz) - self.mu1**2 * np.linalg.solve(
            self.mu1 * np.eye(nz) + Z,
            np.eye(nz)
        )
        
        return Psi, Pi, Z, _sym(Omega), lam

    def _riccati_theorem_update(
        self,
        l: Key,
        P: Mapping[Key, Array],
    ) -> Tuple[Array, Array, Array]:

        _, _, _, Omega, _ = self._theorem_matrices(l, P)

        H = (
            self.R[l]
            + self.G[l].T @ Omega @ self.G[l]
        )
        H = _sym(H)

        B = self.G[l].T @ Omega @ self.F[l]

        K = -np.linalg.solve(H, B)

        P_new = (
            self.Q[l]
            + self.F[l].T @ Omega @ self.F[l]
            - B.T @ np.linalg.solve(H, B)
        )

        return _sym(P_new), K, Omega

    def _riccati_corollary_update(
        self,
        l: Key,
        P: Mapping[Key, Array],
    ) -> Tuple[Array, Array, Array]:

        _, _, Z, _, _ = self._theorem_matrices(l, P)

        H = (
            self.R[l]
            + self.G[l].T @ Z @ self.G[l]
        )
        H = _sym(H)

        B = self.G[l].T @ Z @ self.F[l]

        K = -np.linalg.solve(H, B)

        P_new = (
            self.Q[l]
            + self.F[l].T @ Z @ self.F[l]
            - B.T @ np.linalg.solve(H, B)
        )

        return _sym(P_new), K, Z

    def solve_TheoremIV3(
        self,
        P0: Mapping[Key, Array] | None = None,
        verbose: bool = False,
    ) -> RiccatiResult:

        if P0 is None:
            P = {
                l: self.Q[l].copy()
                for l in self.clusters
            }
        else:
            P = {
                l: np.asarray(P0[l], dtype=float).copy()
                for l in self.clusters
            }

        converged = False
        error = np.inf

        for it in range(1, self.max_iter + 1):

            P_new: Dict[Key, Array] = {}
            K_new: Dict[Key, Array] = {}

            for l in self.clusters:
                P_new[l], K_new[l], _ = self._riccati_theorem_update(l, P)

            error = max(
                np.linalg.norm(
                    P_new[l] - P[l],
                    ord="fro",
                )
                for l in self.clusters
            )

            if verbose and (it == 1 or it % 50 == 0):
                print(
                    f"iteration {it:5d} | "
                    f"error = {error:.3e}"
                )

            P = P_new

            if error < self.tol:
                converged = True
                break

        # Compute controller from the final Riccati set.
        K_final: Dict[Key, Array] = {}
        Psi_final: Dict[Key, Array] = {}
        Pi_final: Dict[Key, Array] = {}
        Z_final: Dict[Key, Array] = {}
        Omega_final: Dict[Key, Array] = {}
        lam_final: Dict[Key, float] = {}

        for l in self.clusters:
            _, K_final[l], _ = self._riccati_theorem_update(l, P)
            (
                Psi_final[l],
                Pi_final[l],
                Z_final[l],
                Omega_final[l],
                lam_final[l],
            ) = self._theorem_matrices(l, P)

        return RiccatiResult(
            P=P,
            K=K_final,
            Psi=Psi_final,
            Pi=Pi_final,
            Z=Z_final,
            Omega=Omega_final,
            lam=lam_final,
            iterations=it,
            converged=converged,
            error=error,
        )

    def solve_CorollaryVI(self,
            P0: Mapping[Key, Array] | None = None,
            verbose: bool = False,
        ) -> RiccatiResult:
    
            if P0 is None:
                P = {
                    l: self.Q[l].copy()
                    for l in self.clusters
                }
            else:
                P = {
                    l: np.asarray(P0[l], dtype=float).copy()
                    for l in self.clusters
                }
    
            converged = False
            error = np.inf

    
            for it in range(1, self.max_iter + 1):
    
                P_new: Dict[Key, Array] = {}
                K_new: Dict[Key, Array] = {}
    
                for l in self.clusters:
                    P_new[l], K_new[l], _ = self._riccati_corollary_update(l, P)
    
                error = max(
                    np.linalg.norm(
                        P_new[l] - P[l],
                        ord="fro",
                    )
                    for l in self.clusters
                )
    
                if verbose and (it == 1 or it % 50 == 0):
                    print(
                        f"iteration {it:5d} | "
                        f"error = {error:.3e}"
                    )
    
                P = P_new
    
                if error < self.tol:
                    converged = True
                    break
    
            # Compute controller from the final Riccati set.
            K_final: Dict[Key, Array] = {}
            Psi_final: Dict[Key, Array] = {}
            Pi_final: Dict[Key, Array] = {}
            Z_final: Dict[Key, Array] = {}
            Omega_final: Dict[Key, Array] = {}
            lam_final: Dict[Key, float] = {}
    
            for l in self.clusters:
                _, K_final[l], _ = self._riccati_corollary_update(l, P)
                (
                    Psi_final[l],
                    Pi_final[l],
                    Z_final[l],
                    Omega_final[l],
                    lam_final[l],
                ) = self._theorem_matrices(l, P)
    
            return RiccatiResult(
                P=P,
                K=K_final,
                Psi=Psi_final,
                Pi=Pi_final,
                Z=Z_final,
                Omega=Omega_final,
                lam=lam_final,
                iterations=it,
                converged=converged,
                error=error,
            )


if __name__ == "__main__":

    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    ## Example extracted from 
    # Todorov, Fragoso, do Valle Costa "Detector-Based H Infinity Results for Discrete-Time Markov 
    # Jump Linear Systems with Partial Observations" 

    A_normal = np.array([
        [0.5637,  0.1133, -0.6607, -0.0062],
        [0.0198,  0.8368,  1.0512,  0.0089],
        [0.0033, -0.0450,  0.9481,  0.0159],
        [0.0381,  0.0073, -0.0164,  0.9999],
    ])

    B_normal = np.array([
        [ 2.9735, -0.0618],
        [-0.1175,  0.6414],
        [ 0.0112, -0.0165],
        [ 0.0812, -0.0006],
    ])

    A_fault = A_normal.copy()
    B_fault = B_normal.copy()


    F_lift = np.vstack([
        A_normal,
        A_fault,
        A_fault,
    ])

    G_lift = np.vstack([
        B_normal,
        0.1 * B_fault,
        0.1 * B_fault,
    ])

    n = A_normal.shape[0]     # 4
    nz = F_lift.shape[0]      # 12
    m = B_normal.shape[1]     # 2

    I4 = np.eye(n)
    Z4 = np.zeros((n, n))

    clusters = ("A", "B")

    F = {
        "A": F_lift.copy(),
        "B": F_lift.copy(),
    }

    G = {
        "A": G_lift.copy(),
        "B": G_lift.copy(),
    }

    # ---------------------------------------------------------------
    # Nominal selection matrices
    # ---------------------------------------------------------------

    # A = {Normal, UF}
    #
    # Cbar_A = [0.5 I   0.5 I   0]
    #
    Cbar_A = np.hstack([
        0.5 * I4,
        0.5 * I4,
        Z4,
    ])

    # B = {DF}
    #
    # Cbar_B = [0   0   I]
    #
    Cbar_B = np.hstack([
        Z4,
        Z4,
        I4,
    ])

    Cbar = {
        "A": Cbar_A,
        "B": Cbar_B,
    }

    # ---------------------------------------------------------------
    # System-selection uncertainty
    #
    # delta C_l = M_C_l Delta_C_l E_C_l
    # ---------------------------------------------------------------

    # Cluster A:
    #
    # delta C_A =
    #
    # [0.5 Delta I   -0.5 Delta I   0]
    #
    # = I Delta [0.5 I   -0.5 I   0]
    #
    M_C_A = I4

    E_C_A = np.hstack([
         0.5 * I4,
        -0.5 * I4,
         Z4,
    ])

    # Cluster B:
    #
    # delta C_B = 0


    M_C = {
        "A": I4,
        "B": I4,
    }
    E_C = {
        "A": np.hstack([0.5 * I4, -0.5 * I4, Z4]),
        "B": np.zeros((n, nz)),
    }


    # ---------------------------------------------------------------
    # Nominal clustered Markov chain
    #
    #         A       B
    #
    # A      0.95    0.05
    # B      1.00    0.00
    # ---------------------------------------------------------------

    qbar = {
        "A": {
            "A": 0.95,
            "B": 0.05,
        },
        "B": {
            "A": 1.00,
            "B": 0.00,
        },
    }

    # ---------------------------------------------------------------
    # Transition-probability uncertainty bounds
    #
    # delta q_ld = alpha_ld Delta_ld,  |Delta_ld| <= 1
    # ---------------------------------------------------------------

    alpha = {
        "A": {
            "A": 0.05,
            "B": 0.05,
        },
        "B": {
            "A": 0.00,
            "B": 0.00,
        },
    }

    # ---------------------------------------------------------------
    # Cost matrices
    # ---------------------------------------------------------------

    Q = {
        "A": np.eye(n),
        "B": np.eye(n),
    }

    R = {
        "A": np.eye(m),
        "B": np.eye(m),
    }

    S = {
        "A": np.eye(nz),
        "B": np.eye(nz),
    }

    # ---------------------------------------------------------------
    # Solve
    # ---------------------------------------------------------------

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
        mu1=1e8,
        mu2=10.0,
        beta=1.01,
        tol=1e-10,
        max_iter=10_000,
    )

    result = solver.solve_CorollaryVI(
        verbose=True
    )

    # result = solver.solve_TheoremIV3(
    #     verbose=True
    # )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    print(
        "\nConverged:",
        result.converged,
    )

    print(
        "Iterations:",
        result.iterations,
    )

    print(
        "Final error:",
        result.error,
    )

    for l in clusters:

        # print(f"\nP[{l}] =")
        # print(result.P[l])

        print(f"\nK[{l}] =")
        print(result.K[l])
