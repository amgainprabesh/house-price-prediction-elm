"""ELM core: a single-hidden-layer feedforward network trained in one shot.

Model
-----
For a training set {(x_i, y_i)} with x_i in R^d and y_i in R:

    H = g(X W + b)                      (N x L hidden-layer output matrix)
    beta = H^+ Y                        (Moore-Penrose pseudoinverse solution)
    y_hat(x) = h(x) beta                (h(x) = g(x W + b), row vector)

W (d x L) and b (L,) are drawn once from a uniform distribution and are never
updated; only the output weights beta (L x 1) are learned, by solving the
linear least-squares system in closed form. This removes gradient descent and
back-propagation entirely, so training is a single matrix factorization.

A ridge-regularised variant is offered (`alpha > 0`), solving

    beta = (H^T H + alpha I)^-1 H^T Y

which stabilises the solution when H is ill-conditioned or L is large.
"""

from __future__ import annotations

import time

import numpy as np

# --------------------------------------------------------------------------
# Activation functions g(.)
# --------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic sigmoid, numerically stable for large |z|."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def _sine(z: np.ndarray) -> np.ndarray:
    return np.sin(z)


def _hardlim(z: np.ndarray) -> np.ndarray:
    return (z >= 0.0).astype(float)


ACTIVATIONS = {
    "sigmoid": _sigmoid,
    "tanh": _tanh,
    "relu": _relu,
    "sine": _sine,
    "hardlim": _hardlim,
}


def get_activation(name: str):
    """Return the activation callable registered under `name`."""
    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown activation {name!r}. "
            f"Available: {', '.join(sorted(ACTIVATIONS))}"
        ) from None


# --------------------------------------------------------------------------
# Regressor
# --------------------------------------------------------------------------


class ELMRegressor:
    """Extreme Learning Machine for single-output regression.

    Parameters
    ----------
    n_hidden : int
        Number of hidden-layer neurons L.
    activation : str
        One of 'sigmoid', 'tanh', 'relu', 'sine', 'hardlim'.
    alpha : float
        Ridge regularisation strength. 0 (default) uses the plain
        Moore-Penrose pseudoinverse; a small positive value (e.g. 1e-4)
        uses the regularised closed-form solution.
    scale : float
        Bound of the uniform distribution U(-scale, scale) used for the
        random input weights and biases.
    random_state : int | None
        Seed for the weight draw - makes a fit exactly reproducible.
    """

    def __init__(
        self,
        n_hidden: int = 256,
        activation: str = "sigmoid",
        alpha: float = 0.0,
        scale: float = 1.0,
        random_state: int | None = 42,
    ) -> None:
        self.n_hidden = int(n_hidden)
        self.activation = activation
        self.alpha = float(alpha)
        self.scale = float(scale)
        self.random_state = random_state

        self.input_weights_: np.ndarray | None = None
        self.biases_: np.ndarray | None = None
        self.beta_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.training_time_: float | None = None

    # -- internals ---------------------------------------------------------

    def _hidden(self, X: np.ndarray) -> np.ndarray:
        """Hidden-layer output matrix H = g(X W + b)."""
        g = get_activation(self.activation)
        return g(X @ self.input_weights_ + self.biases_)

    def _init_weights(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.input_weights_ = rng.uniform(
            -self.scale, self.scale, size=(n_features, self.n_hidden)
        )
        self.biases_ = rng.uniform(-self.scale, self.scale, size=self.n_hidden)

    # -- public API --------------------------------------------------------

    def fit(self, X, y) -> "ELMRegressor":
        """Draw the random hidden layer and solve for the output weights."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]} values"
            )

        self.n_features_in_ = X.shape[1]
        self._init_weights(self.n_features_in_)

        start = time.perf_counter()
        H = self._hidden(X)

        if self.alpha > 0.0:
            A = H.T @ H + self.alpha * np.eye(self.n_hidden)
            self.beta_ = np.linalg.solve(A, H.T @ y)
        else:
            self.beta_ = np.linalg.pinv(H) @ y

        self.training_time_ = time.perf_counter() - start
        return self

    def predict(self, X) -> np.ndarray:
        """Predict a price for every row of X (1-D array of length n_rows)."""
        if self.beta_ is None:
            raise RuntimeError("ELMRegressor must be fitted before predict()")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        return (self._hidden(X) @ self.beta_).ravel()

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the trained parameters to an .npz file."""
        if self.beta_ is None:
            raise RuntimeError("Nothing to save - fit the model first")
        np.savez(
            path,
            input_weights=self.input_weights_,
            biases=self.biases_,
            beta=self.beta_,
            n_hidden=self.n_hidden,
            activation=np.array(self.activation),
            alpha=self.alpha,
            scale=self.scale,
            random_state=np.array(-1 if self.random_state is None
                                  else self.random_state),
            n_features_in=self.n_features_in_,
            training_time=self.training_time_ or 0.0,
        )

    @classmethod
    def load(cls, path: str) -> "ELMRegressor":
        """Load a model previously written by save()."""
        with np.load(path, allow_pickle=False) as data:
            model = cls(
                n_hidden=int(data["n_hidden"]),
                activation=str(data["activation"]),
                alpha=float(data["alpha"]),
                scale=float(data["scale"]),
                random_state=int(data["random_state"]),
            )
            model.input_weights_ = data["input_weights"]
            model.biases_ = data["biases"]
            model.beta_ = data["beta"]
            model.n_features_in_ = int(data["n_features_in"])
            model.training_time_ = float(data["training_time"])
        return model

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ELMRegressor(n_hidden={self.n_hidden}, "
            f"activation={self.activation!r}, alpha={self.alpha})"
        )