"""
GOLLuM DeepGP (PEP 8–styled)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.optim.lr_scheduler import StepLR

import gpytorch
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import Mean
from gpytorch.module import Module

import wandb

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.posteriors import Posterior
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from model.gollum_LLM import LLMFeaturizer


class SurrogateModel(ABC):
    """Abstract base class for surrogate models."""

    @abstractmethod
    def fit(self) -> None:
        """Train the surrogate model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test: Tensor) -> Tensor:
        """Make predictions for the given test inputs.

        Parameters
        ----------
        X_test : Tensor
            Test inputs.

        Returns
        -------
        Tensor
            Model predictions.
        """
        raise NotImplementedError


class DeepGP(SurrogateModel, SingleTaskGP):
    """Deep Gaussian Process surrogate with an LLM featurizer front-end.

    This class wraps a BoTorch :class:`SingleTaskGP` whose inputs are the
    embeddings produced by an :class:`LLMFeaturizer`. It supports training
    with a custom closure and basic Bayesian optimization over the learned
    latent space via Expected Improvement.

    Notes
    -----
    - Argument names are preserved exactly as provided in the original file.
    - Behavior is preserved; only style, hints, and docstrings were added.
    - No shape/device validation lines were introduced.
    """

    def __init__(
        self,
        train_x: Union[np.ndarray, Tensor] = None,
        train_y: Union[np.ndarray, Tensor] = None,
        likelihood: Union[GaussianLikelihood, None] = None,
        covar_module: Union[Module, None] = None,
        mean_module: Union[Mean, None] = None,
        standardize: bool = True,
        normalize: bool = False,
        initial_noise_val: float = 1e-4,
        noise_constraint: float = 1e-5,
        initial_outputscale_val: float = 2.0,
        initial_lengthscale_val: float = 5.0,
        ft_lr: float = 0.002,
        gp_lr: float = 0.02,
        gp_step_lr: float = 0.95,
        wd: float = 1e-3,
        wd_llm: float = 1e-3,
        scale_embeddings: bool = False,
        train_mll_additionally: bool = False,
        # LLMFeaturizer inputs
        model_name: str = "T5-base",
        trainable: bool = True,
        target_ratio: float = 0.25,
        from_top: bool = True,
        lora_dropout: float = 0.2,
        modules_to_save: Optional[List[str]] = ["head"],
        pooling_method: str = "cls",
        normalize_embeddings: bool = False,
        projection_dim: Optional[int] = None,
        # Additional dynamic decision inputs
        device: torch.device = torch.device("cuda"),
        data_type: torch.dtype = torch.float64,
    ) -> None:
        """Initialize the DeepGP model.

        Parameters
        ----------
        train_x, train_y
            Training inputs and targets (NumPy arrays or PyTorch tensors).
        likelihood
            Optional Gaussian likelihood for the GP.
        covar_module, mean_module
            Optional covariance and mean modules for the GP.
        standardize, normalize
            Whether to standardize outputs and/or normalize inputs for the GP.
        initial_noise_val, noise_constraint, initial_outputscale_val,
        initial_lengthscale_val
            Initial GP hyperparameters.
        ft_lr, gp_lr, gp_step_lr
            Learning rates for the featurizer and GP, and LR gamma for scheduler.
        wd, wd_llm
            Weight decay values for GP and LLM featurizer parameter groups.
        scale_embeddings
            If ``True``, scale embeddings to [-1, 1] before passing to the GP.
        train_mll_additionally
            If ``True``, freeze the featurizer and re-fit only the GP MLL.
        model_name, trainable, target_ratio, from_top, lora_dropout, modules_to_save,
        pooling_method, normalize_embeddings, projection_dim
            Configuration arguments for :class:`LLMFeaturizer`.
        device, data_type
            Compute device and default dtype for the model and tensors.
        """
        print(f"Selected device = {device}")
        print(f"Selected data type = {data_type}")

        if device == torch.device("cuda"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == torch.device("cpu"):
                print("Unfortunately, CPU is used")  # kept for behavior parity

        tkwargs = {"device": device, "dtype": data_type}
        self.device = device
        self.data_type = data_type

        # Cast training data to the requested dtype/device (behavior preserved)
        train_x = train_x.to(dtype = torch.float64)
        train_y = train_y.to(device = self.device, dtype = torch.float64)

        super().__init__(
            train_X=train_x,
            train_Y=train_y,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
            input_transform=Normalize(train_x.shape[-1]) if normalize else None,
        )

        self.train_x = train_x
        self.train_y = train_y

        self.mean_module = mean_module
        self.covar_module = covar_module

        finetuning_model = LLMFeaturizer(
            model_name=model_name,
            trainable=trainable,
            target_ratio=target_ratio,
            from_top=from_top,
            lora_dropout=lora_dropout,
            pooling_method=pooling_method,
            projection_dim=projection_dim,
            normalize_embeddings=normalize_embeddings,
            device=device,
            data_type=data_type,
        )

        self.finetuning_model = finetuning_model.to(**tkwargs)
        self.likelihood.noise_covar.register_constraint(
            "raw_noise", GreaterThan(noise_constraint)
        )

        hypers = {
            "likelihood.noise_covar.noise": torch.tensor(initial_noise_val),
            "covar_module.base_kernel.lengthscale": torch.tensor(
                initial_lengthscale_val
            ),
            "covar_module.outputscale": torch.tensor(initial_outputscale_val),
        }

        existing_parameters = {name for name, _ in self.named_parameters()}
        hypers_to_use = {
            k: torch.tensor(v) for k, v in hypers.items() if k in existing_parameters and v is not None
        }

        self.initialize(**hypers_to_use)
        self.train_x = self.train_x.to(**tkwargs)
        self.train_y = self.train_y.to(**tkwargs)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

        self.ft_lr = ft_lr
        self.gp_lr = gp_lr
        self.gp_step_lr = gp_step_lr
        self.wd = wd
        self.wd_llm = wd_llm
        self.scale_embeddings = scale_embeddings
        self.train_mll_additionally = train_mll_additionally
        self.projection_dim = projection_dim

        self.mll_history: List[float] = []
        self.mll_eval_history: List[float] = []
        self.mse_eval_history: List[float] = []

        self.to_device()

    def forward(self, x: Tensor) -> MultivariateNormal:
        """GP forward pass using LLM embeddings as inputs.

        Parameters
        ----------
        x : Tensor
            Input tensor to be embedded and passed to the GP.

        Returns
        -------
        MultivariateNormal
            GP prior/posterior distribution over outputs.
        """
        finetuned = self.finetuning_model(x)
        if self.scale_embeddings:
            finetuned = self.scale_to_bounds(finetuned)
        self.finetuned = finetuned  # preserved side effect

        mean_x = self.mean_module(self.finetuned)
        covar_x = self.covar_module(self.finetuned)

        # Preserve wandb logging side effects (requires self.optimizer to exist)
        wandb.log({"lr/llm_lr": self.optimizer.param_groups[0]["lr"]})
        wandb.log({"lr/gp_lr": self.optimizer.param_groups[1]["lr"]})

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def to_device(self) -> None:
        """Move model components and training data to the configured device/dtype."""
        self.to(device=self.device, dtype=self.data_type)
        self.likelihood.to(device=self.device, dtype=self.data_type)
        self.finetuning_model.to(device=self.device, dtype=self.data_type)
        self.train_x = self.train_x.to(device=self.device, dtype=self.data_type)
        self.train_y = self.train_y.to(device=self.device, dtype=self.data_type)

    def fit(self) -> None:
        """Train the model by maximizing the exact marginal log-likelihood."""
        self.train()
        self.likelihood.train()
        self.finetuning_model.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        mll.train()
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)  # noqa: F841
        mll.to(device=self.device, dtype=self.data_type)

        def gp_closure() -> Tuple[Tensor, list]:
            """Training closure used by BoTorch's MLL optimizer.

            Returns
            -------
            tuple
                The MLL loss tensor and a list of gradients of trainable params.
            """
            self.optimizer.zero_grad()
            output = self(self.train_x)  # TRAIN mode forward
            mll_loss = -mll(output, self.train_targets.squeeze())
            mll_loss.backward()

            # training-mode NMLL (what you're optimizing)
            self.mll_history.append(mll_loss.item())

            # ------- eval-mode metrics (no grad, and we restore modes) -------
            with torch.no_grad():
                was_train = self.training
                was_lik_train = self.likelihood.training
                was_ft_train = self.finetuning_model.training

                self.eval()
                self.likelihood.eval()
                self.finetuning_model.eval()

                # Eval MSE on ORIGINAL target scale
                post = self.posterior(self.train_x, observation_noise=True)
                y_pred = post.mean
                if y_pred.dim() > 1:
                    y_pred = y_pred.squeeze(-1)
                y_true = self.train_y
                if y_true.dim() > 1:
                    y_true = y_true.squeeze(-1)
                mse_eval = torch.nn.functional.mse_loss(y_pred, y_true)
                self.mse_eval_history.append(mse_eval.item())

                # Eval negative MLL (standardized targets)
                out_eval = self(self.train_x)  # eval-mode forward
                nmlle = -mll(out_eval, self.train_targets.squeeze()).item()
                self.mll_eval_history.append(nmlle)

                # restore modes
                if was_train:
                    self.train()
                if was_lik_train:
                    self.likelihood.train()
                if was_ft_train:
                    self.finetuning_model.train()

            grads = [p.grad for p in self.parameters() if p.requires_grad]
            return mll_loss, grads

        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": (
                        p for p in self.finetuning_model.parameters() if p.requires_grad
                    ),
                    "lr": self.ft_lr,
                    "weight_decay": self.wd_llm,
                },
                {"params": self.covar_module.parameters()},
                {"params": self.mean_module.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=self.gp_lr,
            weight_decay=self.wd,
        )

        scheduler = StepLR(self.optimizer, step_size=1, gamma=0.95)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        fit_gpytorch_mll(
            mll,
            closure=gp_closure,
            optimizer=fit_gpytorch_mll_torch,
            optimizer_kwargs={"optimizer": self.optimizer, "scheduler": scheduler},
        )

        if self.train_mll_additionally:
            for param in self.finetuning_model.parameters():
                param.requires_grad = False
            fit_gpytorch_mll(mll)

    def predict(
        self,
        x: Tensor,
        observation_noise: bool = True,
        return_var: bool = True,
        return_posterior: bool = False,
    ) -> Union[Posterior, Tuple[Tensor, Tensor], Tensor]:
        """Predict outputs or posterior from the trained model.

        Parameters
        ----------
        x : Tensor
            Inputs for prediction.
        observation_noise : bool, default=True
            If ``True``, include observation noise in the posterior.
        return_var : bool, default=True
            If ``True`` and ``return_posterior`` is ``False``, return
            a tuple of (mean, variance).
        return_posterior : bool, default=False
            If ``True``, return the full posterior object.

        Returns
        -------
        Posterior | (Tensor, Tensor) | Tensor
            Posterior, or (mean, variance), or mean depending on flags.
        """
        torch.cuda.empty_cache()
        for param in self.finetuning_model.parameters():
            param.requires_grad = False

        self.eval()
        self.finetuning_model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            posterior = self.posterior(x, observation_noise=observation_noise)

        if return_posterior:
            return posterior
        if return_var:
            return posterior.mean, posterior.variance
        return posterior.mean

    def _latent_view(self) -> SingleTaskGP:
        """Create a read-only GP on latent z using fitted hyperparameters.

        Returns
        -------
        SingleTaskGP
            A GP that operates directly in the embedding (latent) space.
        """
        self.eval()
        with torch.no_grad():
            z_train = self.finetuning_model(self.train_x)
            if self.scale_embeddings:
                z_train = self.scale_to_bounds(z_train)

        gp_latent = SingleTaskGP(
            train_X=z_train.to(dtype = torch.float64),
            train_Y=self.train_y.to(dtype = torch.float64),
            likelihood=self.likelihood,
            covar_module=self.covar_module,
            mean_module=self.mean_module,
            outcome_transform=self.outcome_transform,  # if you used it
            input_transform=None,  # z is already in the GP space
        )
        gp_latent.eval()
        for p in gp_latent.parameters():
            p.requires_grad_(False)
        return gp_latent

    def opt_from_embed(
        self,
        q: int = 10,
        maximize: bool = True,
        pad_frac: float = 0.25,
        num_restarts: int = 20,
        raw_samples: int = 512,
        sampler_n: int = 128,
    ) -> Tensor:
        """Optimize q-EI over the latent space to propose new embeddings.

        Parameters
        ----------
        q : int, default=10
            Number of candidates to propose.
        maximize : bool, default=True
            If ``True``, maximize the objective; otherwise minimize.
        pad_frac : float, default=0.25
            Fractional padding for the latent bounds around observed z.
        num_restarts : int, default=20
            Number of restarts for the acquisition optimizer.
        raw_samples : int, default=512
            Number of raw samples for initialization.
        sampler_n : int, default=128
            Number of MC samples for the qEI sampler.

        Returns
        -------
        Tensor
            Proposed latent candidates of shape ``q x d``.
        """
        gp_latent = self._latent_view()

        with torch.no_grad():
            z_train = gp_latent.train_inputs[0]
            post = gp_latent.posterior(z_train)
            best_f = post.mean.max() if maximize else (-post.mean).max()

        # per-dim bounds around z_train (+ small padding)
        zmin, _ = z_train.min(dim=0)
        zmax, _ = z_train.max(dim=0)
        pad = (zmax - zmin).clamp_min(1e-8) * pad_frac
        bounds = torch.stack([zmin - pad, zmax + pad])

        acq = qExpectedImprovement(
            model=gp_latent,
            best_f=best_f,
            sampler=SobolQMCNormalSampler(torch.Size([sampler_n, self.projection_dim])),
        )
        X, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 4, "maxiter": 200},
        )
        return X.detach()
