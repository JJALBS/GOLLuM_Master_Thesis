"""
Bayesian optimization loop over latent embeddings (PEP 8 styled).

This module provides two functions used to perform Bayesian optimization (BO)
directly in the latent space produced by the GOLLuM pipeline:

- :func:`next_sample_point`: fit a GP to current samples and optimize a
  q-Log Expected Improvement acquisition function to propose the next
  candidate(s).
- :func:`GPbased_optimizer`: mimic the optimization algorithm of
  Gomez-Bombarelli et al. by using a GP over latent embeddings as the
  black-box objective. It tracks the best objective and decoded molecular
  properties from three decoders (GRU, MLP, Vec2Text).
"""

from typing import Any, List, Tuple, Union

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor

from util.decoder_util import convert_to_string
from util.util import safe_logp


def next_sample_point(
    sample_x: Tensor,
    sample_y: Tensor,
    bounds: Tensor,
    num_restarts: int,
    raw_samples: int,
    n_points: int = 1,
) -> Tensor:
    """Propose the next candidate(s) via q-Log Expected Improvement.

    A GP is fit to the provided samples ``(sample_x, sample_y)``. The best
    observed objective is computed from the GP posterior mean at the observed
    points, and then q-LogEI is optimized within ``bounds`` to generate
    ``n_points`` new candidates.

    Parameters
    ----------
    sample_x : Tensor
        Training inputs of shape ``(N, d)`` used to fit the GP.
    sample_y : Tensor
        Training targets of shape ``(N,)`` or ``(N, 1)``.
    bounds : Tensor
        Search-space bounds of shape ``(2, d)``.
    num_restarts : int
        Number of restarts for acquisition optimization.
    raw_samples : int
        Number of raw samples for initialization of the optimizer.
    n_points : int, default=1
        Batch size (``q``) for the acquisition optimizer.

    Returns
    -------
    Tensor
        Candidate points of shape ``(n_points, d)``.
    """
    d = sample_x.shape[-1]

    # Fit a GP (with Normalize/Standardize transforms) to current data
    bo_gp = SingleTaskGP(
        sample_x.double(),
        sample_y.double().unsqueeze(-1),
        input_transform=Normalize(d),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(bo_gp.likelihood, bo_gp)
    fit_gpytorch_mll(mll)

    # Best observed objective from posterior mean at observed points
    with torch.no_grad():
        best_f = bo_gp.posterior(sample_x).mean.max()

    # Optimize q-LogEI to get the next candidate(s)
    acq = qLogExpectedImprovement(
        model=bo_gp,
        best_f=best_f,
        sampler=SobolQMCNormalSampler(torch.Size([1024])),
    )
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=n_points,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 5, "maxiter": 200},
    )
    return candidates


def GPbased_optimizer(
    DeepGP: Any,
    SimpleMLP: Any,
    GRU: Any,
    Vec2Text: Any,
    train_embed_data: Any,
    int_key: Any,
    max_loop: int,
    early_stop_tol: float,
    patience: int,
    iter_budget: int,
    initial_sample_num: int,
    projection_dim: int,
    num_restarts: int,
    raw_samples: int,
    seed: int,
) -> Tuple[
    List[Union[float, Tensor]], List[float], List[float], List[float], List[int]
]:
    """Run BO in the latent space and track best decoded properties.

    The GP used as the objective is the latent-space GP exposed by ``DeepGP``
    via its ``_latent_view()`` method. We seed initial points by sampling from
    a normal and matching their mean/scale to ``train_embed_data``. At each
    BO step, we propose a new candidate using :func:`next_sample_point`,
    update the running best objective, and evaluate decoders (GRU, MLP,
    Vec2Text) by converting outputs to strings and scoring with ``safe_logp``.

    Parameters
    ----------
    DeepGP : Any
        Trained model exposing ``device``, ``data_type``, and ``_latent_view()``.
    SimpleMLP : Any
        MLP decoder; callable on embeddings to produce token logits.
    GRU : Any
        GRU decoder; callable on embeddings to produce token logits.
    Vec2Text : Any
        Two-stage decoder; callable with keyword arguments
        ``max_loop``, ``latent_emb``, ``true_emb``, ``early_stop_tol``, ``patience``.
    train_embed_data : Any
        Iterable/Series with ``.to_list()`` producing training embeddings.
    int_key : Any
        Mapping from token indices to string tokens; passed to ``convert_to_string``.
    max_loop : int
        Max iterative refinement steps for ``Vec2Text``.
    early_stop_tol : float
        Early stopping tolerance on cosine-similarity improvements (``Vec2Text``).
    patience : int
        Number of non-improving steps before stopping (``Vec2Text``).
    iter_budget : int
        BO iteration budget.
    initial_sample_num : int
        Number of initial random samples for seeding the GP.
    projection_dim : int
        Dimensionality of latent embeddings.
    num_restarts : int
        Number of restarts for acquisition optimization.
    raw_samples : int
        Number of raw samples for initialization of the optimizer.
    seed : int
        Random seed for the initial sampling RNG.

    Returns
    -------
    tuple[list[float | Tensor], list[float], list[float], list[float], list[int]]
        ``(true_list, MLP_list, GRU_list, V2T_list, steps)`` where
        ``true_list`` stores the running best GP mean (first element is a
        0-dim Tensor for parity with the original code). Other lists store the
        running best decoded scores. ``steps`` is ``[0..iter_budget]``.
    """
    DeepGP.eval()
    SimpleMLP.eval()
    GRU.eval()
    Vec2Text.eval()

    # Running best trackers
    true_list: List[Union[float, Tensor]] = []
    GRU_list: List[float] = []
    MLP_list: List[float] = []
    V2T_list: List[float] = []

    # Objective GP from the DeepGP model
    obj_func = DeepGP._latent_view()

    # ----- Initial data generation -----
    rng = np.random.default_rng(seed)
    norm_sample = rng.standard_normal((initial_sample_num, projection_dim))

    data = np.array(train_embed_data.to_list())
    mean, std = np.mean(data, axis=0), np.std(data, axis=0)

    sample_x = torch.from_numpy(norm_sample * std + mean).to(
        device=DeepGP.device, dtype=DeepGP.data_type
    )

    with torch.no_grad():
        prior = obj_func.posterior(sample_x)
        sample_y = prior.mean.squeeze(-1).detach().cpu()

    best_y = torch.max(sample_y)

    # Bounds in latent space
    mins = torch.tensor(data.min(axis=0).tolist(), dtype=torch.double)
    maxs = torch.tensor(data.max(axis=0).tolist(), dtype=torch.double)
    bound = torch.stack([mins, maxs]).to(device=DeepGP.device, dtype=DeepGP.data_type)

    # Track current best
    true_list.append(best_y)

    argbest_y = torch.argmax(sample_y)
    best_embed = sample_x[argbest_y, :].unsqueeze(0).to(
        device=DeepGP.device, dtype=DeepGP.data_type
    )

    # Decode initial best with each decoder
    with torch.no_grad():
        best_GRU = safe_logp(
            convert_to_string(
                GRU(best_embed).detach().cpu(),
                id_to_token=int_key,
                to_smiles=True,
                input_data_type="one-hot",
            )[0]
        )
        best_MLP = safe_logp(
            convert_to_string(
                SimpleMLP(best_embed).detach().cpu(),
                id_to_token=int_key,
                to_smiles=True,
                input_data_type="one-hot",
            )[0]
        )
        best_V2T = safe_logp(
            convert_to_string(
                Vec2Text(
                    max_loop=max_loop,
                    latent_emb=best_embed,
                    true_emb=best_embed,
                    early_stop_tol=early_stop_tol,
                    patience=patience,
                ).detach().cpu(),
                id_to_token=int_key,
                to_smiles=True,
                input_data_type="label",
            )[0]
        )

    GRU_list.append(best_GRU)
    MLP_list.append(best_MLP)
    V2T_list.append(best_V2T)

    # ----- BO iterations -----
    for i in range(iter_budget):
        print(f"No. of optimization run: {i}")
        next_candidate = next_sample_point(
            sample_x=sample_x,
            sample_y=sample_y,
            bounds=bound,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        with torch.no_grad():
            new_post = obj_func.posterior(next_candidate)
        true_result = new_post.mean.squeeze(-1).to(
            device=sample_y.device, dtype=sample_y.dtype
        )

        sample_x = torch.cat([sample_x, next_candidate])
        sample_y = torch.cat([sample_y, true_result], dim=0)
        best_y = sample_y.max().item()

        # Update trackers
        true_list.append(best_y)

        with torch.no_grad():
            GRU_result = safe_logp(
                convert_to_string(
                    GRU(next_candidate).detach().cpu(),
                    id_to_token=int_key,
                    to_smiles=True,
                    input_data_type="one-hot",
                )[0]
            )
            if GRU_result > best_GRU:
                best_GRU = GRU_result
            GRU_list.append(best_GRU)

            MLP_result = safe_logp(
                convert_to_string(
                    SimpleMLP(next_candidate).detach().cpu(),
                    id_to_token=int_key,
                    to_smiles=True,
                    input_data_type="one-hot",
                )[0]
            )
            if MLP_result > best_MLP:
                best_MLP = MLP_result
            MLP_list.append(best_MLP)

            V2T_result = safe_logp(
                convert_to_string(
                    Vec2Text(
                        max_loop=max_loop,
                        latent_emb=next_candidate,
                        true_emb=next_candidate,
                        early_stop_tol=early_stop_tol,
                        patience=patience,
                    ).detach().cpu(),
                    id_to_token=int_key,
                    to_smiles=True,
                    input_data_type="label",
                )[0]
            )
            if V2T_result > best_V2T:
                best_V2T = V2T_result
            V2T_list.append(best_V2T)

    return true_list, MLP_list, GRU_list, V2T_list, list(range(iter_budget + 1))
