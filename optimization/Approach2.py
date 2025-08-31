"""
Decoder-driven Bayesian optimization in latent space (PEP 8 styled).

This module provides utilities to perform Bayesian optimization (BO) directly
over latent embeddings when the *decoder* (e.g., GRU, MLP, or Vec2Text) is
treated as the black-box objective. The core idea is to decode candidates to
SMILES, score them (e.g., logP), and then fit a GP model to the decoder's
scores to propose new latent candidates via q-Log Expected Improvement.
"""

from typing import Any, List, Optional, Tuple, Union

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

from model.Vec2Text_decoder import Vec2Text
from util.decoder_util import convert_to_string
from util.util import safe_logp


def to_module(x: Tensor, module: torch.nn.Module) -> Tensor:
    """Move ``x`` to the device/dtype of ``module``'s first parameter.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    module : torch.nn.Module
        Reference module whose parameter device/dtype should be adopted.

    Returns
    -------
    Tensor
        ``x`` moved to the device/dtype of ``module``.
    """
    p = next(module.parameters())
    return x.to(device=p.device, dtype=p.dtype)


def batch_decode(
    decoder: torch.nn.Module,
    latents: Tensor,
    int_key: Any,
    max_loop: Optional[int] = None,
    early_stop_tol: Optional[float] = None,
    patience: Optional[int] = None,
    chunk: int = 8,
) -> Tensor:
    """Decode latent points to scores in mini-batches.

    For :class:`Vec2Text`, this calls the module with its iterative refinement
    signature. For other decoders, it forwards once and converts the output
    tensor to SMILES via :func:`convert_to_string`. Each SMILES is scored
    with :func:`safe_logp` and returned as a tensor.

    Parameters
    ----------
    decoder : torch.nn.Module
        The decoder model (e.g., :class:`Vec2Text`, GRU, or MLP).
    latents : Tensor
        Candidate latent embeddings of shape ``(N, d)``.
    int_key : Any
        Mapping from integer token IDs to tokens, passed to
        :func:`convert_to_string`.
    max_loop : int | None, default=None
        Maximum refinement iterations for :class:`Vec2Text`.
    early_stop_tol : float | None, default=None
        Early-stopping tolerance for :class:`Vec2Text`.
    patience : int | None, default=None
        Early-stopping patience for :class:`Vec2Text`.
    chunk : int, default=8
        Mini-batch size to limit memory usage during decoding.

    Returns
    -------
    Tensor
        A 1-D tensor of scores on the same device/dtype as ``latents``.
    """
    vals: List[float] = []
    for i in range(0, latents.size(0), chunk):
        xb = to_module(latents[i : i + chunk], decoder)
        with torch.no_grad():
            if isinstance(decoder, Vec2Text):
                out = decoder(
                    max_loop=max_loop,
                    latent_emb=xb,
                    true_emb=xb,
                    early_stop_tol=early_stop_tol,
                    patience=patience,
                )
                smis = convert_to_string(
                    out.detach().cpu(),
                    id_to_token=int_key,
                    to_smiles=True,
                    input_data_type="label",
                )
            else:
                out = decoder(xb)
                smis = convert_to_string(
                    out.detach().cpu(),
                    id_to_token=int_key,
                    to_smiles=True,
                    input_data_type="one-hot",
                )
        if isinstance(smis, str):
            smis = [smis]
        vals.extend(safe_logp(s) for s in smis)

    return torch.tensor(vals, device=latents.device, dtype=latents.dtype)


def next_sample_point(
    sample_x: Tensor,
    sample_y: Tensor,
    bounds: Tensor,
    num_restarts: int,
    raw_samples: int,
    n_points: int = 1,
) -> Tensor:
    """Propose the next candidate(s) via q-Log Expected Improvement.

    A GP is fit to the provided samples ``(sample_x, sample_y)`` with
    normalization and standardization transforms. The best observed objective
    is computed from the GP posterior mean at the observed points, and then
    q-LogEI is optimized within ``bounds`` to generate ``n_points`` new
    candidates.

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
    device, dtype = sample_x.device, sample_x.dtype
    sample_y = sample_y.to(device=device, dtype=dtype)
    bounds = bounds.to(device=device, dtype=dtype)

    d = sample_x.shape[-1]

    # Fit GP to current data
    bo_gp = SingleTaskGP(
        sample_x.to(dtype=torch.double),
        sample_y.to(dtype=torch.double).unsqueeze(-1),
        input_transform=Normalize(d),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(bo_gp.likelihood, bo_gp)
    fit_gpytorch_mll(mll)

    # Best observed objective
    with torch.no_grad():
        best_f = bo_gp.posterior(sample_x).mean.max()

    # Optimize q-LogEI
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


def decoderbased_optimizer(
    DeepGP: Any,
    decoder: torch.nn.Module,
    train_embed_data: Any,
    iter_budget: int,
    initial_sample_num: int,
    projection_dim: int,
    num_restarts: int,
    raw_samples: int,
    seed: int,
    int_key: Any,
    max_loop: Optional[int] = None,
    early_stop_tol: Optional[float] = None,
    patience: Optional[int] = None,
) -> Tuple[List[Union[float, Tensor]], List[float], List[int]]:
    """Run BO in latent space using the decoder as the objective.

    A GP is fit to the decoder's scores (e.g., logP from decoded SMILES)
    evaluated at latent points. At each BO step, a new candidate is proposed
    via :func:`next_sample_point`, and both the running best decoder score and
    the GP mean at the best point are tracked.

    Parameters
    ----------
    DeepGP : Any
        Trained model exposing ``device``, ``data_type``, and ``_latent_view()``.
    decoder : torch.nn.Module
        Decoder model used to obtain scores from latents.
    train_embed_data : Any
        Iterable/Series with ``.to_list()`` producing training embeddings.
    iter_budget : int
        BO iteration budget.
    initial_sample_num : int
        Number of initial random samples used to seed the GP.
    projection_dim : int
        Dimensionality of latent embeddings.
    num_restarts : int
        Number of restarts for acquisition optimization.
    raw_samples : int
        Number of raw samples for initialization of the optimizer.
    seed : int
        Random seed for the initial sampling RNG.
    int_key : Any
        Mapping for token ID → token strings used by :func:`convert_to_string`.
    max_loop : int | None, default=None
        Max refinement iterations for :class:`Vec2Text` (if used).
    early_stop_tol : float | None, default=None
        Early-stopping tolerance for :class:`Vec2Text` (if used).
    patience : int | None, default=None
        Early-stopping patience for :class:`Vec2Text` (if used).

    Returns
    -------
    tuple[list[float | Tensor], list[float], list[int]]
        ``(true_list, decoder_list, steps)`` where
        ``true_list`` stores the running best GP means (first element is a
        float, later elements may be tensors for parity with original code),
        ``decoder_list`` stores the running best decoder scores,
        and ``steps`` is ``[0..iter_budget]``.
    """
    DeepGP.eval()
    decoder.eval()

    true_func = DeepGP._latent_view()

    # Running trackers to return
    true_list: List[Union[float, Tensor]] = []
    decoder_list: List[float] = []

    # ----- Initial data generation -----
    rng = np.random.default_rng(seed)
    norm_sample = rng.standard_normal((initial_sample_num, projection_dim))

    data = np.array(train_embed_data.to_list())
    mean, std = np.mean(data, axis=0), np.std(data, axis=0)
    sample_x = torch.from_numpy(norm_sample * std + mean).to(
        device=DeepGP.device, dtype=DeepGP.data_type
    )

    with torch.no_grad():
        sample_y = batch_decode(
            decoder,
            sample_x,
            int_key,
            max_loop=max_loop,
            early_stop_tol=early_stop_tol,
            patience=patience,
            chunk=8,
        )
    best_y = sample_y.max()

    # Bounds in latent space
    mins = torch.tensor(data.min(axis=0).tolist(), dtype=torch.double)
    maxs = torch.tensor(data.max(axis=0).tolist(), dtype=torch.double)
    bound = torch.stack([mins, maxs]).to(device=DeepGP.device, dtype=DeepGP.data_type)

    # Initialize trackers
    decoder_list.append(best_y.item())

    argbest_y = torch.argmax(sample_y)
    best_embed = sample_x[argbest_y, :].unsqueeze(0)
    with torch.no_grad():
        true_best_y = true_func.posterior(best_embed).mean.squeeze(-1)
    true_list.append(true_best_y.item())

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
            true_result = true_func.posterior(next_candidate).mean.squeeze(-1)
            decoder_result = batch_decode(
                decoder,
                next_candidate,
                int_key,
                max_loop=max_loop,
                early_stop_tol=early_stop_tol,
                patience=patience,
                chunk=8,
            )

        sample_x = torch.cat([sample_x, next_candidate])
        sample_y = torch.cat([sample_y, decoder_result], dim=0)
        best_y = sample_y.max().item()

        decoder_list.append(best_y)
        if true_result > true_best_y:
            true_best_y = true_result
        true_list.append(true_best_y)

    return true_list, decoder_list, list(range(iter_budget + 1))
