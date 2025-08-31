import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qExpectedImprovement
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem import Crippen

from util.decoder_util import convert_to_string
from util.util import safe_logp

def next_sample_point(sample_x, sample_y, best_y, bounds, num_restarts, raw_samples, n_points=1):
    # Constructing priori GP using the sample data
    BO_GP = SingleTaskGP(sample_x.double(), sample_y.double().unsqueeze(-1))
    mll = ExactMarginalLogLikelihood(BO_GP.likelihood, BO_GP)

    fit_gpytorch_mll(mll)

    # Constructing acquisition function from the priori
    EI = qExpectedImprovement(model=BO_GP, best_f=best_y, sampler=SobolQMCNormalSampler(1024))

    # First sampling point
    candidates, _ = optimize_acqf(acq_function=EI, bounds=bounds, q=n_points, num_restarts=num_restarts, raw_samples=raw_samples, options={"batch_limit": 5, "maxiter": 200})

    return candidates


def optimizer(DeepGP, SimpleMLP, GRU, Vec2Text, train_embed_data, int_key, max_loop, early_stop_tol, patience, iter_budget, initial_sample_num, projection_dim, num_restarts, raw_samples, seed):
    DeepGP.eval()
    SimpleMLP.eval()
    GRU.eval()
    Vec2Text.eval()

    # Producing lists of objective values to return
    true_list, GRU_list, MLP_list, V2T_list = [], [], [], []

    # Putting the GP of the GOLLuM architecture exposed and ready for use
    obj_func = DeepGP._latent_view()

    # Initial data generation
    rng = np.random.default_rng(seed)
    norm_sample = rng.standard_normal((initial_sample_num, projection_dim))
    data = np.array(train_embed_data.to_list())
    mean, std = np.mean(data, axis=0), np.std(data, axis=0)
    sample_x = torch.from_numpy(norm_sample * std + mean).to(device=DeepGP.device, dtype=DeepGP.data_type)

    prior = obj_func.posterior(sample_x)
    sample_y = prior.mean.squeeze(-1).detach().cpu()
    best_y = torch.max(sample_y)

    # Boundary of search space setting
    mins = torch.tensor(data.min(axis=0).tolist(), dtype=torch.double)
    maxs = torch.tensor(data.max(axis=0).tolist(), dtype=torch.double)
    bound = torch.stack([mins, maxs])

    # Appending current best_y to list objects
    true_list.append(best_y)

    argbest_y = torch.argmax(sample_y)
    best_embed = sample_x[argbest_y, :].unsqueeze(0).to(device=DeepGP.device, dtype=DeepGP.data_type)

    with torch.no_grad():
        best_GRU = safe_logp(convert_to_string(GRU(best_embed), id_to_token=int_key, to_smiles=True, input_data_type="one-hot"))
        best_MLP = safe_logp(convert_to_string(SimpleMLP(best_embed), id_to_token=int_key, to_smiles=True, input_data_type="one-hot"))
        best_V2T = safe_logp(convert_to_string(Vec2Text(max_loop=max_loop, latent_emb=best_embed, true_emb=best_embed, early_stop_tol=early_stop_tol, patience=patience), id_to_token=int_key, to_smiles=True, input_data_type="label"))

    GRU_list.append(best_GRU)
    MLP_list.append(best_MLP)
    V2T_list.append(best_V2T)

    # BO iteration
    for i in range(iter_budget):
        print(f"No. of optimization run: {i}")
        next_candidate = next_sample_point(sample_x=sample_x, sample_y=sample_y, best_y=best_y, bounds=bound, num_restarts=num_restarts, raw_samples=raw_samples)
        new_post = obj_func.posterior(next_candidate.double())
        true_result = new_post.mean.squeeze(-1).detach().cpu().to(sample_y.dtype)
        
        sample_x = torch.cat([sample_x, next_candidate])
        sample_y = torch.cat([sample_y, true_result.squeeze(-1)])
        best_y = sample_y.max().item()

        # Appending results to list objects
        true_list.append(best_y)

        with torch.no_grad():
            GRU_result = safe_logp(convert_to_string(GRU(next_candidate), id_to_token=int_key, to_smiles=True, input_data_type="one-hot"))
            if GRU_result > best_GRU:
                best_GRU = GRU_result
            GRU_list.append(best_GRU)

            MLP_result = safe_logp(convert_to_string(SimpleMLP(next_candidate), id_to_token=int_key, to_smiles=True, input_data_type="one-hot"))
            if MLP_result > best_MLP:
                best_MLP = MLP_result
            MLP_list.append(best_MLP)

            V2T_result = safe_logp(convert_to_string(Vec2Text(max_loop=max_loop, latent_emb=next_candidate, true_emb=next_candidate, early_stop_tol=early_stop_tol, patience=patience), id_to_token=int_key, to_smiles=True, input_data_type="label"))
            if V2T_result > best_V2T:
                best_V2T = V2T_result
            V2T_list.append(best_V2T)

    return true_list, MLP_list, GRU_list, V2T_list, list(range(iter_budget + 1))