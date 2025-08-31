"""
General utilities for token handling, hashing, and scoring (PEP 8 styled).

This module includes helpers to:
- Split and sanitize SELFIES token sequences for comparison.
- Compute token-level accuracy metrics (micro/macro) with per-sample details.
- Compare molecules using RDKit's registration hash scheme.
- Safely compute Crippen LogP from SMILES (vectorized over sequences).
- Split SELFIES into bracketed tokens.
- Filter a DataFrame by a given token vocabulary and report OOV tokens.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import selfies
from rdkit.Chem import Crippen, RegistrationHash
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.rdchem import Mol


def _split_and_clean(
    s: str,
    special_tokens: Iterable[str],
    eos_token: str = "[eos]",
    bos_token: str = "[bos]",
) -> List[str]:
    """Split a SELFIES string into tokens and sanitize for comparison.

    Drops BOS / PAD-like tokens present in ``special_tokens`` while preserving
    ``eos_token`` (so we can truncate up to it). If ``eos_token`` is present,
    tokens are truncated at its first occurrence (excluded).

    Parameters
    ----------
    s : str
        Input SELFIES string.
    special_tokens : Iterable[str]
        Tokens to drop everywhere (e.g., PAD, BOS). ``eos_token`` is kept.
    eos_token : str, default="[eos]"
        End-of-sequence token; used for truncation.
    bos_token : str, default="[bos]"
        Beginning-of-sequence token (kept only if not listed in
        ``special_tokens``).

    Returns
    -------
    list[str]
        Filtered sequence of tokens.
    """
    if s is None or not isinstance(s, str):
        return []

    toks = list(selfies.split_selfies(s))
    # Drop BOS, PAD, etc., but keep EOS for later truncation
    toks = [t for t in toks if t not in special_tokens or t == eos_token]

    # Truncate at EOS (excluded)
    if eos_token in toks:
        idx = toks.index(eos_token)
        toks = toks[:idx]

    return toks


def token_match_scores(
    y_true_selfies: Iterable[str],
    y_pred_selfies: Iterable[str],
    special_tokens: Iterable[str],
    eos_token: str = "[eos]",
    bos_token: str = "[bos]",
) -> Tuple[float, float, pd.DataFrame]:
    """Compute token-level accuracy between predicted and true SELFIES.

    Parameters
    ----------
    y_true_selfies : Iterable[str]
        Ground-truth SELFIES strings.
    y_pred_selfies : Iterable[str]
        Predicted SELFIES strings (same length as ``y_true_selfies``).
    special_tokens : Iterable[str]
        Tokens to be removed when comparing sequences.
    eos_token : str, default="[eos]"
        End-of-sequence token used to truncate sequences.
    bos_token : str, default="[bos]"
        Beginning-of-sequence token.

    Returns
    -------
    micro_acc : float
        Total correct tokens / total true tokens (0..1).
    macro_acc : float
        Mean of per-sample accuracies (0..1).
    details : pandas.DataFrame
        Per-sample breakdown including lengths, matches, and accuracy.
    """
    true_list = list(y_true_selfies)
    pred_list = list(y_pred_selfies)
    assert len(true_list) == len(pred_list), "true/pred lengths differ."

    rows: List[dict] = []
    total_correct = 0
    total_true = 0

    for i, (t, p) in enumerate(zip(true_list, pred_list)):
        t_tokens = _split_and_clean(t, special_tokens, eos_token, bos_token)
        p_tokens = _split_and_clean(p, special_tokens, eos_token, bos_token)

        len_true = len(t_tokens)
        len_cmp = min(len_true, len(p_tokens))
        correct = sum(t_tokens[j] == p_tokens[j] for j in range(len_cmp))

        acc = (correct / len_true) if len_true > 0 else np.nan  # per-sample

        rows.append(
            {
                "idx": i,
                "true_len": len_true,
                "pred_len": len(p_tokens),
                "compare_len": len_cmp,
                "correct": int(correct),
                "acc": acc,
            }
        )
        total_correct += correct
        total_true += len_true

    details = pd.DataFrame(rows)
    micro_acc = (total_correct / total_true) if total_true > 0 else np.nan
    macro_acc = details["acc"].mean()

    return micro_acc, macro_acc, details


def same_molecule_hash(m1: Mol, m2: Mol, scheme: str = "ALL_LAYERS") -> bool:
    """Compare two molecules using RDKit's registration hash scheme.

    Parameters
    ----------
    m1, m2 : rdkit.Chem.rdchem.Mol
        Molecules to compare.
    scheme : str, default="ALL_LAYERS"
        Hash scheme name under :class:`RegistrationHash.HashScheme`.

    Returns
    -------
    bool
        ``True`` if the two molecules produce the same registration hash.
    """
    layers1 = RegistrationHash.GetMolLayers(m1)
    layers2 = RegistrationHash.GetMolLayers(m2)
    hs = RegistrationHash.HashScheme
    scheme_enum = getattr(hs, scheme)
    h1 = RegistrationHash.GetMolHash(all_layers=layers1, hash_scheme=scheme_enum)
    h2 = RegistrationHash.GetMolHash(all_layers=layers2, hash_scheme=scheme_enum)
    return h1 == h2


def safe_logp(smi: Union[str, Sequence[str]]) -> Union[float, List[float]]:
    """Compute Crippen LogP from SMILES; returns NaN if parsing fails.

    Parameters
    ----------
    smi : str | Sequence[str]
        One SMILES string or a sequence of SMILES strings.

    Returns
    -------
    float | list[float]
        LogP value(s); returns ``float('nan')`` for unparseable strings.
    """
    if isinstance(smi, (list, tuple)):
        return [safe_logp(s) for s in smi]

    m = MolFromSmiles(str(smi))
    return Crippen.MolLogP(m) if m is not None else float("nan")


def split_selfies_tokens(s: str) -> List[str]:
    """Return the list of bracketed SELFIES tokens from a string.

    Example
    -------
    >>> split_selfies_tokens("[C][Branch1_1]")
    ['[C]', '[Branch1_1]']
    """
    return list(selfies.split_selfies(s))


def filter_df_by_vocab(
    df: pd.DataFrame,
    selfies_col: str,
    allowed_tokens: Set[str],
    keep_debug_cols: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter a DataFrame to rows whose SELFIES use only allowed tokens.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a column of SELFIES strings.
    selfies_col : str
        Name of the column with SELFIES strings.
    allowed_tokens : set[str]
        Vocabulary of permitted tokens.
    keep_debug_cols : bool, default=False
        If ``True``, keep the helper columns (``_tokens``, ``_oov``).

    Returns
    -------
    kept_df : pandas.DataFrame
        Rows where all tokens are in ``allowed_tokens``.
    dropped_df : pandas.DataFrame
        Rows that were removed, with a column ``oov_tokens`` listing OOVs.
    """
    work = df.copy()
    work["_tokens"] = work[selfies_col].map(split_selfies_tokens)
    work["_oov"] = work["_tokens"].map(
        lambda toks: [t for t in toks if t not in allowed_tokens]
    )

    keep_mask = work["_oov"].map(len).eq(0)
    kept_df = work.loc[keep_mask].copy()
    dropped_df = work.loc[~keep_mask, [selfies_col]].copy()
    dropped_df["oov_tokens"] = work.loc[~keep_mask, "_oov"].tolist()

    if not keep_debug_cols:
        kept_df = kept_df.drop(columns=["_tokens", "_oov"], errors="ignore")

    return kept_df, dropped_df
