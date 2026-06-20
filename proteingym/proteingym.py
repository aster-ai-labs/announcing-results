"""Tail calibration with layer consensus and attention-neighborhood residuals."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import re

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL = "AI4Protein/ProSST-2048"
AAS = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AAS)}
NULLS = {"", "WT", "W", "WILDTYPE", "WILD-TYPE", "NONE"}
SKIP = NULLS | {"P", "C", "G", "M", "R", "SYNONYMOUS"}
PREFIXES = tuple(
    "MUT: MUT= VARIANT: VARIANT= SUBSTITUTIONS: SUBSTITUTION: MUTATION: "
    "MUTATIONS: MISSENSE: AA: P. C. G. R. HGVS:".split()
)
SEPARATORS = str.maketrans({c: ":" for c in ",;/|_ +."})
MUT_RE = re.compile(r"([A-Z*])(\d+)([A-Z*])")
EPS = torch.finfo(torch.float32).tiny
CACHE_LIMIT = 32

MODEL_CACHE = {}
TOKEN_CACHE = {}
SCORE_CACHE = {}


def load(device):
    cached = MODEL_CACHE.get(device)
    if cached is None:
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(MODEL, trust_remote_code=True)
        model.cls.predictions.decoder.weight = model.prosst.embeddings.word_embeddings.weight
        cached = MODEL_CACHE[device] = (model.to(device).eval(), tok)
    return cached


def canonical_ids(tok):
    key = id(tok)
    ids = TOKEN_CACHE.get(key)
    if ids is None:
        vocab = tok.get_vocab()
        ids = TOKEN_CACHE[key] = torch.tensor([vocab[aa] for aa in AAS], dtype=torch.long)
    return ids


def ss_ids(tokens):
    return torch.tensor([[1] + [token + 3 for token in tokens] + [2]], dtype=torch.long)


def center(values, probs):
    return values - (probs * values).sum(dim=-1, keepdim=True)


def normalize(values):
    return values / values.sum(dim=-1, keepdim=True).clamp_min(EPS)


def logit(values):
    values = values.clamp_min(EPS).clamp_max(1 - EPS)
    return values.log() - (-values).log1p()


def groups(sorted_values):
    starts = torch.ones_like(sorted_values, dtype=torch.bool)
    starts[1:] = sorted_values[1:] != sorted_values[:-1]
    return starts.cumsum(dim=0) - 1


def protein_rank_residual(values):
    flat = values.reshape(-1)
    order = torch.argsort(flat)
    group_ids = groups(flat[order])
    counts = torch.bincount(group_ids, minlength=int(group_ids[-1]) + 1)
    lower = counts.cumsum(0)
    upper = counts.flip(0).cumsum(0).flip(0)
    ties = (counts[group_ids].to(values.dtype) - 1) / 2
    residual = (lower[group_ids].to(values.dtype) - ties).log() - (
        upper[group_ids].to(values.dtype) - ties
    ).log()
    ranked = torch.empty_like(flat)
    ranked[order] = residual
    return ranked.reshape_as(values)


def protein_pit_residual(values, probs):
    flat = values.reshape(-1)
    order = torch.argsort(flat)
    group_ids = groups(flat[order])
    size = int(group_ids[-1]) + 1
    weights = (probs / probs.shape[0]).reshape(-1)[order]
    masses = torch.zeros(size, dtype=values.dtype, device=values.device)
    masses.scatter_add_(0, group_ids, weights)
    counts = torch.bincount(group_ids, minlength=size)
    mass_cdf = masses.cumsum(0) - masses / 2
    rank_cdf = (counts.cumsum(0).to(values.dtype) - counts.to(values.dtype) / 2) / flat.numel()
    residual = logit(mass_cdf[group_ids]) - logit(rank_cdf[group_ids])
    ranked = torch.empty_like(flat)
    ranked[order] = residual
    return ranked.reshape_as(values)


def tail_residual(probs, ref_probs):
    lower = (probs.unsqueeze(-2) <= probs.unsqueeze(-1)).to(probs.dtype)
    tail_mass = (lower * probs.unsqueeze(-2)).sum(dim=-1).clamp_min(EPS)
    tail_rank = lower.sum(dim=-1).to(probs.dtype) / probs.shape[-1]
    return center(tail_mass.log() - tail_rank.log(), ref_probs)


def attention_context(attentions, probs):
    if probs.shape[0] == 1:
        return probs
    graph = torch.stack(attentions, dim=0)[:, 0, :, 1:-1, 1:-1].float().mean(dim=(0, 1))
    graph = graph + graph.transpose(0, 1)
    graph.fill_diagonal_(0)
    return normalize(graph) @ probs


def compute_scores(model, tok, seq, struct_tokens, device):
    batch = tok([seq], return_tensors="pt")
    aa_ids = canonical_ids(tok).to(device)
    with torch.inference_mode():
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            ss_input_ids=ss_ids(struct_tokens).to(device),
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    logits = outputs.logits[:, 1:-1].float()
    probs = torch.softmax(logits.index_select(-1, aa_ids), dim=-1)[0]
    local = tail_residual(probs, probs)

    layer_states = torch.stack([state[:, 1:-1].float() for state in outputs.hidden_states], dim=0)
    layer_readout = model.cls.predictions.transform(layer_states)
    decoder = model.cls.predictions.decoder.weight.index_select(0, aa_ids).float()
    layer_logits = torch.einsum("lbpd,ad->lbpa", layer_readout, decoder)
    layer_log_probs = torch.log_softmax(layer_logits, dim=-1)
    layer_consensus = normalize(layer_log_probs.mean(dim=0).exp())[0]
    consensus_local = tail_residual(layer_consensus, probs)

    context = attention_context(outputs.attentions, probs)
    neighbor = center(probs.clamp_min(EPS).log() - context.clamp_min(EPS).log(), probs)

    aggregate = (
        local
        + center(protein_pit_residual(local, probs), probs)
        + center(protein_rank_residual(local), probs)
        + center(protein_rank_residual(consensus_local), probs)
        + center(protein_rank_residual(neighbor), probs)
    )
    return aggregate.cpu()


def score_matrix(model, tok, seq, struct_tokens, device):
    key = (seq, tuple(struct_tokens))
    cache = SCORE_CACHE.setdefault(device, OrderedDict())
    scores = cache.get(key)
    if scores is None:
        scores = compute_scores(model, tok, seq, struct_tokens, device)
        cache[key] = scores
        if len(cache) > CACHE_LIMIT:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    return scores


@lru_cache(maxsize=262144)
def parse_mutation_string(text):
    text = text.strip().upper()
    if text in NULLS:
        return ()
    for token in PREFIXES:
        text = text.replace(token, "")
    for ch in "()[]{}\"'":
        text = text.replace(ch, "")
    text = text.translate(SEPARATORS).replace("->", "").replace(">", "").replace("=", "")
    text = text.replace("TER", "*").replace("STOP", "*")

    parsed = []
    for part in (piece.strip(":.") for piece in text.split(":") if piece.strip()):
        if len(part) >= 2 and part[1] == "." and part[0] in "PCGMR":
            part = part[2:]
        if part in SKIP:
            continue
        part = re.sub(r"[^A-Z0-9*]", "", part)
        matches = list(MUT_RE.finditer(part))
        if not matches or MUT_RE.sub("", part) or "".join(m.group(0) for m in matches) != part:
            raise ValueError(f"Malformed mutation string: {text}")
        for match in matches:
            wt, pos, mut = match.groups()
            index = int(pos) - 1
            if index < 0:
                raise ValueError(f"Malformed mutation string: {text}")
            parsed.append((wt, mut, index))
    return tuple(parsed)


def predict_fitness(target_sequence, mutation_string_list, structure_token_list, device="cuda"):
    if not mutation_string_list:
        return []
    model, tok = load(device)
    scores = score_matrix(model, tok, target_sequence, structure_token_list, device)
    n = len(target_sequence)
    out = []
    for text in mutation_string_list:
        try:
            mutations = parse_mutation_string(text)
        except ValueError:
            out.append(0.0)
            continue
        total = 0.0
        valid = bool(mutations)
        for wt, mut, pos in mutations:
            idx = AA_TO_INDEX.get(mut)
            if idx is None or pos < 0 or pos >= n or target_sequence[pos] != wt:
                valid = False
                break
            total += float(scores[pos, idx])
        out.append(total if valid else 0.0)
    return out
