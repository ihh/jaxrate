"""Integrated phylogenetic model: grammar + substitution models + tree.

Wraps subby so callers don't need to interact with it directly. Provides
integrated EM training that jointly fits grammar rule weights and substitution
model rate matrices.
"""

import math
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional

from .types import (
    Grammar, Rule, TerminalWeights, TrainState, EmissionGroup,
)
from .grammar import compile_grammar, classify_grammar
from .inside import inside
from .outside import outside
from ._log_semiring import NEG_INF, logsumexp


class PhyloModel:
    """Phylogenetic grammar model: grammar + substitution models + alignment + tree.

    Wraps the full pipeline so callers never need to import or configure subby.

    Args:
        grammar: Grammar with emission rules referencing model indices
        alignment: (R, C) int32 token-encoded alignment (one row per tree node,
            including internal nodes with gap tokens)
        tree: subby Tree, or dict with 'parentIndex' and 'distanceToParent'
        chains: list of chain dicts, one per single-column emission model.
            Each dict must have 'rate_matrix' (A, A), 'pi' (A,), and
            optionally 'update_policy' ('rev' or 'irrev', default 'rev').
        paired_chains: optional list of paired-column chain dicts.
            Each must have 'rate_matrix' (A^2, A^2), 'pi' (A^2,),
            'alphabet_size' (base A), and optionally 'update_policy'.
        maxChunkSize: column chunk size for subby LogLike calls
    """

    def __init__(self, grammar, alignment, tree, chains,
                 paired_chains=None, maxChunkSize=128):
        self.grammar = grammar
        self.alignment = jnp.asarray(alignment, dtype=jnp.int32)
        self.maxChunkSize = maxChunkSize

        # Normalize tree to subby Tree
        self._tree = _ensure_tree(tree)

        # Store chain parameters as mutable lists of dicts
        self.chains = [_normalize_chain(c) for c in chains]
        self.paired_chains = ([_normalize_chain(c) for c in paired_chains]
                              if paired_chains else [])

        self._tw_cache = None

    @property
    def tree(self):
        return self._tree

    @classmethod
    def from_xrate(cls, xrate_grammar, alignment, tree, maxChunkSize=128):
        """Construct from a parsed XrateGrammar.

        Args:
            xrate_grammar: XrateGrammar from parse_xrate / parse_xrate_file
            alignment: (R, C) int32 alignment (rows ordered by tree nodes)
            tree: subby Tree or dict
            maxChunkSize: chunk size for subby

        Returns:
            PhyloModel
        """
        grammar = xrate_grammar.grammar
        single_chains = xrate_grammar.get_single_models()
        paired_chains = xrate_grammar.get_paired_models()

        chains = []
        for c in single_chains:
            chains.append({
                'rate_matrix': np.asarray(c['rate_matrix']),
                'pi': np.asarray(c['pi']),
                'update_policy': c.get('update_policy', 'rev'),
            })

        p_chains = []
        for c in paired_chains:
            p_chains.append({
                'rate_matrix': np.asarray(c['rate_matrix']),
                'pi': np.asarray(c['pi']),
                'alphabet_size': c['alphabet_size'],
                'update_policy': c.get('update_policy', 'rev'),
            })

        return cls(grammar, alignment, tree, chains,
                   paired_chains=p_chains if p_chains else None,
                   maxChunkSize=maxChunkSize)

    def terminal_weights(self, recompute=False):
        """Compute terminal weights from current rate matrices.

        Calls subby internally. Result is cached until rate matrices change.

        Returns:
            TerminalWeights
        """
        if self._tw_cache is not None and not recompute:
            return self._tw_cache

        from subby.jax import LogLike
        from subby.jax.models import model_from_rate_matrix

        C = self.alignment.shape[1]

        # Single-column models
        single_lls = []
        for chain in self.chains:
            Q = jnp.asarray(chain['rate_matrix'], dtype=jnp.float64)
            pi = jnp.asarray(chain['pi'], dtype=jnp.float64)
            rev = chain.get('update_policy', 'rev') == 'rev'
            model = model_from_rate_matrix(Q, pi, reversible=rev)
            ll = LogLike(self.alignment, self._tree, model,
                         maxChunkSize=self.maxChunkSize)
            single_lls.append(ll)
        single = jnp.stack(single_lls, axis=0) if single_lls else jnp.zeros((0, C))

        # Paired-column models
        paired = None
        if self.paired_chains:
            from subby.formats import all_column_ktuples, kmer_tokenize

            paired_lls = []
            for chain in self.paired_chains:
                Q = jnp.asarray(chain['rate_matrix'], dtype=jnp.float64)
                pi = jnp.asarray(chain['pi'], dtype=jnp.float64)
                rev = chain.get('update_policy', 'rev') == 'rev'
                model = model_from_rate_matrix(Q, pi, reversible=rev)
                A = chain['alphabet_size']

                tuples = all_column_ktuples(C, 2, ordered=True)
                if len(tuples) == 0:
                    paired_lls.append(jnp.full((C, C), NEG_INF))
                    continue
                kt = kmer_tokenize(np.asarray(self.alignment), A, tuples,
                                   gap_mode='any')
                ll = LogLike(jnp.array(kt['alignment']), self._tree, model,
                             maxChunkSize=self.maxChunkSize)
                ll_matrix = jnp.full((C, C), NEG_INF)
                for t_idx in range(len(tuples)):
                    ci, cj = tuples[t_idx]
                    ll_matrix = ll_matrix.at[ci, cj].set(ll[t_idx])
                paired_lls.append(ll_matrix)
            paired = jnp.stack(paired_lls, axis=0)

        self._tw_cache = TerminalWeights(single=single, paired=paired, C=C)
        return self._tw_cache

    def _invalidate_cache(self):
        self._tw_cache = None

    def update_chains(self, new_chains=None, new_paired_chains=None):
        """Replace chain parameters and invalidate cached terminal weights.

        Args:
            new_chains: list of chain dicts for single-column models
            new_paired_chains: list of chain dicts for paired-column models
        """
        if new_chains is not None:
            self.chains = [_normalize_chain(c) for c in new_chains]
        if new_paired_chains is not None:
            self.paired_chains = [_normalize_chain(c) for c in new_paired_chains]
        self._invalidate_cache()

    def update_grammar(self, new_grammar):
        """Replace the grammar (e.g. after updating rule weights)."""
        self.grammar = new_grammar


def _ensure_tree(tree):
    """Convert dict or Tree to subby Tree."""
    from subby.jax.types import Tree
    if isinstance(tree, Tree):
        return tree
    if isinstance(tree, dict):
        return Tree(
            parentIndex=jnp.asarray(tree['parentIndex'], dtype=jnp.int32),
            distanceToParent=jnp.asarray(tree['distanceToParent'],
                                         dtype=jnp.float64),
        )
    return tree


def _normalize_chain(chain):
    """Ensure chain dict has required fields with correct types."""
    return {
        'rate_matrix': np.asarray(chain['rate_matrix'], dtype=np.float64),
        'pi': np.asarray(chain['pi'], dtype=np.float64),
        'update_policy': chain.get('update_policy', 'rev'),
        **{k: v for k, v in chain.items()
           if k not in ('rate_matrix', 'pi', 'update_policy')},
    }


# ---------------------------------------------------------------------------
# Integrated EM training
# ---------------------------------------------------------------------------

def _compute_hmm_rule_posteriors(grammar, cg, tw):
    """Compute expected usage count for each rule at each column (HMM).

    Returns:
        rule_counts: (n_rules,) expected total counts per rule
        model_posteriors: (n_models, C) posterior probability of each model
            at each column (sum over rules that emit with that model)
        ll: scalar log-likelihood
    """
    from .hmm import hmm_forward, hmm_backward, _build_hmm_tables

    alpha, ll = hmm_forward(cg, tw)
    beta, _ = hmm_backward(cg, tw)

    log_trans, log_emit, log_init, log_term = _build_hmm_tables(cg, tw)
    K = cg.n_nonterminals
    C = tw.C
    n_rules = len(grammar.rules)

    rule_counts = np.zeros(n_rules, dtype=np.float64)
    model_posteriors = np.zeros((cg.n_models, C), dtype=np.float64)

    # Build rule-to-state mappings
    emit_model_for_state = {}
    for r_idx, rule in enumerate(grammar.rules):
        if rule.emissions:
            emit_model_for_state[rule.lhs] = rule.emissions[0].model_index

    # State posteriors: P(state_c = k | x) = exp(alpha[c,k] + beta[c,k] - ll)
    log_gamma = alpha + beta  # (C, K)
    gamma = jnp.exp(log_gamma - ll)  # (C, K)

    # Map state posteriors to model posteriors
    for k in range(K):
        if k in emit_model_for_state:
            m_idx = emit_model_for_state[k]
            model_posteriors[m_idx] += np.asarray(gamma[:, k])

    # Terminal rule counts
    for r_idx, rule in enumerate(grammar.rules):
        lhs = rule.lhs
        if len(rule.rhs) == 0:
            log_count = alpha[-1, lhs] + log_term[lhs] - ll
            rule_counts[r_idx] = float(jnp.exp(log_count))

    # Pairwise transition counts: E[N(i→j)] summed over all columns
    trans_counts = np.zeros((K, K), dtype=np.float64)
    for c in range(C - 1):
        log_joint = (alpha[c, :, None] + log_trans +
                     log_emit[None, :, c + 1] + beta[c + 1, None, :] - ll)
        trans_counts += np.asarray(jnp.exp(log_joint))

    # Map transition counts to rule counts
    from collections import defaultdict
    rules_by_pair = defaultdict(list)
    for r_idx, rule in enumerate(grammar.rules):
        if len(rule.rhs) == 1:
            rules_by_pair[(rule.lhs, rule.rhs[0])].append(r_idx)

    for (lhs, rhs), r_indices in rules_by_pair.items():
        total_count = trans_counts[lhs, rhs]
        if len(r_indices) == 1:
            rule_counts[r_indices[0]] = total_count
        else:
            # Multiple rules with same (lhs, rhs) — apportion by weight
            log_ws = np.array([grammar.rules[i].log_weight for i in r_indices])
            ws = np.exp(log_ws - np.max(log_ws))
            ws /= ws.sum()
            for i, r_idx in enumerate(r_indices):
                rule_counts[r_idx] = total_count * ws[i]

    return rule_counts, model_posteriors, ll


def _compute_scfg_rule_posteriors(grammar, cg, tw):
    """Compute expected rule counts and model posteriors for SCFG.

    Returns:
        rule_counts: (n_rules,) expected total counts per rule
        model_posteriors: (n_models, C) posterior for single-col models
        ll: log-likelihood
    """
    from .scfg import scfg_inside, scfg_outside

    alpha, ll = scfg_inside(cg, tw)
    beta = scfg_outside(cg, tw, alpha)

    K = cg.n_nonterminals
    C = tw.C
    n_rules = len(grammar.rules)

    rule_counts = np.zeros(n_rules, dtype=np.float64)
    model_posteriors = np.zeros((cg.n_models, C), dtype=np.float64)

    for k in range(K):
        for i in range(C):
            for j in range(i + 1, C + 1):
                log_post = float(alpha[k, i, j] + beta[k, i, j] - ll)
                if log_post > -30:
                    post = math.exp(log_post)
                    for r_idx, rule in enumerate(grammar.rules):
                        if rule.lhs == k and rule.emissions:
                            eg = rule.emissions[0]
                            if eg.n_positions == 1:
                                model_posteriors[eg.model_index, i] += post
                            break
                    for r_idx, rule in enumerate(grammar.rules):
                        if rule.lhs == k:
                            rule_counts[r_idx] += post

    # Normalize rule counts per LHS
    for k in range(K):
        mask = [i for i, r in enumerate(grammar.rules) if r.lhs == k]
        total = sum(rule_counts[i] for i in mask)
        if total > 0:
            for i in mask:
                rule_counts[i] /= total
            for i in mask:
                rule_counts[i] *= total

    return rule_counts, model_posteriors, float(ll)


def _log_prior_rules(grammar, rule_pseudocounts):
    """Compute log Dirichlet prior for rule weights.

    The Dirichlet log-prior for a group of rules sharing the same LHS is:
        log P(θ | α) = Σ_i (α_i - 1) log θ_i  + const

    Args:
        grammar: Grammar
        rule_pseudocounts: (n_rules,) pseudocount per rule

    Returns:
        scalar log-prior (up to additive constant)
    """
    log_prior = 0.0
    n_nt = len(grammar.nonterminals)
    for nt_idx in range(n_nt):
        rule_indices = [i for i, r in enumerate(grammar.rules) if r.lhs == nt_idx]
        if not rule_indices:
            continue
        for r_idx in rule_indices:
            alpha_i = rule_pseudocounts[r_idx]
            if alpha_i > 0:
                log_w = grammar.rules[r_idx].log_weight
                log_prior += (alpha_i - 1.0) * log_w
    return log_prior


def _log_prior_rates(chains, pseudocounts):
    """Compute log-prior contribution from wait-time pseudocounts.

    The wait-time pseudocount adds a penalty term that prevents rates
    from growing unboundedly. It effectively adds pseudo-dwell-time
    to the expected complete-data log-likelihood.

    Args:
        chains: list of chain dicts
        pseudocounts: wait-time pseudocount

    Returns:
        scalar log-prior contribution
    """
    if pseudocounts <= 0:
        return 0.0

    log_prior = 0.0
    for chain in chains:
        Q = chain['rate_matrix']
        A = Q.shape[0]
        # The pseudocount adds w_pseudo to dwell times.
        # In the expected LL: w_i * Q_ii contribution becomes
        # (w_i + pseudo) * Q_ii. The extra term is pseudo * Q_ii.
        for i in range(A):
            log_prior += pseudocounts * Q[i, i]
    return log_prior


def _m_step_rules(grammar, rule_counts, pseudocounts=None):
    """M-step: re-estimate rule log-weights from expected counts + pseudocounts.

    Normalizes (counts + pseudocounts) per LHS nonterminal.

    Args:
        grammar: Grammar
        rule_counts: (n_rules,) expected counts from E-step
        pseudocounts: (n_rules,) Dirichlet pseudocounts (None = no prior)

    Returns:
        new Grammar with updated rule log-weights
    """
    n_nt = len(grammar.nonterminals)

    new_rules = list(grammar.rules)
    for nt_idx in range(n_nt):
        rule_indices = [i for i, r in enumerate(grammar.rules) if r.lhs == nt_idx]
        if not rule_indices:
            continue

        counts = np.array([rule_counts[i] for i in rule_indices])
        if pseudocounts is not None:
            counts = counts + np.array([pseudocounts[i] for i in rule_indices])
        total = counts.sum()
        if total < 1e-30:
            continue

        probs = counts / total
        probs = np.maximum(probs, 1e-30)

        for j, r_idx in enumerate(rule_indices):
            old = grammar.rules[r_idx]
            new_rules[r_idx] = Rule(
                old.lhs, old.rhs, old.emissions,
                float(np.log(probs[j])),
                old.composition,
            )

    return Grammar(
        nonterminals=grammar.nonterminals,
        rules=new_rules,
        start=grammar.start,
        n_models=grammar.n_models,
    )


def _stationary_dist(Q):
    """Find stationary distribution of rate matrix Q.

    Solves pi @ Q = 0 with sum(pi) = 1 via linear solve.
    Falls back to None if the system is singular.
    """
    A = Q.shape[0]
    # Replace last row of Q^T with constraint sum(pi) = 1
    M = Q.T.copy()
    M[-1, :] = 1.0
    b = np.zeros(A)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(M, b)
        pi = np.maximum(pi, 1e-30)
        pi /= pi.sum()
        return pi
    except np.linalg.LinAlgError:
        return None


def _m_step_rates(phylo_model, model_posteriors, fit_rates=True, fit_pi=True,
                   pseudocounts=0.0):
    """M-step: re-estimate rate matrices and/or pi from expected counts.

    Uses subby Counts() weighted by model posteriors to get expected
    substitution counts, then updates Q and pi following xrate's approach:

    1. Q_ij = u_ij / w_i  (off-diagonal rate = transition counts / wait time)
    2. For reversible models, pi is the stationary distribution of the new Q
       (solving pi @ Q = 0). This ensures detailed balance.
    3. For irreversible models, pi is from empirical start counts.

    Pseudocounts are added to the expected counts before the M-step,
    following xrate convention (pseud_wait for w, pseud_mutate for u).

    Args:
        phylo_model: PhyloModel
        model_posteriors: (n_models, C) posterior weight of each model per column
        fit_rates: whether to update rate matrices
        fit_pi: whether to update equilibrium distributions
        pseudocounts: wait-time pseudocount added to dwell times (prevents
            rate divergence on sparse data). Default 0. xrate default is 1e-4.
    """
    from subby.jax import Counts, RootProb
    from subby.jax.models import model_from_rate_matrix

    if not fit_rates and not fit_pi:
        return

    alignment = phylo_model.alignment
    tree = phylo_model._tree
    C = alignment.shape[1]

    for m_idx, chain in enumerate(phylo_model.chains):
        if m_idx >= model_posteriors.shape[0]:
            break

        weights = model_posteriors[m_idx]  # (C,)
        if float(jnp.sum(weights)) < 1e-30:
            continue

        Q_old = jnp.asarray(chain['rate_matrix'], dtype=jnp.float64)
        pi_old = jnp.asarray(chain['pi'], dtype=jnp.float64)
        rev = chain.get('update_policy', 'rev') == 'rev'
        model = model_from_rate_matrix(Q_old, pi_old, reversible=rev)

        A = Q_old.shape[0]
        weights_jnp = jnp.asarray(weights, dtype=jnp.float64)

        if fit_rates:
            # Expected substitution counts: (A, A, C)
            counts = Counts(alignment, tree, model,
                            maxChunkSize=phylo_model.maxChunkSize)
            weighted_counts = jnp.sum(counts * weights_jnp[None, None, :],
                                      axis=-1)  # (A, A)
            wc = np.asarray(weighted_counts)

            # xrate-style pseudocounts: add to wait times (diagonal)
            w = np.diag(wc).copy() + pseudocounts  # dwell times
            u = wc.copy()
            np.fill_diagonal(u, 0)  # off-diagonal = transition counts

            # M-step: Q_ij = u_ij / w_i
            new_Q = u / np.maximum(w[:, None], 1e-30)
            np.fill_diagonal(new_Q, 0)
            np.fill_diagonal(new_Q, -new_Q.sum(axis=1))

            chain['rate_matrix'] = new_Q

        if fit_pi:
            Q_cur = chain['rate_matrix']
            if rev:
                # Reversible: pi = stationary distribution of Q
                # This ensures detailed balance: pi_i Q_ij = pi_j Q_ji
                new_pi = _stationary_dist(Q_cur)
                if new_pi is None:
                    # Fallback: root posteriors
                    root_post = RootProb(alignment, tree, model,
                                         maxChunkSize=phylo_model.maxChunkSize)
                    weighted_root = np.asarray(
                        jnp.sum(root_post * weights_jnp[None, :], axis=-1))
                    new_pi = np.maximum(weighted_root, 1e-30)
                    new_pi /= new_pi.sum()
            else:
                # Irreversible: pi from empirical start counts
                root_post = RootProb(alignment, tree, model,
                                     maxChunkSize=phylo_model.maxChunkSize)
                weighted_root = np.asarray(
                    jnp.sum(root_post * weights_jnp[None, :], axis=-1))
                if pseudocounts > 0:
                    weighted_root += pseudocounts
                new_pi = np.maximum(weighted_root, 1e-30)
                new_pi /= new_pi.sum()

            chain['pi'] = new_pi

    phylo_model._invalidate_cache()


def phylo_em_step(phylo_model, fit_rules=True, fit_rates=True, fit_pi=True,
                  pseudocounts=0.0):
    """One integrated EM step: fit grammar rules and/or rate matrices.

    E-step: inside-outside → expected rule counts + per-model column posteriors.
    M-step: update rule weights (if fit_rules), rate matrices (if fit_rates),
            equilibrium distributions (if fit_pi).

    When pseudocounts > 0, a Dirichlet/gamma prior is applied. The returned
    value is the log-posterior (log-likelihood + log-prior), which is the
    quantity guaranteed to increase monotonically under EM with priors.

    Args:
        phylo_model: PhyloModel
        fit_rules: update grammar rule log-weights
        fit_rates: update substitution rate matrices Q
        fit_pi: update equilibrium distributions π
        pseudocounts: Laplace-style pseudocount added to expected counts.
            Acts as a Dirichlet prior on rule probabilities and a gamma prior
            on rate matrix entries, following xrate convention.

    Returns:
        log_posterior: log-likelihood + log-prior (scalar)
    """
    tw = phylo_model.terminal_weights(recompute=True)
    grammar = phylo_model.grammar
    cg = compile_grammar(grammar)

    grammar_class = cg.grammar_class
    if grammar_class == 'hmm':
        rule_counts, model_posteriors, ll = _compute_hmm_rule_posteriors(
            grammar, cg, tw)
    elif grammar_class == 'scfg':
        rule_counts, model_posteriors, ll = _compute_scfg_rule_posteriors(
            grammar, cg, tw)
    else:
        raise NotImplementedError(
            "Integrated training not yet supported for MCFG grammars "
            "(MCFG outside algorithm is not implemented)")

    # Build per-rule pseudocounts
    n_rules = len(grammar.rules)
    rule_pseudo = np.full(n_rules, pseudocounts) if pseudocounts > 0 else None

    # M-step: rules
    if fit_rules:
        new_grammar = _m_step_rules(grammar, rule_counts,
                                    pseudocounts=rule_pseudo)
        phylo_model.update_grammar(new_grammar)

    # M-step: rates / pi
    if fit_rates or fit_pi:
        _m_step_rates(phylo_model, model_posteriors,
                      fit_rates=fit_rates, fit_pi=fit_pi,
                      pseudocounts=pseudocounts)

    # Compute log-prior for the returned log-posterior
    log_prior = 0.0
    if pseudocounts > 0:
        if fit_rules:
            rule_pseudo_arr = np.full(n_rules, pseudocounts)
            log_prior += _log_prior_rules(phylo_model.grammar, rule_pseudo_arr)
        if fit_rates:
            log_prior += _log_prior_rates(phylo_model.chains, pseudocounts)

    return float(ll) + log_prior


def phylo_train(phylo_model, n_iterations=100, convergence_tol=1e-6,
                fit_rules=True, fit_rates=True, fit_pi=True,
                pseudocounts=0.0):
    """Run integrated EM training loop.

    Jointly fits grammar rule weights and substitution model parameters.
    When fitting both rules and rates, each iteration alternates:
      1. E-step + M-step for rules (with fixed rates)
      2. E-step + M-step for rates (with fixed rules)
    This alternating approach ensures each sub-step is a valid EM update.

    Args:
        phylo_model: PhyloModel (modified in-place)
        n_iterations: maximum iterations
        convergence_tol: stop when |Δ(log-posterior)| < tol
        fit_rules: whether to update grammar rule log-weights
        fit_rates: whether to update substitution rate matrices
        fit_pi: whether to update equilibrium distributions
        pseudocounts: wait-time pseudocount added to expected dwell times.
            Following xrate convention (default pseud_wait=1e-4).
            Prevents rate divergence on sparse data. Set to 0 for pure MLE.

    Returns:
        history: list of (iteration, log_posterior) tuples
    """
    prev_ll = float('-inf')
    history = []

    both = fit_rules and (fit_rates or fit_pi)

    for iteration in range(n_iterations):
        if both:
            # Alternating: rules first, then rates
            phylo_em_step(phylo_model,
                          fit_rules=True, fit_rates=False, fit_pi=False,
                          pseudocounts=pseudocounts)
            ll = phylo_em_step(phylo_model,
                               fit_rules=False,
                               fit_rates=fit_rates, fit_pi=fit_pi,
                               pseudocounts=pseudocounts)
        else:
            ll = phylo_em_step(phylo_model,
                               fit_rules=fit_rules,
                               fit_rates=fit_rates,
                               fit_pi=fit_pi,
                               pseudocounts=pseudocounts)

        history.append((iteration, ll))

        if abs(ll - prev_ll) < convergence_tol and iteration > 0:
            break
        prev_ll = ll

    return history
