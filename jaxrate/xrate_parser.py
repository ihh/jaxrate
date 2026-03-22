"""Parser for xrate .eg grammar files (S-expression format).

Parses xrate grammars from the dart project and converts them to
jaxrate Grammar + subby-compatible substitution model specifications.

Reference: https://github.com/ihh/dart/blob/master/doc/XrateFormat.txt
"""

import copy
import math
import re
from typing import Optional

import jax.numpy as jnp
import numpy as np

from .types import Grammar, EmissionGroup, Rule
from .grammar import GrammarBuilder


# ---------------------------------------------------------------------------
# S-expression tokenizer & parser
# ---------------------------------------------------------------------------

def _tokenize(text):
    """Tokenize S-expression text, stripping comments."""
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == ';':
            # Skip to end of line
            while i < n and text[i] != '\n':
                i += 1
            continue
        if c in '()':
            tokens.append(c)
            i += 1
        elif c in ' \t\n\r':
            i += 1
        else:
            # Atom: read until whitespace or paren
            j = i
            while j < n and text[j] not in ' \t\n\r()':
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def _parse_tokens(tokens, pos=0):
    """Parse tokens into nested lists. Returns (expr, new_pos)."""
    if pos >= len(tokens):
        raise ValueError("Unexpected end of input")
    if tokens[pos] == '(':
        lst = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ')':
            val, pos = _parse_tokens(tokens, pos)
            lst.append(val)
        if pos >= len(tokens):
            raise ValueError("Missing closing parenthesis")
        pos += 1  # skip ')'
        return lst, pos
    elif tokens[pos] == ')':
        raise ValueError("Unexpected ')'")
    else:
        atom = tokens[pos]
        # Try to convert to number
        try:
            return int(atom), pos + 1
        except ValueError:
            try:
                return float(atom), pos + 1
            except ValueError:
                return atom, pos + 1


def parse_sexpr(text):
    """Parse S-expression text into nested Python lists.

    Returns a list of top-level expressions.
    """
    tokens = _tokenize(text)
    results = []
    pos = 0
    while pos < len(tokens):
        expr, pos = _parse_tokens(tokens, pos)
        results.append(expr)
    return results


# ---------------------------------------------------------------------------
# Macro expander (xrate preprocessor)
# ---------------------------------------------------------------------------
# Three-pass expansion matching the C++ svisitor pipeline:
#   Pass 1 (preorder):  &define, &foreach-integer, &foreach-token, &foreach
#   Pass 2 (postorder): &cat/&., &sum/&+, &sub/&-, &mul/&*, &div/&/,
#                        &mod/&%, &if/&?, &eq/&=, &neq/&!=, &gt/&>,
#                        &lt/&<, &geq/&>=, &leq/&<=, &and, &or, &not,
#                        &int, &chr, &ord
#   Pass 3:             &scheme (stub — raises for unsupported expressions)

# Operator aliases (short → canonical)
_OP_ALIASES = {
    '&.': '&cat', '&+': '&sum', '&-': '&sub', '&*': '&mul',
    '&/': '&div', '&%': '&mod', '&?': '&if', '&=': '&eq',
    '&!=': '&neq', '&>': '&gt', '&<': '&lt', '&>=': '&geq',
    '&<=': '&leq',
}


def _deep_copy(expr):
    """Deep copy an S-expression tree (nested lists + atoms)."""
    if isinstance(expr, list):
        return [_deep_copy(x) for x in expr]
    return expr


def _substitute(expr, var, value):
    """Replace all occurrences of atom `var` with `value` in expr."""
    if isinstance(expr, list):
        return [_substitute(x, var, value) for x in expr]
    if expr == var:
        return _deep_copy(value)
    return expr


def _substitute_multi(expr, bindings):
    """Replace multiple atom→value bindings in expr."""
    if isinstance(expr, list):
        return [_substitute_multi(x, bindings) for x in expr]
    if expr in bindings:
        return _deep_copy(bindings[expr])
    return expr


def _flatten_splices(lst):
    """Flatten any _Splice markers from foreach expansion."""
    result = []
    for item in lst:
        if isinstance(item, _Splice):
            result.extend(item.items)
        elif isinstance(item, list):
            result.append(_flatten_splices(item))
        else:
            result.append(item)
    return result


class _Splice:
    """Marker for items that should be spliced into the parent list."""
    __slots__ = ('items',)
    def __init__(self, items):
        self.items = items


def _pass1_preorder(expr, defines=None, alphabet_tokens=None):
    """Pass 1: expand &define, &foreach-integer, &foreach-token, &foreach.

    Operates top-down (preorder). &define adds to the defines dict.
    &foreach* expands by cloning the template body for each iteration value.

    Args:
        expr: S-expression (nested lists + atoms)
        defines: dict of atom → replacement value (mutated in place)
        alphabet_tokens: list of alphabet token strings (for &foreach-token, &TOKENS)

    Returns:
        Expanded expression, or _Splice for multi-element expansions.
    """
    if defines is None:
        defines = {}
    if alphabet_tokens is None:
        alphabet_tokens = []

    # Atom: substitute defines and &TOKENS
    if not isinstance(expr, list):
        if expr == '&TOKENS':
            return len(alphabet_tokens)
        if expr in defines:
            return _deep_copy(defines[expr])
        return expr

    if len(expr) == 0:
        return expr

    head = expr[0]

    # &define VAR VALUE — store binding, return empty splice
    if head == '&define' and len(expr) == 3:
        var = expr[1]
        val = _pass1_preorder(expr[2], defines, alphabet_tokens)
        defines[var] = val
        return _Splice([])

    # &foreach-integer VAR (LO HI) BODY...
    if head == '&foreach-integer' and len(expr) >= 4:
        var = expr[1]
        range_expr = _pass1_preorder(expr[2], defines, alphabet_tokens)
        lo = int(range_expr[0]) if isinstance(range_expr, list) else int(range_expr)
        hi = int(range_expr[1]) if isinstance(range_expr, list) else int(range_expr)
        body_templates = expr[3:]
        results = []
        for i in range(lo, hi + 1):
            child_defines = dict(defines)
            child_defines[var] = i
            for tmpl in body_templates:
                expanded = _pass1_preorder(
                    _substitute(tmpl, var, i), child_defines, alphabet_tokens)
                if isinstance(expanded, _Splice):
                    results.extend(expanded.items)
                else:
                    results.append(expanded)
        return _Splice(results)

    # &foreach-token VAR BODY...
    if head == '&foreach-token' and len(expr) >= 3:
        var = expr[1]
        body_templates = expr[2:]
        results = []
        for tok in alphabet_tokens:
            child_defines = dict(defines)
            child_defines[var] = tok
            for tmpl in body_templates:
                expanded = _pass1_preorder(
                    _substitute(tmpl, var, tok), child_defines, alphabet_tokens)
                if isinstance(expanded, _Splice):
                    results.extend(expanded.items)
                else:
                    results.append(expanded)
        return _Splice(results)

    # &foreach VAR (LIST...) BODY...
    if head == '&foreach' and len(expr) >= 4:
        var = expr[1]
        values_expr = _pass1_preorder(expr[2], defines, alphabet_tokens)
        if not isinstance(values_expr, list):
            values_expr = [values_expr]
        body_templates = expr[3:]
        results = []
        for val in values_expr:
            child_defines = dict(defines)
            child_defines[var] = val
            for tmpl in body_templates:
                expanded = _pass1_preorder(
                    _substitute(tmpl, var, val), child_defines, alphabet_tokens)
                if isinstance(expanded, _Splice):
                    results.extend(expanded.items)
                else:
                    results.append(expanded)
        return _Splice(results)

    # Regular list: recurse into children, then flatten splices
    result = []
    for child in expr:
        expanded = _pass1_preorder(child, defines, alphabet_tokens)
        if isinstance(expanded, _Splice):
            result.extend(expanded.items)
        else:
            result.append(expanded)
    return result


def _to_num(x):
    """Coerce an atom to a number (int or float)."""
    if isinstance(x, (int, float)):
        return x
    try:
        return int(x)
    except (ValueError, TypeError):
        try:
            return float(x)
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert {x!r} to number")


def _pass2_postorder(expr):
    """Pass 2: evaluate arithmetic, concatenation, and conditionals.

    Operates bottom-up (postorder). Each operator consumes its arguments
    from the same list.
    """
    if not isinstance(expr, list):
        return expr

    # First recurse into children
    expr = [_pass2_postorder(child) for child in expr]

    if len(expr) == 0:
        return expr

    head = expr[0]
    # Resolve alias
    if isinstance(head, str):
        head = _OP_ALIASES.get(head, head)

    # &cat / &. — string concatenation
    if head == '&cat':
        return ''.join(str(x) for x in expr[1:])

    # &sum / &+ — addition
    if head == '&sum':
        return sum(_to_num(x) for x in expr[1:])

    # &sub / &- — subtraction
    if head == '&sub' and len(expr) >= 3:
        result = _to_num(expr[1])
        for x in expr[2:]:
            result -= _to_num(x)
        return result

    # &mul / &* — multiplication
    if head == '&mul':
        result = 1
        for x in expr[1:]:
            result *= _to_num(x)
        return result

    # &div / &/ — division
    if head == '&div' and len(expr) >= 3:
        result = _to_num(expr[1])
        for x in expr[2:]:
            result = result / _to_num(x)
        return result

    # &mod / &% — modulus
    if head == '&mod' and len(expr) == 3:
        return _to_num(expr[1]) % _to_num(expr[2])

    # &eq / &= — equality
    if head == '&eq' and len(expr) == 3:
        return 1 if expr[1] == expr[2] else 0

    # &neq / &!= — inequality
    if head == '&neq' and len(expr) == 3:
        return 1 if expr[1] != expr[2] else 0

    # &gt / &> — greater than
    if head == '&gt' and len(expr) == 3:
        return 1 if _to_num(expr[1]) > _to_num(expr[2]) else 0

    # &lt / &< — less than
    if head == '&lt' and len(expr) == 3:
        return 1 if _to_num(expr[1]) < _to_num(expr[2]) else 0

    # &geq / &>= — greater or equal
    if head == '&geq' and len(expr) == 3:
        return 1 if _to_num(expr[1]) >= _to_num(expr[2]) else 0

    # &leq / &<= — less or equal
    if head == '&leq' and len(expr) == 3:
        return 1 if _to_num(expr[1]) <= _to_num(expr[2]) else 0

    # &if / &? — conditional (test, then, else)
    if head == '&if' and len(expr) == 4:
        test = expr[1]
        if isinstance(test, (int, float)):
            cond = test != 0
        elif isinstance(test, str):
            cond = test not in ('0', '', 'false')
        else:
            cond = bool(test)
        return expr[2] if cond else expr[3]

    # &and — logical and
    if head == '&and':
        return 1 if all(_to_num(x) != 0 for x in expr[1:]) else 0

    # &or — logical or
    if head == '&or':
        return 1 if any(_to_num(x) != 0 for x in expr[1:]) else 0

    # &not — logical not
    if head == '&not' and len(expr) == 2:
        return 0 if _to_num(expr[1]) != 0 else 1

    # &int — convert to integer
    if head == '&int' and len(expr) == 2:
        return int(_to_num(expr[1]))

    # &chr — integer to character
    if head == '&chr' and len(expr) == 2:
        return chr(int(_to_num(expr[1])))

    # &ord — character to integer
    if head == '&ord' and len(expr) == 2:
        val = expr[1]
        if isinstance(val, str) and len(val) == 1:
            return ord(val)
        return int(_to_num(val))

    return expr


def _pass3_scheme(expr):
    """Pass 3: handle &scheme directives (stub).

    Raises NotImplementedError for &scheme expressions.
    Silently discards &scheme-discard.
    """
    if not isinstance(expr, list):
        return expr

    if len(expr) > 0:
        head = expr[0]
        if head == '&scheme':
            raise NotImplementedError(
                "Guile Scheme evaluation (&scheme) is not supported. "
                "Use pre-expanded grammar files instead.")
        if head == '&scheme-discard':
            return _Splice([])

    result = []
    for child in expr:
        expanded = _pass3_scheme(child)
        if isinstance(expanded, _Splice):
            result.extend(expanded.items)
        else:
            result.append(expanded)
    return result


def _extract_alphabet_tokens(exprs):
    """Extract alphabet tokens from parsed S-expressions (pre-expansion).

    Scans for (alphabet ... (token (a c g u)) ...) to provide token list
    for &foreach-token and &TOKENS before full expansion.
    """
    for expr in exprs:
        if isinstance(expr, list) and len(expr) > 0 and expr[0] == 'alphabet':
            tok_expr = _find1(expr, 'token')
            if tok_expr is not None:
                if isinstance(tok_expr[1], list):
                    return [str(t) for t in tok_expr[1]]
                else:
                    return [str(t) for t in tok_expr[1:]]
    return []


def expand_macros(exprs, alphabet_tokens=None):
    """Expand xrate macro directives in parsed S-expressions.

    Implements the three-pass xrate macro expansion pipeline:
      1. &define, &foreach-integer, &foreach-token, &foreach (preorder)
      2. &cat, &sum, &sub, &mul, &div, &if, &eq, etc. (postorder)
      3. &scheme (stub — raises if encountered)

    Args:
        exprs: list of top-level S-expressions (from parse_sexpr)
        alphabet_tokens: optional list of alphabet token strings.
            If None, extracted automatically from (alphabet ...) block.

    Returns:
        Expanded list of S-expressions.
    """
    if alphabet_tokens is None:
        alphabet_tokens = _extract_alphabet_tokens(exprs)

    # Wrap in a container list for uniform processing
    container = ['__top__'] + exprs

    # Pass 1: preorder macro expansion
    defines = {}
    container = _pass1_preorder(container, defines, alphabet_tokens)
    if isinstance(container, _Splice):
        container = ['__top__'] + container.items
    container = _flatten_splices(container)

    # Pass 2: postorder arithmetic/concatenation
    container = _pass2_postorder(container)
    if not isinstance(container, list):
        container = [container]

    # Pass 3: scheme stubs
    container = _pass3_scheme(container)
    if isinstance(container, _Splice):
        container = ['__top__'] + container.items
    container = _flatten_splices(container)

    # Unwrap
    if isinstance(container, list) and len(container) > 0 and container[0] == '__top__':
        return container[1:]
    return container if isinstance(container, list) else [container]


# ---------------------------------------------------------------------------
# Helper to extract named fields from S-expression lists
# ---------------------------------------------------------------------------

def _find(sexpr, tag):
    """Find all sub-expressions starting with `tag` in an S-expr list."""
    return [item for item in sexpr if isinstance(item, list) and len(item) > 0
            and item[0] == tag]


def _find1(sexpr, tag, default=None):
    """Find first sub-expression starting with `tag`, or default."""
    matches = _find(sexpr, tag)
    return matches[0] if matches else default


def _get_value(sexpr, tag, default=None):
    """Get the value (second element) of a (tag value) pair."""
    match = _find1(sexpr, tag)
    if match is None:
        return default
    return match[1] if len(match) > 1 else default


# ---------------------------------------------------------------------------
# Alphabet extraction
# ---------------------------------------------------------------------------

def _parse_alphabet(alpha_sexpr):
    """Parse an (alphabet ...) S-expression.

    Returns:
        dict with 'name', 'tokens' (list of str), 'complement' (dict or None)
    """
    name = _get_value(alpha_sexpr, 'name', 'unknown')

    tok_expr = _find1(alpha_sexpr, 'token')
    if tok_expr is None:
        raise ValueError("Alphabet has no (token ...) definition")
    # (token (a c g u)) — the tokens are in the second element (a list)
    if isinstance(tok_expr[1], list):
        tokens = [str(t) for t in tok_expr[1]]
    else:
        tokens = [str(t) for t in tok_expr[1:]]

    comp_expr = _find1(alpha_sexpr, 'complement')
    complement = None
    if comp_expr is not None:
        if isinstance(comp_expr[1], list):
            comp_tokens = [str(t) for t in comp_expr[1]]
        else:
            comp_tokens = [str(t) for t in comp_expr[1:]]
        complement = dict(zip(tokens, comp_tokens))

    return {'name': name, 'tokens': tokens, 'complement': complement}


# ---------------------------------------------------------------------------
# Parametric expression evaluator
# ---------------------------------------------------------------------------

def _parse_params(grammar_sexpr):
    """Extract parameter bindings from (pgroup ...) and (rate ...) blocks.

    Returns:
        dict mapping parameter name → float value
    """
    params = {}

    # (pgroup ((p1 val1) (p2 val2)) ((p3 val3) (p4 val4)) ...)
    for pg in _find(grammar_sexpr, 'pgroup'):
        for group in pg[1:]:
            if isinstance(group, list):
                for item in group:
                    if isinstance(item, list) and len(item) == 2:
                        name, val = item
                        if isinstance(name, str):
                            try:
                                params[name] = float(val)
                            except (ValueError, TypeError):
                                pass

    # (rate (R val) (lambda val) ...)
    for rg in _find(grammar_sexpr, 'rate'):
        for item in rg[1:]:
            if isinstance(item, list) and len(item) == 2:
                name, val = item
                if isinstance(name, str):
                    try:
                        params[name] = float(val)
                    except (ValueError, TypeError):
                        pass

    return params


def _eval_parametric(expr, params):
    """Evaluate a parametric expression using parameter bindings.

    Handles:
        - numeric literals: 0.5 → 0.5
        - parameter names: stayProb → params['stayProb']
        - infix arithmetic: (a * b), (a / b), (a + b), (a - b)
        - nested S-expr: (param) → params['param']

    Args:
        expr: a value from prob/rate field — number, string, or list
        params: dict of parameter name → float

    Returns:
        float value, or None if unevaluable
    """
    if isinstance(expr, (int, float)):
        return float(expr)

    if isinstance(expr, str):
        if expr in params:
            return params[expr]
        try:
            return float(expr)
        except ValueError:
            return None

    if isinstance(expr, list):
        # Empty list
        if len(expr) == 0:
            return None

        # Single element: (param_name) → resolve
        if len(expr) == 1:
            return _eval_parametric(expr[0], params)

        # Infix binary: (a OP b) or (a OP b OP c ...)
        # Process left-to-right: a OP1 b OP2 c → ((a OP1 b) OP2 c)
        if len(expr) >= 3 and isinstance(expr[1], str) and expr[1] in ('*', '/', '+', '-'):
            result = _eval_parametric(expr[0], params)
            if result is None:
                return None
            i = 1
            while i < len(expr) - 1:
                op = expr[i]
                right = _eval_parametric(expr[i + 1], params)
                if right is None:
                    return None
                if op == '*':
                    result *= right
                elif op == '/':
                    result = result / right if right != 0 else 0.0
                elif op == '+':
                    result += right
                elif op == '-':
                    result -= right
                i += 2
            return result

        # Two elements without operator: try first as value
        if len(expr) == 2:
            v = _eval_parametric(expr[0], params)
            if v is not None:
                return v

    return None


# ---------------------------------------------------------------------------
# Chain (substitution model) extraction
# ---------------------------------------------------------------------------

def _parse_chain(chain_sexpr, alphabet_tokens, params=None):
    """Parse a (chain ...) S-expression into a substitution model spec.

    Args:
        chain_sexpr: the (chain ...) S-expression
        alphabet_tokens: list of alphabet token strings
        params: dict of parametric parameter bindings (from pgroup/rate)

    Returns:
        dict with:
            'terminals': list of terminal names (e.g. ['NUC'] or ['LNUC', 'RNUC'])
            'n_positions': 1 for single, 2 for paired
            'update_policy': 'rev' or 'irrev'
            'pi': (A,) or (A^2,) numpy array — initial distribution
            'rate_matrix': (A, A) or (A^2, A^2) numpy array — rate matrix Q
            'alphabet_size': A (base alphabet size, e.g. 4 for RNA)
    """
    if params is None:
        params = {}
    A = len(alphabet_tokens)
    tok_to_idx = {t: i for i, t in enumerate(alphabet_tokens)}

    # Terminal names
    term_expr = _find1(chain_sexpr, 'terminal')
    if term_expr is None:
        raise ValueError("Chain has no (terminal ...) definition")
    if isinstance(term_expr[1], list):
        terminals = [str(t) for t in term_expr[1]]
    else:
        terminals = [str(t) for t in term_expr[1:]]

    n_positions = len(terminals)
    state_size = A ** n_positions  # 4 for single, 16 for paired

    update_policy = _get_value(chain_sexpr, 'update-policy', 'rev')

    # Build state index mapping
    def state_to_idx(state_tokens):
        """Map a list of alphabet tokens to a flat index."""
        idx = 0
        for t in state_tokens:
            idx = idx * A + tok_to_idx[str(t)]
        return idx

    # Parse initial distribution
    pi = np.zeros(state_size, dtype=np.float64)
    for init_expr in _find(chain_sexpr, 'initial'):
        state_expr = _find1(init_expr, 'state')
        prob_raw = _get_value(init_expr, 'prob', 0.0)
        prob_val = _eval_parametric(prob_raw, params)
        if prob_val is None:
            prob_val = 0.0
        if state_expr is not None:
            if isinstance(state_expr[1], list):
                state_toks = [str(t) for t in state_expr[1]]
            else:
                state_toks = [str(t) for t in state_expr[1:]]
            idx = state_to_idx(state_toks)
            pi[idx] = float(prob_val)

    # Normalize pi if it doesn't sum to 1
    pi_sum = pi.sum()
    if pi_sum > 0:
        pi = pi / pi_sum

    # Parse mutation rates → rate matrix Q (off-diagonal)
    Q = np.zeros((state_size, state_size), dtype=np.float64)
    for mut_expr in _find(chain_sexpr, 'mutate'):
        from_expr = _find1(mut_expr, 'from')
        to_expr = _find1(mut_expr, 'to')
        rate_raw = _get_value(mut_expr, 'rate', 0.0)
        rate_val = _eval_parametric(rate_raw, params)
        if rate_val is None:
            rate_val = 0.0

        if from_expr is not None and to_expr is not None:
            if isinstance(from_expr[1], list):
                from_toks = [str(t) for t in from_expr[1]]
            else:
                from_toks = [str(t) for t in from_expr[1:]]
            if isinstance(to_expr[1], list):
                to_toks = [str(t) for t in to_expr[1]]
            else:
                to_toks = [str(t) for t in to_expr[1:]]
            from_idx = state_to_idx(from_toks)
            to_idx = state_to_idx(to_toks)
            Q[from_idx, to_idx] = float(rate_val)

    # Set diagonal: Q_ii = -sum_{j != i} Q_ij
    np.fill_diagonal(Q, 0.0)
    row_sums = Q.sum(axis=1)
    np.fill_diagonal(Q, -row_sums)

    return {
        'terminals': terminals,
        'n_positions': n_positions,
        'update_policy': str(update_policy),
        'pi': pi,
        'rate_matrix': Q,
        'alphabet_size': A,
    }


# ---------------------------------------------------------------------------
# Transform (production rule) extraction
# ---------------------------------------------------------------------------

def _parse_transforms(grammar_sexpr, chains, params=None):
    """Parse transform rules from an xrate grammar S-expression.

    Args:
        grammar_sexpr: the (grammar ...) S-expression
        chains: list of parsed chain dicts (from _parse_chain)
        params: dict of parametric parameter bindings

    Returns:
        (Grammar, chain_to_model_index) where chain_to_model_index maps
        terminal tuple → model index
    """
    # Build terminal → chain mapping
    # terminal_name → (chain_index, chain_dict)
    terminal_to_chain = {}
    for i, chain in enumerate(chains):
        for t in chain['terminals']:
            terminal_to_chain[t] = (i, chain)

    # Build chain → model index mapping
    # Each unique chain gets a model index
    # Single-position chains use single model indices
    # Paired chains use paired model indices
    single_chains = [c for c in chains if c['n_positions'] == 1]
    paired_chains = [c for c in chains if c['n_positions'] == 2]

    chain_to_model = {}
    single_idx = 0
    paired_idx = 0
    for chain in chains:
        key = tuple(chain['terminals'])
        if chain['n_positions'] == 1:
            chain_to_model[key] = ('single', single_idx)
            single_idx += 1
        else:
            chain_to_model[key] = ('paired', paired_idx)
            paired_idx += 1

    n_single = single_idx
    n_paired = paired_idx

    # Collect all nonterminal names from transforms
    # Also identify which names are terminals (chain terminal names)
    all_terminal_names = set()
    for chain in chains:
        for t in chain['terminals']:
            all_terminal_names.add(t)

    # Gather nonterminal names from transforms
    nt_names = []
    nt_name_set = set()

    def ensure_nt(name):
        if name not in nt_name_set and name not in all_terminal_names:
            nt_name_set.add(name)
            nt_names.append(name)

    transforms = _find(grammar_sexpr, 'transform')

    for tf in transforms:
        from_expr = _find1(tf, 'from')
        to_expr = _find1(tf, 'to')
        if from_expr is None or to_expr is None:
            continue

        # from is always a single nonterminal
        if isinstance(from_expr[1], list):
            from_syms = [str(s) for s in from_expr[1]]
        else:
            from_syms = [str(s) for s in from_expr[1:]]

        for s in from_syms:
            ensure_nt(s)

        # to can be: () empty, (NT), (NT NT), (TERM NT), (TERM NT TERM), etc.
        if isinstance(to_expr[1], list):
            to_syms = [str(s) for s in to_expr[1]]
        elif len(to_expr) == 1:
            to_syms = []
        else:
            to_syms = [str(s) for s in to_expr[1:]]

        for s in to_syms:
            ensure_nt(s)

    # Build GrammarBuilder
    gb = GrammarBuilder()
    nt_idx = {}
    for name in nt_names:
        idx = gb.add_nonterminal(name)
        nt_idx[name] = idx

    # Now parse each transform into a Rule
    for tf in transforms:
        from_expr = _find1(tf, 'from')
        to_expr = _find1(tf, 'to')
        if from_expr is None or to_expr is None:
            continue

        if isinstance(from_expr[1], list):
            from_syms = [str(s) for s in from_expr[1]]
        else:
            from_syms = [str(s) for s in from_expr[1:]]

        lhs_name = from_syms[0]
        lhs = nt_idx[lhs_name]

        if isinstance(to_expr[1], list):
            to_syms = [str(s) for s in to_expr[1]]
        elif len(to_expr) == 1:
            to_syms = []
        else:
            to_syms = [str(s) for s in to_expr[1:]]

        # Get probability/weight
        prob_raw = _get_value(tf, 'prob')
        if prob_raw is not None:
            if params is None:
                params = {}
            prob_val = _eval_parametric(prob_raw, params)
            if prob_val is not None and prob_val > 0:
                log_weight = math.log(prob_val)
            else:
                log_weight = -1e38
        else:
            log_weight = 0.0  # implicit 1.0

        # Classify symbols in 'to' as terminals or nonterminals
        # and determine the emission groups and RHS nonterminals
        rhs_nts = []
        emissions = []

        # Group consecutive terminal symbols into emission groups
        # Pattern: terminals appear at edges (left-emit, right-emit, or paired)
        # e.g., (LNUC F RNUC) → paired emission with LNUC+RNUC, rhs=[F]
        # e.g., (NUC F) → single left-emission with NUC, rhs=[F]

        # Separate terminals and nonterminals in order
        sym_types = []
        for s in to_syms:
            if s in all_terminal_names:
                sym_types.append(('term', s))
            else:
                sym_types.append(('nt', s))

        # Identify emission patterns:
        # Collect terminal groups and match them to chains
        term_symbols = [s for typ, s in sym_types if typ == 'term']

        if len(term_symbols) > 0:
            # Find which chain these terminals belong to
            # For paired: (LNUC ... RNUC) → chain with terminals [LNUC, RNUC]
            # For single: (NUC ...) → chain with terminal [NUC]
            matched_chain_key = _match_terminals_to_chain(
                term_symbols, chains, chain_to_model)
            if matched_chain_key is not None:
                model_type, model_idx = chain_to_model[matched_chain_key]
                chain = next(c for c in chains
                             if tuple(c['terminals']) == matched_chain_key)
                emissions.append(EmissionGroup(
                    n_positions=chain['n_positions'],
                    model_index=model_idx,
                ))

        # RHS nonterminals (in order they appear, excluding terminals)
        for typ, s in sym_types:
            if typ == 'nt':
                rhs_nts.append(nt_idx[s])

        gb.add_rule(lhs, rhs=rhs_nts, emissions=emissions,
                    log_weight=log_weight)

    # n_models must be >= max model_index + 1 for validation.
    # Single and paired use separate index spaces, but n_models must
    # cover both for validation purposes.
    n_models = max(n_single, n_paired, 1)

    grammar = gb.build(start=0, n_models=n_models)

    return grammar, chain_to_model


def _match_terminals_to_chain(term_symbols, chains, chain_to_model):
    """Match a list of terminal symbols to a chain.

    For single chains: [NUC] matches chain with terminal (NUC)
    For paired chains: [LNUC, RNUC] matches chain with terminals (LNUC RNUC)
    """
    # Try exact match first
    key = tuple(term_symbols)
    if key in chain_to_model:
        return key

    # For paired emissions like (LNUC F* RNUC), term_symbols = [LNUC, RNUC]
    # Check if any chain has exactly these terminals
    for chain in chains:
        chain_key = tuple(chain['terminals'])
        if set(chain_key) == set(term_symbols):
            return chain_key

    # Single terminal match
    if len(term_symbols) == 1:
        for chain in chains:
            if term_symbols[0] in chain['terminals'] and chain['n_positions'] == 1:
                return tuple(chain['terminals'])

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class XrateGrammar:
    """Parsed xrate grammar with grammar and substitution model information.

    Attributes:
        grammar: jaxrate Grammar
        chains: list of chain dicts with 'terminals', 'pi', 'rate_matrix',
                'n_positions', 'update_policy', 'alphabet_size'
        alphabet: dict with 'name', 'tokens', 'complement'
        chain_to_model: dict mapping terminal tuple → ('single'|'paired', index)
    """

    def __init__(self, grammar, chains, alphabet, chain_to_model):
        self.grammar = grammar
        self.chains = chains
        self.alphabet = alphabet
        self.chain_to_model = chain_to_model

    def get_single_models(self):
        """Get single-column chain specs in model index order.

        Returns:
            list of chain dicts for single-column models
        """
        indexed = []
        for chain in self.chains:
            key = tuple(chain['terminals'])
            if key in self.chain_to_model:
                model_type, idx = self.chain_to_model[key]
                if model_type == 'single':
                    indexed.append((idx, chain))
        indexed.sort(key=lambda x: x[0])
        return [c for _, c in indexed]

    def get_paired_models(self):
        """Get paired-column chain specs in model index order.

        Returns:
            list of chain dicts for paired-column models
        """
        indexed = []
        for chain in self.chains:
            key = tuple(chain['terminals'])
            if key in self.chain_to_model:
                model_type, idx = self.chain_to_model[key]
                if model_type == 'paired':
                    indexed.append((idx, chain))
        indexed.sort(key=lambda x: x[0])
        return [c for _, c in indexed]

    def to_subby_models(self):
        """Convert chains to subby DiagModel/IrrevDiagModel objects.

        Returns:
            dict with:
                'single': list of subby models for single-column emissions
                'paired': list of subby models for paired-column emissions
        """
        from subby.jax.models import model_from_rate_matrix

        single_models = []
        for chain in self.get_single_models():
            Q = jnp.array(chain['rate_matrix'])
            pi = jnp.array(chain['pi'])
            reversible = chain['update_policy'] == 'rev'
            model = model_from_rate_matrix(Q, pi, reversible=reversible)
            single_models.append(model)

        paired_models = []
        for chain in self.get_paired_models():
            Q = jnp.array(chain['rate_matrix'])
            pi = jnp.array(chain['pi'])
            reversible = chain['update_policy'] == 'rev'
            model = model_from_rate_matrix(Q, pi, reversible=reversible)
            paired_models.append({
                'model': model,
                'A': chain['alphabet_size'],
            })

        return {
            'single': single_models,
            'paired': paired_models,
        }


def parse_xrate(text, start_nonterminal=None):
    """Parse an xrate .eg grammar file.

    Args:
        text: string contents of an xrate .eg file
        start_nonterminal: name of the start nonterminal (default: first
            nonterminal that appears in a 'from' clause)

    Returns:
        XrateGrammar with grammar, chains, alphabet, and model mappings
    """
    exprs = parse_sexpr(text)

    # Expand macros (&define, &foreach-*, &cat, &+, etc.)
    exprs = expand_macros(exprs)

    # Find alphabet and grammar blocks
    alphabet_expr = None
    grammar_expr = None
    for expr in exprs:
        if isinstance(expr, list) and len(expr) > 0:
            if expr[0] == 'alphabet':
                alphabet_expr = expr
            elif expr[0] == 'grammar':
                grammar_expr = expr

    if grammar_expr is None:
        raise ValueError("No (grammar ...) block found")

    # Parse alphabet (use default RNA if not present)
    if alphabet_expr is not None:
        alphabet = _parse_alphabet(alphabet_expr)
    else:
        alphabet = {'name': 'RNA', 'tokens': ['a', 'c', 'g', 'u'],
                     'complement': None}

    # Parse parametric parameters (pgroup, rate)
    params = _parse_params(grammar_expr)

    # Parse chains
    chain_exprs = _find(grammar_expr, 'chain')
    chains = [_parse_chain(ce, alphabet['tokens'], params) for ce in chain_exprs]

    # Parse transforms into grammar
    grammar, chain_to_model = _parse_transforms(grammar_expr, chains, params)

    # Set start nonterminal if specified
    if start_nonterminal is not None:
        nt_names = [nt.name for nt in grammar.nonterminals]
        if start_nonterminal in nt_names:
            start_idx = nt_names.index(start_nonterminal)
            grammar = grammar._replace(start=start_idx)
        else:
            raise ValueError(
                f"Start nonterminal '{start_nonterminal}' not found. "
                f"Available: {nt_names}")

    return XrateGrammar(grammar, chains, alphabet, chain_to_model)


def parse_xrate_file(filepath, **kwargs):
    """Parse an xrate .eg grammar file from a file path.

    Args:
        filepath: path to the .eg file
        **kwargs: passed to parse_xrate

    Returns:
        XrateGrammar
    """
    with open(filepath, 'r') as f:
        text = f.read()
    return parse_xrate(text, **kwargs)
