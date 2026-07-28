# Wave-24 GP tree representation: signal-formula individuals (SPEC.md "GP 설계").
#
# An individual here is a TREE, not a fixed-shape parameter vector (contrast
# research.wave21_ga.genome.Genome / research.wave23_ga_short.genome23.Genome, both frozen
# dataclasses with a FIXED set of named fields). SPEC.md's whole point is that wave-21/23's
# search SPACE (5 hand-picked strategy shapes, parameters only) was the bottleneck, not the
# search algorithm -- so this wave lets evolution build the SIGNAL FORMULA itself out of a small
# function/terminal alphabet, nested up to depth 5.
#
# Node is a frozen dataclass with tuple children -- Python auto-generates a structural
# __hash__/__eq__ for it (every field is hashable: str, tuple[Node, ...], float | int | None), so
# a Node IS its own fitness-cache key with no separate key-encoding step (contrast
# research.wave21_ga.genome.genome_key, which exists only because Genome's own float genes need
# rounding before they are usable as a dict key -- this wave's only floats are the 4 frozen
# constants {0.5, 1, 2, 5}, drawn from a finite set with no continuous mutation, so no rounding is
# ever needed for two structurally-identical trees to hash equal).
#
# Evaluation operates on pd.DataFrame throughout (not raw numpy at each node) -- every terminal
# panel shares the EXACT SAME index/columns (both come from fitness24.MarketCache, built once),
# so DataFrame alignment on +/-/*// is a no-op, and keeping results as DataFrames end-to-end
# avoids repeated numpy<->pandas conversion at every one of a tree's up-to-63 nodes; only
# fitness24.run_backtest converts the FINAL score to numpy, once.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

# ---------------------------------------------------------------------------
# Frozen alphabet (SPEC.md "GP 설계" -- terminal/function tables, byte-for-byte).
# ---------------------------------------------------------------------------

TERMINAL_VARS: Final[tuple[str, ...]] = (
    "funding_1d",
    "funding_7d",
    "funding_14d",
    "funding_30d",
    "price_ret_1d",
    "price_ret_7d",
    "price_ret_30d",
    "realized_vol_20d",
    "atr_14",
    "quote_volume_30d",
    "basis",
)
CONST_VALUES: Final[tuple[float, ...]] = (0.5, 1.0, 2.0, 5.0)
CONST_OP: Final = "const"

# function name -> arity (number of children). SPEC.md: "+, -, x, /(보호), log(보호), abs, min,
# max, z_score(20d), rank_cross_sectional, ma(n in {5,10,20})".
FUNCTIONS_ARITY: Final[dict[str, int]] = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "log": 1,
    "abs": 1,
    "min": 2,
    "max": 2,
    "zscore": 1,
    "rank_cs": 1,
    "ma": 1,
}
FUNCTION_NAMES: Final[tuple[str, ...]] = tuple(FUNCTIONS_ARITY)
MA_WINDOWS: Final[tuple[int, ...]] = (5, 10, 20)
ZSCORE_WINDOW: Final = 20  # SPEC.md "z_score(20d)" -- fixed window, not itself evolved
MAX_DEPTH: Final = 5  # SPEC.md "트리 깊이 <= 5"
_EPS: Final = 1e-9  # protected-op floor (division / log)

# terminal "kinds" for L7's "사용 터미널 <= 5종" -- every named market terminal is its own kind;
# every constant (regardless of WHICH of the 4 frozen values) counts as the single kind "const"
# (a constant is not a market-data read, so 3 different constants used in one tree should not by
# themselves eat 3 of the 5-kind budget the way 3 different market terminals would).
ALL_TERMINAL_KINDS: Final[tuple[str, ...]] = (*TERMINAL_VARS, CONST_OP)


@dataclass(frozen=True, slots=True)
class Node:
    op: str  # one of TERMINAL_VARS, CONST_OP, or a FUNCTIONS_ARITY key
    children: tuple["Node", ...] = ()
    value: float | int | None = None  # const's numeric value, or ma's window n; else None

    @property
    def is_terminal(self) -> bool:
        return self.op == CONST_OP or self.op in TERMINAL_VARS

    @property
    def is_function(self) -> bool:
        return self.op in FUNCTIONS_ARITY


def _validate(node: Node) -> None:
    """Fail closed on a malformed tree (task instruction 자체가 아니라, this repo's pervasive
    'reject silently-wrong state at construction' convention -- e.g. research.wave21_ga.genome.
    Genome.__post_init__). Called explicitly by every construction path below (random generation,
    crossover, mutation, from_dict) rather than wired into Node.__post_init__, because __post_init__
    would re-validate every already-validated child on every parent construction (O(n^2) over a
    single build) -- validating once, top-down, after a tree is fully assembled is equivalent and
    cheaper."""
    if node.op == CONST_OP:
        if node.children or node.value not in CONST_VALUES:
            raise ValueError(f"tree.Node: const node must have no children and value in {CONST_VALUES}, got {node!r}")
        return
    if node.op in TERMINAL_VARS:
        if node.children or node.value is not None:
            raise ValueError(f"tree.Node: terminal var node must have no children/value, got {node!r}")
        return
    if node.op in FUNCTIONS_ARITY:
        arity = FUNCTIONS_ARITY[node.op]
        if len(node.children) != arity:
            raise ValueError(f"tree.Node: op {node.op!r} needs {arity} children, got {len(node.children)}")
        if node.op == "ma":
            if node.value not in MA_WINDOWS:
                raise ValueError(f"tree.Node: ma window {node.value!r} not in {MA_WINDOWS}")
        elif node.value is not None:
            raise ValueError(f"tree.Node: op {node.op!r} takes no value, got {node.value!r}")
        for child in node.children:
            _validate(child)
        return
    raise ValueError(f"tree.Node: unknown op {node.op!r}")


def validate_tree(node: Node, max_depth: int = MAX_DEPTH) -> None:
    _validate(node)
    observed = depth(node)
    if observed > max_depth:
        raise ValueError(f"tree.validate_tree: depth {observed} exceeds max_depth={max_depth}")


# ---------------------------------------------------------------------------
# Structural measurements.
# ---------------------------------------------------------------------------


def depth(node: Node) -> int:
    """Edges from `node` to its deepest leaf -- a bare terminal has depth 0. MAX_DEPTH=5 therefore
    allows a root-to-leaf chain of up to 5 function nodes."""
    if node.is_terminal:
        return 0
    return 1 + max(depth(child) for child in node.children)


def node_count(node: Node) -> int:
    if node.is_terminal:
        return 1
    return 1 + sum(node_count(child) for child in node.children)


def terminal_kinds_used(node: Node) -> frozenset[str]:
    """L7's own vocabulary: distinct terminal KINDS actually referenced in this tree (see
    ALL_TERMINAL_KINDS' own docstring for why every constant collapses to the single kind
    'const')."""
    if node.op == CONST_OP:
        return frozenset({CONST_OP})
    if node.op in TERMINAL_VARS:
        return frozenset({node.op})
    used: set[str] = set()
    for child in node.children:
        used |= terminal_kinds_used(child)
    return frozenset(used)


def all_nodes(node: Node) -> list[Node]:
    """Pre-order flattening of every subtree (node itself first, then each child's own pre-order
    flattening in order) -- index i into this list is exactly the index replace_subtree expects."""
    out = [node]
    for child in node.children:
        out.extend(all_nodes(child))
    return out


def replace_subtree(node: Node, target_index: int, replacement: Node) -> Node:
    """Returns a NEW tree with the target_index-th node (pre-order, matching all_nodes' own
    ordering -- index 0 is always the root) replaced by `replacement`. Nodes outside the replaced
    subtree's path are structurally rebuilt but value-identical; nothing outside the path is
    mutated in place (Node is frozen, so "in place" is not even possible)."""
    position = 0

    def _walk(current: Node) -> Node:
        nonlocal position
        this_index = position
        position += 1
        if this_index == target_index:
            return replacement
        if current.is_terminal:
            return current
        return Node(op=current.op, children=tuple(_walk(child) for child in current.children), value=current.value)

    result = _walk(node)
    if position <= target_index:
        raise IndexError(f"tree.replace_subtree: target_index={target_index} out of range (tree has {position} nodes)")
    return result


def subtree_depth_budget_ok(node: Node, replacement: Node, target_index: int, max_depth: int = MAX_DEPTH) -> bool:
    """True iff splicing `replacement` in at `target_index` would keep the WHOLE tree's depth
    <= max_depth -- checked WITHOUT materializing the spliced tree (crossover/mutation call this
    many times while hunting for a valid splice point; building the full candidate tree on every
    attempt just to measure its depth and possibly discard it is the same computation done the
    slow way)."""
    depth_at_target = depth_at_index(node, target_index)
    return depth_at_target + depth(replacement) <= max_depth


def depth_at_index(node: Node, target_index: int) -> int:
    """How many edges from the root down to the target_index-th node (pre-order, matching
    all_nodes' own ordering) -- the "depth budget already consumed" a splice at that point
    inherits from its ancestors. Public (used by gp.py's mutate_subtree to size a fresh random
    replacement so it fits the remaining budget without any rejection sampling)."""
    position = 0

    def _walk(current: Node, level: int) -> int | None:
        nonlocal position
        this_index = position
        position += 1
        if this_index == target_index:
            return level
        if current.is_terminal:
            return None
        for child in current.children:
            found = _walk(child, level + 1)
            if found is not None:
                return found
        return None

    found = _walk(node, 0)
    if found is None:
        raise IndexError(f"tree.depth_at_index: target_index={target_index} out of range")
    return found


# ---------------------------------------------------------------------------
# Random generation: grow / full / ramped half-and-half (Koza 1992, standard GP initialization --
# mixing grow (irregular shape, terminal can appear early) and full (every branch reaches
# max_depth) avoids both grow-only's bias toward small trees and full-only's bias toward uniformly
# bushy ones).
# ---------------------------------------------------------------------------


def random_terminal(rng: np.random.Generator) -> Node:
    if rng.uniform() < 0.5:
        return Node(op=CONST_OP, value=float(rng.choice(CONST_VALUES)))
    return Node(op=str(rng.choice(TERMINAL_VARS)))


def _random_function_node(rng: np.random.Generator, child_factory: Callable[[], Node]) -> Node:
    op = str(rng.choice(FUNCTION_NAMES))
    arity = FUNCTIONS_ARITY[op]
    children = tuple(child_factory() for _ in range(arity))
    value = int(rng.choice(MA_WINDOWS)) if op == "ma" else None
    return Node(op=op, children=children, value=value)


def grow(rng: np.random.Generator, max_depth: int) -> Node:
    if max_depth <= 0 or rng.uniform() < 0.5:
        return random_terminal(rng)
    return _random_function_node(rng, lambda: grow(rng, max_depth - 1))


def full(rng: np.random.Generator, max_depth: int) -> Node:
    if max_depth <= 0:
        return random_terminal(rng)
    return _random_function_node(rng, lambda: full(rng, max_depth - 1))


def ramped_half_and_half(rng: np.random.Generator, population_size: int, min_depth: int = 2, max_depth: int = MAX_DEPTH) -> list[Node]:
    """Standard Koza ramped half-and-half: population_size individuals split evenly across
    depths min_depth..max_depth, and within each depth bucket split evenly between grow and full.
    Index-driven (not random) bucket assignment so the ramp is exactly even regardless of rng
    draws; the randomness lives inside each grow()/full() call itself."""
    depths = list(range(min_depth, max_depth + 1))
    trees: list[Node] = []
    for i in range(population_size):
        target_depth = depths[i % len(depths)]
        generator = grow if (i // len(depths)) % 2 == 0 else full
        tree = generator(rng, target_depth)
        validate_tree(tree, max_depth)
        trees.append(tree)
    return trees


# ---------------------------------------------------------------------------
# Human-readable formula string (task requirement: "최종 수식을 사람이 읽을 수 있는 형태로 출력").
# ---------------------------------------------------------------------------

_INFIX_DISPLAY: Final[dict[str, str]] = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def _format_const(value: float) -> str:
    return f"{value:g}"


def to_formula_string(node: Node) -> str:
    if node.op == CONST_OP:
        return _format_const(float(node.value))
    if node.op in TERMINAL_VARS:
        return node.op
    if node.op in _INFIX_DISPLAY:
        left, right = (to_formula_string(child) for child in node.children)
        return f"({left} {_INFIX_DISPLAY[node.op]} {right})"
    if node.op in {"min", "max"}:
        left, right = (to_formula_string(child) for child in node.children)
        return f"{node.op}({left}, {right})"
    if node.op == "ma":
        return f"ma({to_formula_string(node.children[0])}, {node.value}d)"
    if node.op == "zscore":
        return f"zscore({to_formula_string(node.children[0])}, {ZSCORE_WINDOW}d)"
    if node.op == "rank_cs":
        return f"rank_pct({to_formula_string(node.children[0])})"
    if node.op == "log":
        return f"log(|{to_formula_string(node.children[0])}|)"
    if node.op == "abs":
        return f"abs({to_formula_string(node.children[0])})"
    raise ValueError(f"tree.to_formula_string: unknown op {node.op!r}")


# ---------------------------------------------------------------------------
# JSON (de)serialization (results/*.json payloads).
# ---------------------------------------------------------------------------


def to_dict(node: Node) -> dict:
    return {"op": node.op, "value": node.value, "children": [to_dict(child) for child in node.children]}


def from_dict(payload: dict) -> Node:
    children = tuple(from_dict(child) for child in payload.get("children", []))
    value = payload.get("value")
    node = Node(op=str(payload["op"]), children=children, value=value)
    _validate(node)
    return node


# ---------------------------------------------------------------------------
# Evaluation: protected ops (division / log), rolling ops (zscore / ma), cross-sectional rank.
#
# NaN handling (deliberate, load-bearing for correctness -- see tests/test_wave24.py::
# test_safe_div_preserves_nan / test_safe_log_preserves_nan): a terminal is NaN wherever its own
# trailing window has not warmed up yet (e.g. atr_14's first 13 rows for a symbol) or the symbol
# itself has no data that day. That NaN must propagate all the way to the tree's final score --
# turning it into a fabricated 0.0 would silently manufacture a fake "borderline, no signal"
# score out of "the answer is simply unknown yet", and fitness24.run_backtest's own eligibility
# mask (`np.isfinite(score)`) is what actually excludes those cells from trading, matching
# research.wave21_ga.fitness's `np.isfinite(ranking)` convention. The protected ops below
# therefore intervene ONLY on genuinely-finite-but-degenerate inputs (a near-zero divisor, a
# near-zero log argument); a NaN input passes straight through.
# ---------------------------------------------------------------------------


def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = a / b
    near_zero = b.notna() & (b.abs() < _EPS)
    return raw.mask(near_zero, 0.0)


def _safe_log(a: pd.DataFrame) -> pd.DataFrame:
    magnitude = a.abs()
    floored = magnitude.mask(magnitude.notna() & (magnitude < _EPS), _EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(floored)


def _zscore(frame: pd.DataFrame, window: int = ZSCORE_WINDOW) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=window).mean()
    std = frame.rolling(window, min_periods=window).std(ddof=0)
    return _safe_div(frame - mean, std)


def _ma(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).mean()


def _rank_cs(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank IN (0, 1], computed fresh per row (per day) across the
    symbol axis -- pandas' own rank(pct=True) skips NaN cells both as rankable values and as
    ranking targets (a NaN cell stays NaN), which is exactly the propagation this module's NaN
    convention needs."""
    return frame.rank(axis=1, pct=True)


def _apply(op: str, args: list[pd.DataFrame], value: float | int | None) -> pd.DataFrame:
    if op == "add":
        return args[0] + args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "div":
        return _safe_div(args[0], args[1])
    if op == "log":
        return _safe_log(args[0])
    if op == "abs":
        return args[0].abs()
    if op == "min":
        return np.minimum(args[0], args[1])
    if op == "max":
        return np.maximum(args[0], args[1])
    if op == "zscore":
        return _zscore(args[0])
    if op == "rank_cs":
        return _rank_cs(args[0])
    if op == "ma":
        return _ma(args[0], int(value))
    raise ValueError(f"tree._apply: unknown op {op!r}")


def _eval(node: Node, terminals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if node.op == CONST_OP:
        reference = next(iter(terminals.values()))
        return pd.DataFrame(float(node.value), index=reference.index, columns=reference.columns)
    if node.op in TERMINAL_VARS:
        return terminals[node.op]
    args = [_eval(child, terminals) for child in node.children]
    return _apply(node.op, args, node.value)


def evaluate(node: Node, terminals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evaluates `node` over `terminals` (name -> [days, symbols] DataFrame, all sharing the same
    index/columns -- fitness24.MarketCache's own terminal panels, already sliced to whatever date
    range the caller is allowed to see; tree.py itself has no notion of IS/OOS). A final
    inf->NaN sweep (only at the ROOT, not per-node -- cheaper, and equivalent: inf can only ever
    be produced or widened by +/-/*, never erased by a later op, so checking once at the end
    catches every path) guards against e.g. `mul` overflow; treated the same as any other
    "unknown" cell (excluded from eligibility downstream), never clipped to a large finite
    number that would otherwise dominate cross-sectional ranking in an uninterpretable way."""
    result = _eval(node, terminals)
    return result.replace([np.inf, -np.inf], np.nan)


__all__ = [
    "ALL_TERMINAL_KINDS",
    "CONST_OP",
    "CONST_VALUES",
    "FUNCTIONS_ARITY",
    "FUNCTION_NAMES",
    "MA_WINDOWS",
    "MAX_DEPTH",
    "TERMINAL_VARS",
    "ZSCORE_WINDOW",
    "Node",
    "all_nodes",
    "depth",
    "depth_at_index",
    "evaluate",
    "from_dict",
    "full",
    "grow",
    "node_count",
    "ramped_half_and_half",
    "random_terminal",
    "replace_subtree",
    "subtree_depth_budget_ok",
    "terminal_kinds_used",
    "to_dict",
    "to_formula_string",
    "validate_tree",
]
