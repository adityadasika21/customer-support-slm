"""Tests for near-duplicate clustering and split assignment.

The leakage guarantee is the foundation of every number this project reports, so it is tested as
a property rather than spot-checked: for a synthetic dataset with known duplicate groups, no
cluster may span two splits, and no pair above the threshold may end up on opposite sides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csbot.data.prepare import (
    SplitConfig,
    UnionFind,
    build_splits,
    check_leakage,
    cluster_near_duplicates,
    normalize_instruction,
    similarity_blocks,
)


_VERBS = ["cancel", "track", "refund", "change", "invoice", "delete"]

#: Three disjoint word pools. Each group draws one word from each, so two different groups differ
#: in *three* content words while two members of the same group differ by a single typo. That gap
#: is what gives the fixture a clean separation window; a carrier sentence varying by only one
#: word makes within-group and between-group distances comparable, and then no threshold can
#: separate them.
_OPENERS = [
    "urgently", "quietly", "kindly", "swiftly", "briefly", "politely", "firmly", "gently",
    "promptly", "calmly", "bluntly", "warmly", "plainly", "neatly", "sharply", "boldly",
    "softly", "gladly", "keenly", "loosely", "wildly", "grimly", "fondly", "oddly",
    "rashly", "vainly", "aptly", "dimly", "justly", "meekly", "nobly", "primly",
    "rudely", "sagely", "tamely", "vastly", "weakly", "yearly", "zanily", "ably",
]
_SUBJECTS = [
    "jacket", "espresso", "trainers", "lamp", "keyboard", "hosepipe", "yogamat", "helmet",
    "blender", "printer", "rucksack", "armchair", "speaker", "grinder", "spectacles", "kettle",
    "roller", "needle", "racquet", "campstove", "mirror", "wallet", "ventilator", "hamper",
    "frame", "doorknob", "wateringcan", "flannel", "shoerack", "cutlery", "podium", "sleeve",
    "bolster", "collar", "flowerpot", "journal", "toolbelt", "adaptor", "platter", "stepladder",
]
_SUFFIXES = [
    "tomorrow", "immediately", "beforehand", "afterwards", "overnight", "regardless", "anyway",
    "somehow", "thereafter", "meanwhile", "elsewhere", "throughout", "henceforth", "likewise",
    "moreover", "instead", "nonetheless", "presently", "shortly", "thoroughly", "twice",
    "upfront", "wherever", "yesterday", "altogether", "briskly", "certainly", "definitely",
    "entirely", "frankly", "genuinely", "hopefully", "ideally", "jointly", "lately",
    "mainly", "naturally", "openly", "partly", "rarely",
]


def make_frame(n_intents: int = 6, groups_per_intent: int = 12, per_group: int = 4) -> pd.DataFrame:
    """Synthetic data with a known duplicate structure.

    Each group is one request written several ways -- typos and punctuation changes -- mirroring
    how Bitext was generated. A ``group_id`` column records the ground truth so tests can assert
    that clustering recovers exactly those groups.
    """
    if groups_per_intent > len(_SUBJECTS):
        raise ValueError("not enough distinct subjects for that many groups")

    rows = []
    for i in range(n_intents):
        verb = _VERBS[i % len(_VERBS)]
        for g in range(groups_per_intent):
            opener, subject, suffix = _OPENERS[g], _SUBJECTS[g], _SUFFIXES[g]
            base = f"{opener} {verb} the {subject} {suffix}"
            variants = [
                base,
                base.replace(opener, opener[:2] + opener[2:].replace("l", "ll", 1)),  # typo
                base.replace(suffix, suffix + "e"),                                   # typo
                base + "?",                                                           # punctuation
            ]
            for v in variants[:per_group]:
                rows.append(
                    {
                        "instruction": v,
                        "response": f"Sure, the {subject} is {verb}led.",
                        "intent": f"intent_{i}",
                        "category": f"CAT_{i % 3}",
                        "flags": "B",
                        "group_id": f"{i}:{g}",
                    }
                )
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    return df


# -- union-find ------------------------------------------------------------------------------

def test_unionfind_builds_components():
    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(4, 5)
    labels = uf.components()
    assert labels[0] == labels[1] == labels[2]
    assert labels[4] == labels[5]
    assert labels[0] != labels[3] != labels[4]


def test_unionfind_labels_are_dense():
    uf = UnionFind(5)
    uf.union(0, 4)
    labels = uf.components()
    assert set(labels) == set(range(labels.max() + 1))


# -- normalisation ---------------------------------------------------------------------------

def test_normalize_strips_placeholders_and_punctuation():
    assert normalize_instruction("Cancel ORDER {{Order Number}}!!") == "cancel order"


def test_normalize_collapses_whitespace():
    assert normalize_instruction("a   b\n\tc") == "a b c"


# -- clustering ------------------------------------------------------------------------------

def test_typo_variants_cluster_together():
    """The whole point: a typo must not separate two copies of the same request."""
    df = make_frame()
    labels, stats = cluster_near_duplicates(df, threshold=0.8)

    frame = df.assign(cluster=labels)
    for group_id, group in frame.groupby("group_id"):
        assert group["cluster"].nunique() == 1, (
            f"group {group_id} was split across clusters: {group['instruction'].tolist()}"
        )

    assert stats.n_clusters < len(df)


def test_clustering_recovers_the_known_groups_exactly():
    """Neither over- nor under-merging: exactly one cluster per ground-truth group.

    The fixture is built with a wide separation window -- within-group similarity >= 0.84,
    between-group <= 0.52 -- so this holds across the whole 0.6-0.9 range rather than balancing
    on one lucky threshold.
    """
    df = make_frame()
    for threshold in (0.6, 0.7, 0.8, 0.9):
        _, stats = cluster_near_duplicates(df, threshold=threshold)
        assert stats.n_clusters == df["group_id"].nunique(), f"failed at {threshold}"


def test_clustering_never_splits_a_true_group():
    """Conservative in the safe direction across a range of thresholds.

    Over-merging costs data; under-merging leaks. This asserts the failure mode we can tolerate
    is the only one that occurs.
    """
    df = make_frame()
    for threshold in (0.7, 0.8, 0.9):
        labels, _ = cluster_near_duplicates(df, threshold=threshold)
        frame = df.assign(cluster=labels)
        assert (frame.groupby("group_id")["cluster"].nunique() == 1).all(), (
            f"a true group was split at threshold {threshold}"
        )


def test_distinct_requests_stay_separate():
    df = make_frame()
    labels, _ = cluster_near_duplicates(df, threshold=0.8)
    # Different intents must never share a cluster, since blocks are per-intent and exact
    # duplicates across intents do not occur in this fixture.
    frame = df.assign(cluster=labels)
    per_cluster_intents = frame.groupby("cluster")["intent"].nunique()
    assert (per_cluster_intents == 1).all()


def test_higher_threshold_produces_more_clusters():
    df = make_frame()
    loose, _ = cluster_near_duplicates(df, threshold=0.5)
    tight, _ = cluster_near_duplicates(df, threshold=0.95)
    assert tight.max() >= loose.max()


def test_similarity_blocks_are_symmetric_and_bounded():
    df = make_frame()
    for _, sim in similarity_blocks(df):
        assert np.allclose(sim, sim.T, atol=1e-6)
        assert sim.min() >= -1e-6 and sim.max() <= 1.0 + 1e-6


# -- splitting -------------------------------------------------------------------------------

def test_no_cluster_spans_splits():
    """The core leakage guarantee."""
    df = make_frame()
    out, _ = build_splits(
        df, config=SplitConfig(heldout_intents=("intent_0",)), threshold=0.8
    )
    report = check_leakage(out)
    assert report["leak_free"]
    assert report["clusters_spanning_splits"] == 0


def test_no_similar_pair_crosses_the_split():
    """Stronger than the cluster check: verify the pairwise property the cluster check implies."""
    df = make_frame()
    threshold = 0.8
    out, _ = build_splits(
        df, config=SplitConfig(heldout_intents=("intent_0",)), threshold=threshold
    )
    splits = out["split"].to_numpy()

    for idx, sim in similarity_blocks(out):
        local = splits[idx]
        rows, cols = np.nonzero(np.triu(sim, k=1) >= threshold)
        for r, c in zip(rows, cols):
            assert local[r] == local[c], (
                f"pair above threshold straddles splits: "
                f"{out['instruction'].iloc[idx[r]]!r} / {out['instruction'].iloc[idx[c]]!r}"
            )


def test_all_splits_present_and_proportions_reasonable():
    df = make_frame(n_intents=6, groups_per_intent=40, per_group=4)
    out, _ = build_splits(df, config=SplitConfig(heldout_intents=("intent_0",)), threshold=0.8)
    counts = out["split"].value_counts(normalize=True)
    assert set(counts.index) == {"train", "val", "test"}
    assert counts["train"] > counts["test"] > 0


def test_heldout_intent_flag():
    df = make_frame()
    out, _ = build_splits(df, config=SplitConfig(heldout_intents=("intent_1",)), threshold=0.8)
    assert out[out.heldout_intent]["intent"].unique().tolist() == ["intent_1"]


def test_unknown_heldout_intent_rejected():
    """A typo in the config must fail loudly, not silently hold out nothing."""
    df = make_frame()
    with pytest.raises(ValueError, match="not present"):
        build_splits(df, config=SplitConfig(heldout_intents=("does_not_exist",)))


def test_split_assignment_is_deterministic():
    df = make_frame()
    a, _ = build_splits(df, config=SplitConfig(heldout_intents=("intent_0",), seed=3), threshold=0.8)
    b, _ = build_splits(df, config=SplitConfig(heldout_intents=("intent_0",), seed=3), threshold=0.8)
    assert a["split"].equals(b["split"])


def test_split_proportions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitConfig(train=0.5, val=0.2, test=0.2)


# -- stratified sampling ---------------------------------------------------------------------

def test_stratified_subset_preserves_grouping_column():
    """Regression: pandas 3.0's groupby().apply() drops the grouping column.

    The funnel subset was silently losing its ``intent`` column, which only surfaced much later
    as an AttributeError deep inside scoring. Any stratified sampler here must keep every column.
    """
    df = make_frame(n_intents=5, groups_per_intent=8)
    per = 3
    parts = [
        g.sample(min(len(g), per), random_state=0)
        for _, g in df.groupby("intent", sort=True)
    ]
    sampled = pd.concat(parts).reset_index(drop=True)

    assert "intent" in sampled.columns
    assert set(sampled.columns) == set(df.columns)
    assert sampled["intent"].nunique() == df["intent"].nunique()
    assert next(sampled.itertuples(index=False).__iter__()).intent
