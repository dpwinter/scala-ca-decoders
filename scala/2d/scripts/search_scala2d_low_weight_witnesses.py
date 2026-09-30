from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_pheno_hazard import (
    active_lanes,
    actual_logicals,
    advance_group,
    build_matching,
    make_group,
    unpack_syndrome,
)


# =============================================================================
# Eq. (2) reference weight
# =============================================================================

def factor_pairs(w: int):
    out = []

    for k in range(1, w + 1):
        if w % k == 0:
            out.append(
                (
                    k,
                    w // k,
                )
            )

    return out


def f_integer(w: int) -> int:
    return min(
        k + ne
        for k, ne in factor_pairs(w)
    )


def witness_weight(d: int) -> int:
    """
    Discrete SCALA1D / SCALA2D reference weight from Eq. (2):

        w0(d) = min {
            w :
            2w - f(w) + 1 >= (d+1)/2
        }.
    """

    target = (
        d + 1
    ) // 2

    w = 1

    while True:

        if (
            2 * w
            - f_integer(w)
            + 1
            >= target
        ):
            return w

        w += 1


# =============================================================================
# Edge indexing
# =============================================================================

def edge_to_index(
    orientation: int,
    y: int,
    x: int,
    d: int,
) -> int:
    """
    orientation:
        0 = horizontal h[y,x]
        1 = vertical   v[y,x]
    """

    return (
        orientation * d * d
        + y * d
        + x
    )


def index_to_edge(
    idx: int,
    d: int,
):
    block = d * d

    orientation = (
        idx // block
    )

    r = (
        idx % block
    )

    y = (
        r // d
    )

    x = (
        r % d
    )

    return (
        int(orientation),
        int(y),
        int(x),
    )


def incident_edges(
    y: int,
    x: int,
    d: int,
):
    """
    Four qubits incident on syndrome/check node (y,x).

    With the convention used in run_pheno_hazard.py:

        h[y,x] touches (y,x) and (y,x-1)
        v[y,x] touches (y,x) and (y-1,x)
    """

    xp = (
        x + 1
    ) % d

    yp = (
        y + 1
    ) % d

    return [
        edge_to_index(
            0,
            y,
            x,
            d,
        ),
        edge_to_index(
            0,
            y,
            xp,
            d,
        ),
        edge_to_index(
            1,
            y,
            x,
            d,
        ),
        edge_to_index(
            1,
            yp,
            x,
            d,
        ),
    ]


def edge_endpoints(
    idx: int,
    d: int,
):
    orientation, y, x = (
        index_to_edge(
            idx,
            d,
        )
    )

    if orientation == 0:

        # h[y,x]
        return [
            (
                y,
                x,
            ),
            (
                y,
                (x - 1) % d,
            ),
        ]

    # v[y,x]
    return [
        (
            y,
            x,
        ),
        (
            (y - 1) % d,
            x,
        ),
    ]


def neighboring_edges(
    idx: int,
    d: int,
):
    """
    All distinct edges sharing a check node with idx.
    """

    out = set()

    for y, x in edge_endpoints(
        idx,
        d,
    ):

        out.update(
            incident_edges(
                y,
                x,
                d,
            )
        )

    out.discard(
        idx
    )

    return list(
        out
    )


# =============================================================================
# Candidate generation
# =============================================================================

def random_uniform_candidate(
    d: int,
    weight: int,
    rng,
):
    n_edges = (
        2 * d * d
    )

    return tuple(
        sorted(
            rng.choice(
                n_edges,
                size=weight,
                replace=False,
            ).tolist()
        )
    )


def grow_connected_cluster(
    d: int,
    size: int,
    rng,
    forbidden=None,
):
    """
    Grow one connected edge cluster using edge adjacency.
    """

    if forbidden is None:
        forbidden = set()

    n_edges = (
        2 * d * d
    )

    available = [
        e
        for e in range(
            n_edges
        )
        if e not in forbidden
    ]

    if not available:
        return set()

    cluster = {
        int(
            rng.choice(
                available
            )
        )
    }

    while (
        len(cluster) < size
    ):

        source = int(
            rng.choice(
                list(cluster)
            )
        )

        choices = [
            e
            for e in neighboring_edges(
                source,
                d,
            )
            if (
                e not in cluster
                and e not in forbidden
            )
        ]

        if not choices:

            # Try another source.
            all_choices = set()

            for e in cluster:

                all_choices.update(
                    neighboring_edges(
                        e,
                        d,
                    )
                )

            choices = [
                e
                for e in all_choices
                if (
                    e not in cluster
                    and e not in forbidden
                )
            ]

        if not choices:

            # Rare fallback.
            choices = [
                e
                for e in available
                if e not in cluster
            ]

        cluster.add(
            int(
                rng.choice(
                    choices
                )
            )
        )

    return cluster


def random_connected_candidate(
    d: int,
    weight: int,
    rng,
):
    return tuple(
        sorted(
            grow_connected_cluster(
                d=d,
                size=weight,
                rng=rng,
            )
        )
    )


def random_multicluster_candidate(
    d: int,
    weight: int,
    rng,
):
    """
    Split total weight into 2-4 independently grown clusters.
    """

    if weight <= 2:
        return random_connected_candidate(
            d,
            weight,
            rng,
        )

    max_clusters = min(
        4,
        weight,
    )

    n_clusters = int(
        rng.integers(
            2,
            max_clusters + 1,
        )
    )

    # Random positive composition of weight.
    cuts = sorted(
        rng.choice(
            np.arange(
                1,
                weight,
            ),
            size=n_clusters - 1,
            replace=False,
        ).tolist()
    )

    bounds = (
        [0]
        + cuts
        + [weight]
    )

    sizes = [
        bounds[i + 1]
        - bounds[i]
        for i in range(
            n_clusters
        )
    ]

    used = set()

    for size in sizes:

        cluster = grow_connected_cluster(
            d=d,
            size=size,
            rng=rng,
            forbidden=used,
        )

        used.update(
            cluster
        )

    return tuple(
        sorted(
            used
        )
    )


def random_mixed_2d_candidate(
    d: int,
    weight: int,
    rng,
):
    """
    Force presence of both horizontal and vertical edges.

    Start from a connected cluster, then repair if it accidentally
    lies entirely in one orientation.
    """

    c = set(
        random_connected_candidate(
            d,
            weight,
            rng,
        )
    )

    orientations = {
        index_to_edge(
            e,
            d,
        )[0]
        for e in c
    }

    if (
        weight >= 2
        and len(orientations) == 1
    ):

        target_orientation = (
            1
            - next(
                iter(
                    orientations
                )
            )
        )

        remove_edge = int(
            rng.choice(
                list(c)
            )
        )

        c.remove(
            remove_edge
        )

        candidates = [
            e
            for e in range(
                target_orientation
                * d * d,
                (
                    target_orientation
                    + 1
                )
                * d
                * d,
            )
            if e not in c
        ]

        # Prefer an edge adjacent to the remaining set.
        local = []

        for e in c:
            for n in neighboring_edges(
                e,
                d,
            ):
                if (
                    n in candidates
                    and n not in c
                ):
                    local.append(
                        n
                    )

        if local:

            c.add(
                int(
                    rng.choice(
                        local
                    )
                )
            )

        else:

            c.add(
                int(
                    rng.choice(
                        candidates
                    )
                )
            )

    return tuple(
        sorted(
            c
        )
    )


def generate_random_candidate(
    d: int,
    weight: int,
    rng,
):
    """
    Mixture of proposal families.

    Genuine 2D structures are deliberately overrepresented.
    """

    r = rng.random()

    if r < 0.15:

        return (
            random_uniform_candidate(
                d,
                weight,
                rng,
            ),
            "uniform",
        )

    if r < 0.45:

        return (
            random_connected_candidate(
                d,
                weight,
                rng,
            ),
            "connected",
        )

    if r < 0.75:

        return (
            random_multicluster_candidate(
                d,
                weight,
                rng,
            ),
            "multicluster",
        )

    return (
        random_mixed_2d_candidate(
            d,
            weight,
            rng,
        ),
        "mixed2d",
    )


# =============================================================================
# Candidate mutation
# =============================================================================

def mutate_candidate(
    candidate,
    d: int,
    rng,
):
    """
    Preserve the exact Hamming weight.

    Mostly local mutations, with occasional global jumps.
    """

    c = set(
        candidate
    )

    remove_edge = int(
        rng.choice(
            list(c)
        )
    )

    c.remove(
        remove_edge
    )

    # 80% local replacement, 20% global replacement.
    if (
        c
        and rng.random() < 0.80
    ):

        anchor = int(
            rng.choice(
                list(c)
            )
        )

        choices = [
            e
            for e in neighboring_edges(
                anchor,
                d,
            )
            if e not in c
        ]

    else:

        choices = []

    if not choices:

        n_edges = (
            2 * d * d
        )

        choices = [
            e
            for e in range(
                n_edges
            )
            if e not in c
        ]

    c.add(
        int(
            rng.choice(
                choices
            )
        )
    )

    return tuple(
        sorted(
            c
        )
    )


# =============================================================================
# Packed injection
# =============================================================================

def inject_candidates(
    state,
    candidates,
    d: int,
):
    """
    Put one different candidate in each packed lane.
    """

    for lane, candidate in enumerate(
        candidates
    ):

        bit = (
            np.uint64(1)
            << np.uint64(
                lane
            )
        )

        for idx in candidate:

            orientation, y, x = (
                index_to_edge(
                    idx,
                    d,
                )
            )

            if orientation == 0:

                state["h"][
                    y,
                    x,
                ] ^= bit

            else:

                state["v"][
                    y,
                    x,
                ] ^= bit


# =============================================================================
# Physical weight per lane
# =============================================================================

def data_weights(
    state,
    lanes,
):
    """
    Current number of data-qubit errors in each lane.
    """

    words = np.concatenate(
        [
            state["h"].reshape(-1),
            state["v"].reshape(-1),
        ]
    )

    shifts = (
        lanes.astype(
            np.uint64
        )[:, None]
    )

    bits = (
        (
            words[None, :]
            >> shifts
        )
        & np.uint64(1)
    )

    return bits.sum(
        axis=1
    ).astype(
        np.int64
    )


# =============================================================================
# Packed deterministic evaluator
# =============================================================================

def evaluate_batch_one_phase(
    candidates,
    d: int,
    reset_period: int,
    reset_phase: int,
    max_rounds: int,
    matching,
):
    """
    Evaluate up to 64 candidates simultaneously.

    No additional data or measurement noise is applied.

    Returns:
        failed[i]
        fail_time[i]
        max_weight[i]
    """

    n = len(
        candidates
    )

    if n == 0:
        return (
            np.zeros(
                0,
                dtype=bool,
            ),
            np.zeros(
                0,
                dtype=float,
            ),
            np.zeros(
                0,
                dtype=int,
            ),
        )

    if n > 64:
        raise ValueError(
            "evaluate_batch_one_phase supports at most 64 candidates."
        )

    state = make_group(
        d=d,
        n_lanes=n,
        seed=1,
    )

    inject_candidates(
        state=state,
        candidates=candidates,
        d=d,
    )

    state[
        "reset_clock"
    ] = int(
        reset_phase
    )

    failed_out = np.zeros(
        n,
        dtype=bool,
    )

    fail_time = np.full(
        n,
        np.nan,
        dtype=float,
    )

    max_weight = np.full(
        n,
        -1,
        dtype=int,
    )

    # Candidate lanes remain alive only until first logical failure.
    for t in range(
        1,
        max_rounds + 1,
    ):

        if (
            state["alive"]
            == 0
        ):
            break

        advance_group(
            state=state,
            d=d,
            p_threshold=0,
            q_threshold=0,
            reset_period=reset_period,
        )

        lanes = active_lanes(
            state["alive"],
            state[
                "n_lanes"
            ],
        )

        if len(lanes) == 0:
            break

        # -------------------------------------------------------------
        # Fitness:
        # maximum physical error weight reached before failure.
        # -------------------------------------------------------------

        weights = data_weights(
            state,
            lanes,
        )

        for j, lane in enumerate(
            lanes
        ):

            lane = int(
                lane
            )

            max_weight[
                lane
            ] = max(
                max_weight[
                    lane
                ],
                int(
                    weights[j]
                ),
            )

        # -------------------------------------------------------------
        # MWPM logical checker
        # -------------------------------------------------------------

        syndromes = unpack_syndrome(
            state[
                "true_defect"
            ],
            lanes,
        )

        actual = actual_logicals(
            state["h"],
            state["v"],
            lanes,
        )

        predicted = np.asarray(
            matching.decode_batch(
                syndromes
            ),
            dtype=np.uint8,
        )

        if predicted.ndim == 1:
            predicted = (
                predicted[
                    :,
                    None,
                ]
            )

        logical_fail = np.any(
            predicted
            != actual,
            axis=1,
        )

        if not np.any(
            logical_fail
        ):
            continue

        failed_word = np.uint64(
            0
        )

        for lane in lanes[
            logical_fail
        ]:

            lane = int(
                lane
            )

            failed_out[
                lane
            ] = True

            fail_time[
                lane
            ] = t

            failed_word |= (
                np.uint64(1)
                << np.uint64(
                    lane
                )
            )

        state[
            "alive"
        ] &= ~failed_word

    # Lanes which never evolved due to pathological input:
    max_weight[
        max_weight < 0
    ] = 0

    return (
        failed_out,
        fail_time,
        max_weight,
    )


def evaluate_candidates(
    candidates,
    d: int,
    reset_period: int,
    phases,
    max_rounds: int,
    matching,
):
    """
    Evaluate candidates over several reset phases.

    Fitness is the largest physical weight reached over all tested phases.

    A candidate counts as failure if it fails at any tested phase.
    """

    n = len(
        candidates
    )

    failed_any = np.zeros(
        n,
        dtype=bool,
    )

    earliest = np.full(
        n,
        np.nan,
        dtype=float,
    )

    best_phase = np.full(
        n,
        -1,
        dtype=int,
    )

    fitness = np.zeros(
        n,
        dtype=int,
    )

    for phase in phases:

        for start in range(
            0,
            n,
            64,
        ):

            stop = min(
                start + 64,
                n,
            )

            batch = candidates[
                start:stop
            ]

            (
                failed,
                fail_time,
                max_weight,
            ) = evaluate_batch_one_phase(
                candidates=batch,
                d=d,
                reset_period=reset_period,
                reset_phase=phase,
                max_rounds=max_rounds,
                matching=matching,
            )

            fitness[
                start:stop
            ] = np.maximum(
                fitness[
                    start:stop
                ],
                max_weight,
            )

            for j in range(
                stop - start
            ):

                idx = (
                    start + j
                )

                if not failed[j]:
                    continue

                if (
                    not failed_any[idx]
                    or (
                        np.isfinite(
                            fail_time[j]
                        )
                        and (
                            not np.isfinite(
                                earliest[idx]
                            )
                            or fail_time[j]
                            < earliest[idx]
                        )
                    )
                ):

                    failed_any[
                        idx
                    ] = True

                    earliest[
                        idx
                    ] = (
                        fail_time[j]
                    )

                    best_phase[
                        idx
                    ] = int(
                        phase
                    )

    return (
        failed_any,
        earliest,
        best_phase,
        fitness,
    )


# =============================================================================
# Full phase verification
# =============================================================================

def verify_candidate_all_phases(
    candidate,
    d: int,
    reset_period: int,
    max_rounds: int,
    matching,
):
    """
    Once a witness is found, check every reset phase exactly.
    """

    failing_phases = []

    earliest = None

    for phase in range(
        reset_period
    ):

        (
            failed,
            times,
            _,
        ) = evaluate_batch_one_phase(
            candidates=[
                candidate
            ],
            d=d,
            reset_period=reset_period,
            reset_phase=phase,
            max_rounds=max_rounds,
            matching=matching,
        )

        if failed[0]:

            t = int(
                times[0]
            )

            failing_phases.append(
                (
                    int(phase),
                    t,
                )
            )

            if (
                earliest is None
                or t < earliest
            ):
                earliest = t

    return (
        failing_phases,
        earliest,
    )


# =============================================================================
# JSON witness representation
# =============================================================================

def candidate_to_json(
    candidate,
    d: int,
):
    edges = []

    for idx in candidate:

        orientation, y, x = (
            index_to_edge(
                idx,
                d,
            )
        )

        edges.append(
            {
                "type":
                    (
                        "h"
                        if orientation == 0
                        else "v"
                    ),

                "y":
                    int(y),

                "x":
                    int(x),
            }
        )

    return json.dumps(
        edges,
        separators=(
            ",",
            ":",
        ),
    )


# =============================================================================
# Search one (d, tR, weight)
# =============================================================================

def search_weight(
    d: int,
    weight: int,
    reset_period: int,
    random_trials: int,
    generations: int,
    population: int,
    elite: int,
    phase_samples: int,
    max_rounds: int,
    rng,
):
    matching = build_matching(
        d
    )

    # -------------------------------------------------------------------------
    # Cheap phase subset during search.
    # Full phase scan is done only after finding a witness.
    # -------------------------------------------------------------------------

    if (
        phase_samples >= reset_period
    ):

        phases = list(
            range(
                reset_period
            )
        )

    else:

        phases = sorted(
            set(
                np.linspace(
                    0,
                    reset_period - 1,
                    phase_samples,
                    dtype=int,
                ).tolist()
            )
        )

    print(
        f"        search phases = {phases}"
    )

    # =========================================================================
    # Stage 1: random structured search
    # =========================================================================

    tested = 0

    best_candidate = None
    best_fitness = -1
    best_family = None

    while (
        tested < random_trials
    ):

        batch_size = min(
            population,
            random_trials
            - tested,
        )

        candidates = []
        families = []

        seen = set()

        while (
            len(candidates)
            < batch_size
        ):

            candidate, family = (
                generate_random_candidate(
                    d=d,
                    weight=weight,
                    rng=rng,
                )
            )

            if candidate in seen:
                continue

            seen.add(
                candidate
            )

            candidates.append(
                candidate
            )

            families.append(
                family
            )

        (
            failed,
            times,
            fail_phases,
            fitness,
        ) = evaluate_candidates(
            candidates=candidates,
            d=d,
            reset_period=reset_period,
            phases=phases,
            max_rounds=max_rounds,
            matching=matching,
        )

        tested += (
            len(candidates)
        )

        if np.any(
            failed
        ):

            inds = np.where(
                failed
            )[0]

            # Earliest logical failure among witnesses.
            finite_times = np.where(
                np.isfinite(
                    times[
                        inds
                    ]
                ),
                times[
                    inds
                ],
                np.inf,
            )

            idx = int(
                inds[
                    np.argmin(
                        finite_times
                    )
                ]
            )

            witness = (
                candidates[
                    idx
                ]
            )

            full_phases, earliest = (
                verify_candidate_all_phases(
                    candidate=witness,
                    d=d,
                    reset_period=reset_period,
                    max_rounds=max_rounds,
                    matching=matching,
                )
            )

            return {
                "found":
                    True,

                "candidate":
                    witness,

                "family":
                    families[
                        idx
                    ],

                "search_phase":
                    int(
                        fail_phases[
                            idx
                        ]
                    ),

                "failure_time":
                    int(
                        times[
                            idx
                        ]
                    ),

                "failing_phases":
                    full_phases,

                "earliest_all_phases":
                    earliest,

                "max_weight":
                    int(
                        fitness[
                            idx
                        ]
                    ),

                "tested":
                    tested,

                "stage":
                    "random",
            }

        # Track strongest nonfailure for evolutionary seeding.
        idx = int(
            np.argmax(
                fitness
            )
        )

        if (
            fitness[
                idx
            ]
            > best_fitness
        ):

            best_fitness = int(
                fitness[
                    idx
                ]
            )

            best_candidate = (
                candidates[
                    idx
                ]
            )

            best_family = (
                families[
                    idx
                ]
            )

        print(
            f"            random tested "
            f"{tested:7d}/{random_trials} "
            f"best growth={best_fitness}"
        )

    # =========================================================================
    # Stage 2: evolutionary local search
    # =========================================================================

    if best_candidate is None:

        best_candidate, best_family = (
            generate_random_candidate(
                d=d,
                weight=weight,
                rng=rng,
            )
        )

    current_elites = [
        best_candidate
    ]

    total_evolution_tested = 0

    for generation in range(
        generations
    ):

        candidates = []
        seen = set()

        # Retain elites themselves.
        for c in current_elites:

            if c not in seen:

                candidates.append(
                    c
                )

                seen.add(
                    c
                )

        # Fill population with mutations of elites.
        while (
            len(candidates)
            < population
        ):

            parent = current_elites[
                int(
                    rng.integers(
                        0,
                        len(
                            current_elites
                        ),
                    )
                )
            ]

            child = mutate_candidate(
                candidate=parent,
                d=d,
                rng=rng,
            )

            if child in seen:
                continue

            seen.add(
                child
            )

            candidates.append(
                child
            )

        (
            failed,
            times,
            fail_phases,
            fitness,
        ) = evaluate_candidates(
            candidates=candidates,
            d=d,
            reset_period=reset_period,
            phases=phases,
            max_rounds=max_rounds,
            matching=matching,
        )

        total_evolution_tested += (
            len(candidates)
        )

        if np.any(
            failed
        ):

            inds = np.where(
                failed
            )[0]

            finite_times = np.where(
                np.isfinite(
                    times[
                        inds
                    ]
                ),
                times[
                    inds
                ],
                np.inf,
            )

            idx = int(
                inds[
                    np.argmin(
                        finite_times
                    )
                ]
            )

            witness = (
                candidates[
                    idx
                ]
            )

            full_phases, earliest = (
                verify_candidate_all_phases(
                    candidate=witness,
                    d=d,
                    reset_period=reset_period,
                    max_rounds=max_rounds,
                    matching=matching,
                )
            )

            return {
                "found":
                    True,

                "candidate":
                    witness,

                "family":
                    "evolution",

                "search_phase":
                    int(
                        fail_phases[
                            idx
                        ]
                    ),

                "failure_time":
                    int(
                        times[
                            idx
                        ]
                    ),

                "failing_phases":
                    full_phases,

                "earliest_all_phases":
                    earliest,

                "max_weight":
                    int(
                        fitness[
                            idx
                        ]
                    ),

                "tested":
                    (
                        tested
                        + total_evolution_tested
                    ),

                "stage":
                    "evolution",
            }

        order = np.argsort(
            fitness
        )[
            ::-1
        ]

        elite_indices = order[
            :min(
                elite,
                len(order),
            )
        ]

        current_elites = [
            candidates[
                int(i)
            ]
            for i in elite_indices
        ]

        gen_best = int(
            fitness[
                elite_indices[
                    0
                ]
            ]
        )

        if gen_best > best_fitness:

            best_fitness = (
                gen_best
            )

            best_candidate = (
                current_elites[
                    0
                ]
            )

            best_family = (
                "evolution"
            )

        print(
            f"            generation "
            f"{generation + 1:3d}/{generations} "
            f"best growth={gen_best}"
        )

    return {
        "found":
            False,

        "candidate":
            best_candidate,

        "family":
            best_family,

        "search_phase":
            -1,

        "failure_time":
            np.nan,

        "failing_phases":
            [],

        "earliest_all_phases":
            np.nan,

        "max_weight":
            int(
                best_fitness
            ),

        "tested":
            (
                tested
                + total_evolution_tested
            ),

        "stage":
            "none",
    }


# =============================================================================
# Append/replace
# =============================================================================

def append_replace(
    path: Path,
    new_df: pd.DataFrame,
):
    key = [
        "d",
        "reset_period",
        "weight",
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not path.exists():

        new_df.to_csv(
            path,
            index=False,
        )

        return

    old = pd.read_csv(
        path
    )

    new_keys = set(
        new_df[
            key
        ].itertuples(
            index=False,
            name=None,
        )
    )

    keep = [
        tuple(row)
        not in new_keys
        for row in old[
            key
        ].itertuples(
            index=False,
            name=None,
        )
    ]

    old = old[
        np.asarray(
            keep,
            dtype=bool,
        )
    ]

    out = pd.concat(
        [
            old,
            new_df,
        ],
        ignore_index=True,
    )

    out = out.sort_values(
        [
            "d",
            "reset_period",
            "weight",
        ]
    )

    out.to_csv(
        path,
        index=False,
    )


# =============================================================================
# CLI
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--d",
        nargs="+",
        type=int,
        default=[
            21,
            41,
        ],
    )

    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=[
            0.20,
            0.25,
            0.30,
            0.35,
        ],
    )

    parser.add_argument(
        "--below",
        nargs="+",
        type=int,
        default=[
            3,
            2,
            1,
        ],
        help=(
            "Search weights w0-below. "
            "Default tests w0-3, w0-2, w0-1."
        ),
    )

    parser.add_argument(
        "--random-trials",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--population",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--elite",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--phase-samples",
        type=int,
        default=4,
        help=(
            "Number of reset phases sampled during search. "
            "A found witness is subsequently checked at every phase."
        ),
    )

    parser.add_argument(
        "--rounds-multiplier",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/pheno/"
            "scala2d_low_weight_witness_search.csv"
        ),
    )

    args = parser.parse_args()

    rng = np.random.default_rng(
        args.seed
    )

    rows = []

    print()
    print("=" * 110)
    print(
        "SCALA2D LOW-WEIGHT 2D WITNESS SEARCH"
    )
    print("=" * 110)

    for d in args.d:

        w0 = witness_weight(
            d
        )

        print()
        print(
            f"d={d}: Eq.(2) reference "
            f"w0={w0}"
        )

        max_rounds = max(
            d,
            int(
                math.ceil(
                    args.rounds_multiplier
                    * d
                )
            ),
        )

        weights = sorted(
            {
                w0 - offset
                for offset
                in args.below
                if (
                    w0 - offset
                    > 0
                )
            }
        )

        for alpha in args.alpha:

            reset_period = max(
                1,
                int(
                    round(
                        alpha
                        * d
                    )
                ),
            )

            actual_alpha = (
                reset_period
                / d
            )

            print()
            print(
                f"    alpha={alpha:.3f} "
                f"tR={reset_period} "
                f"(actual {actual_alpha:.5f})"
            )

            # Search smallest requested weight first.
            for weight in weights:

                print()
                print(
                    f"        weight={weight} "
                    f"(w0-{w0-weight})"
                )

                result = search_weight(
                    d=d,
                    weight=weight,
                    reset_period=reset_period,
                    random_trials=(
                        args.random_trials
                    ),
                    generations=(
                        args.generations
                    ),
                    population=(
                        args.population
                    ),
                    elite=(
                        args.elite
                    ),
                    phase_samples=(
                        args.phase_samples
                    ),
                    max_rounds=(
                        max_rounds
                    ),
                    rng=rng,
                )

                if result[
                    "found"
                ]:

                    print()
                    print(
                        "        >>> FAILURE FOUND"
                    )

                    print(
                        f"            weight       = "
                        f"{weight}"
                    )

                    print(
                        f"            family       = "
                        f"{result['family']}"
                    )

                    print(
                        f"            fail time    = "
                        f"{result['failure_time']}"
                    )

                    print(
                        f"            max wt       = "
                        f"{result['max_weight']}"
                    )

                    print(
                        f"            phases       = "
                        f"{result['failing_phases']}"
                    )

                else:

                    print()
                    print(
                        "        no failure found"
                    )

                    print(
                        f"            candidates   = "
                        f"{result['tested']}"
                    )

                    print(
                        f"            best growth  = "
                        f"{result['max_weight']}"
                    )

                candidate_json = (
                    candidate_to_json(
                        result[
                            "candidate"
                        ],
                        d,
                    )
                    if result[
                        "candidate"
                    ] is not None
                    else ""
                )

                rows.append(
                    {
                        "d":
                            int(d),

                        "w0_reference":
                            int(w0),

                        "weight":
                            int(weight),

                        "alpha_requested":
                            float(alpha),

                        "reset_period":
                            int(
                                reset_period
                            ),

                        "actual_alpha":
                            float(
                                actual_alpha
                            ),

                        "found_failure":
                            bool(
                                result[
                                    "found"
                                ]
                            ),

                        "search_stage":
                            result[
                                "stage"
                            ],

                        "family":
                            result[
                                "family"
                            ],

                        "failure_time":
                            result[
                                "failure_time"
                            ],

                        "earliest_all_phases":
                            result[
                                "earliest_all_phases"
                            ],

                        "failing_phases":
                            json.dumps(
                                result[
                                    "failing_phases"
                                ]
                            ),

                        "max_error_weight_reached":
                            int(
                                result[
                                    "max_weight"
                                ]
                            ),

                        "candidates_tested":
                            int(
                                result[
                                    "tested"
                                ]
                            ),

                        "max_rounds":
                            int(
                                max_rounds
                            ),

                        "witness":
                            candidate_json,
                    }
                )

    new_df = pd.DataFrame(
        rows
    )

    append_replace(
        path=args.output,
        new_df=new_df,
    )

    print()
    print("=" * 110)
    print(
        f"Saved: {args.output}"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
