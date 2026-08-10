"""Тесты src.features.taxonomy_weighted_distance (ablation-рецепт ГИС Интегро)."""
import numpy as np

from src import config, features


def _toy_dist_by_role(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return {role: rng.uniform(0, 1000, n) for role in config.TAXONOMY_TRANSFORMS}


def test_no_override_matches_manual_recipe():
    dist_by_role = _toy_dist_by_role()
    c = features.taxonomy_weighted_distance(dist_by_role)

    columns, weights = [], []
    for role, kind in config.TAXONOMY_TRANSFORMS.items():
        t = features._criterion_transform(dist_by_role[role], kind)
        lo, hi = t.min(), t.max()
        columns.append((t - lo) / (hi - lo))
        weights.append(config.TAXONOMY_WEIGHTS[role])
    expected = (np.abs(np.column_stack(columns)) * np.asarray(weights)).sum(axis=1)
    np.testing.assert_allclose(c, expected)


def test_override_replaces_only_named_role():
    dist_by_role = _toy_dist_by_role()
    baseline = features.taxonomy_weighted_distance(dist_by_role)

    same = features.taxonomy_weighted_distance(
        dist_by_role, overrides={"facies": dist_by_role["facies"]})
    np.testing.assert_allclose(same, baseline)

    other = dist_by_role.copy()
    other["facies"] = dist_by_role["facies"][::-1]
    replaced = features.taxonomy_weighted_distance(dist_by_role, overrides={"facies": other["facies"]})
    manual = features.taxonomy_weighted_distance(other)
    np.testing.assert_allclose(replaced, manual)
    assert not np.allclose(replaced, baseline)


def test_exclude_drops_role_without_renormalizing_weights():
    dist_by_role = _toy_dist_by_role()
    baseline = features.taxonomy_weighted_distance(dist_by_role)
    without_paleo = features.taxonomy_weighted_distance(dist_by_role, exclude={"paleo"})

    manual = {k: v for k, v in dist_by_role.items() if k != "paleo"}
    columns, weights = [], []
    for role, kind in config.TAXONOMY_TRANSFORMS.items():
        if role == "paleo":
            continue
        t = features._criterion_transform(manual[role], kind)
        lo, hi = t.min(), t.max()
        columns.append((t - lo) / (hi - lo))
        weights.append(config.TAXONOMY_WEIGHTS[role])
    expected = (np.abs(np.column_stack(columns)) * np.asarray(weights)).sum(axis=1)
    np.testing.assert_allclose(without_paleo, expected)
    assert not np.allclose(without_paleo, baseline)


def test_constant_override_contributes_zero():
    dist_by_role = _toy_dist_by_role()
    n = len(next(iter(dist_by_role.values())))
    c = features.taxonomy_weighted_distance(
        dist_by_role, overrides={"facies": np.full(n, 5.0)})
    without_facies = {k: v for k, v in dist_by_role.items() if k != "facies"}
    without_facies["facies"] = np.zeros(n)  # min-max нормировка константы -> нули
    expected = features.taxonomy_weighted_distance(without_facies)
    np.testing.assert_allclose(c, expected)
