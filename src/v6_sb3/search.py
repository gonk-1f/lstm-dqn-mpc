"""Project search space and a completion-first Validation objective."""
from math import atan, pi


def sample_parameters(trial):
    """Tune numerical DQN settings; keep task/reward/TD semantics frozen."""
    width = trial.suggest_categorical('network_width', [128, 256])
    return dict(
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 3e-4, log=True),
        batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]),
        target_update_interval=trial.suggest_categorical('target_update_interval', [500, 1000, 2000]),
        buffer_size=trial.suggest_categorical('buffer_size', [100_000, 300_000, 1_000_000]),
        policy_kwargs=dict(net_arch=[width, width]))


def validation_objective(summary, *, cost_scale=100_000.):
    """Strict completion tiers; compare coverage until all voyages complete.

    Incomplete: k + 0.5*coverage. Prefix costs never reward early termination.
    Complete: n + 1 + bounded monotonic economic-cost preference (range 0..1).
    The cost transform accepts signed SOC-equivalent economic costs, without
    clipping away charging credits. It is not a training reward/normalization.
    """
    completed, count = summary['completed'], summary['episodes']
    if completed < count:
        return float(completed + .5 * summary['mean_duration_coverage'])
    cost = summary['fixed_complete_set_cost_cny']
    return float(count + 1. + .5 - atan(cost/cost_scale)/pi)
