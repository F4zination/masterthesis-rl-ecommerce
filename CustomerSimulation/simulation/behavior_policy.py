import random as _random_module


class BehaviorPolicy:
    """Uniform epsilon-soft behavior policy for training data generation.

    Uses a uniform distribution over eligible actions so every action has
    equal probability 1/n.  This maximises action coverage and produces
    valid propensities for IPS/SNIPS/DR off-policy evaluation.

    The ``epsilon`` and ``per_archetype_epsilon`` parameters are stored for
    future greedy-arm extensions but do not affect propensity under the
    current uniform implementation.
    """

    def __init__(
        self,
        epsilon: float = 0.2,
        per_archetype_epsilon: dict[str, float] | None = None,
    ) -> None:
        self.epsilon = epsilon
        self.per_archetype_epsilon: dict[str, float] = per_archetype_epsilon or {}

    def select_action(
        self,
        eligible_actions: list[str],
        archetype_name: str | None = None,
        rng: _random_module.Random | None = None,
    ) -> tuple[str, float]:
        """Sample an action uniformly and return (action, propensity).

        Args:
            eligible_actions: Actions valid at the current decision point.
            archetype_name: Used for per-archetype epsilon lookup (reserved).
            rng: Optional seeded Random instance for reproducibility.

        Returns:
            (action, propensity) where propensity = 1 / len(eligible_actions).
        """
        rand = rng or _random_module
        n = len(eligible_actions)
        action = rand.choice(eligible_actions)
        propensity = 1.0 / n
        return action, propensity
