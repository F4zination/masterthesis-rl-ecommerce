# Improvements for V3

## 1. Sequential RL Agent: Temporal Widget Policy

**Current state (V2):** The bandit makes a stateless, per-request decision. It sees a snapshot of the user's context at the moment of the request and picks the best arm — but has no memory of the session so far, no knowledge of what was already shown, and no concept of *when* in the customer journey a widget will be most effective.

**The idea:** Instead of deciding *what* to show at a fixed trigger point, a sequential agent *observes the unfolding session* periodically and learns to decide **when + what** to show:
> "Given that this user has been on the site for 3 minutes, viewed 2 products, and scrolled 60% of the PDP — *now* is the right moment to show a cart nudge."

**Why this is fundamentally different — MDP vs. Bandit:**

| Property | Contextual Bandit (V2) | Sequential RL Agent (V3) |
|---|---|---|
| State | Snapshot at request time | Full session trajectory so far |
| Actions | Which widget variant to show | When + what to show |
| Reward | Immediate (click, purchase) | Delayed (conversion later in journey) |
| Memory | None — each decision is independent | State carries forward across steps |
| Model type | Epsilon-greedy | DQN, PPO, or Recurrent policy |

**Key capability unlocked — temporal credit assignment:** The current bandit partially captures delayed rewards via a 30-minute lookback window, but it cannot learn *that showing a widget earlier or later would have been better*. A sequential agent would learn policies like:
- "Users who see a discount widget *before* viewing 3 products rarely convert — wait longer"
- "Trust badges are most effective on mobile users *right after* they add to cart, not before"

**Connection to MouseTracking work:** The session clustering in `MouseTracking/` (episode sequences, cluster profiles, behavioral segments) provides exactly the **state representations** a sequential agent needs. The trajectory clusters define where a user is in their journey. A natural pipeline:

```
Mouse + event stream → session state vector → RL policy → widget timing + selection
```

**Classification:** Upgrade from a contextual bandit (stateless, single-step) to a full **Markov Decision Process** solved by a sequential RL policy — the bandit becomes the baseline, the RL agent becomes the full solution.
