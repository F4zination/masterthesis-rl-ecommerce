# Concept: Simulated Customer Journey Data for RL Training

## 1. Goal
Build a simulation pipeline that generates realistic training trajectories for offline RL before enough real traffic data is available.

Core idea:
- Model different user types (archetypes).
- Give each archetype different transition and action-response probabilities across journey stages.
- Log synthetic trajectories in the same format as real data so they can be used by existing extraction, offline training, and OPE workflows.

## 2. Why Simulation Is Useful for the Thesis
1. Reduces cold-start risk for RL policy initialization.
2. Enables controlled experiments where the true environment logic is known.
3. Allows ablation studies on reward shaping, exploration, and policy robustness.
4. Makes it possible to test rare but important user paths (for example high-intent checkout users).

## 3. High-Level Simulation Design
The simulator should behave like a Markov process with hidden user type.

At session start:
- Sample user type z from P(z).
- Initialize state s_0.

At each step t:
- Environment state is s_t (page stage, engagement, cart state, time features).
- Policy chooses action a_t (widget or no-op).
- Simulator samples next state s_{t+1} and event outcomes based on:
  - user type z,
  - current state s_t,
  - chosen action a_t.
- Compute reward r_t.
- Continue until terminal state (purchase, abandon, timeout).

Output per step:
- trajectory_id, t, state, action, propensity, reward, next_state, done, decision_point, eligible_actions.

## 4. User Archetypes (Initial Set)
Example archetypes for your thesis:
1. Explorer
- Browses many pages, low immediate conversion, high sensitivity to recommendations.

2. Fast Buyer
- Short sessions, fast path to cart/checkout, low tolerance for intrusive widgets.

3. Detailed Comparator
- Revisits PDPs, long dwell time, reacts well to trust and comparison cues.

4. Discount Hunter
- High response to discount banners, higher abandonment without incentives.

5. Window Shopper
- High bounce probability, low purchase probability, occasional engagement events.

Example mixture prior (adjustable):
- Explorer: 0.30
- Fast Buyer: 0.20
- Detailed Comparator: 0.20
- Discount Hunter: 0.15
- Window Shopper: 0.15

## 5. State Space for Simulation
Use state features aligned with your real logging schema:
- decision_point: landing, pdp, scroll_engagement, cart, checkout
- device_type: mobile, tablet, desktop
- traffic_source: direct, search, social, referral
- page_depth_bucket
- dwell_bucket
- scroll_bucket
- cart_total_bucket
- item_count_bucket
- time_since_last_event_bucket
- session_step

Important:
- Keep feature names compatible with your production/offline pipeline to simplify transfer.

## 6. Action Space
Suggested action set:
- no-op
- trending_carousel
- discount_banner
- frequently_bought_together
- trust_badge
- help_popup

Action availability can depend on decision_point.

## 7. Transition and Event Probability Modeling
Use factorized probabilities to keep the simulator interpretable.

For each archetype z define:
1. Stage transition model
- P(next_decision_point | current_decision_point, action, z)

2. Engagement events model
- P(widget_click | s, a, z)
- P(widget_dismiss | s, a, z)
- P(add_to_cart | s, a, z)
- P(checkout_submit | s, a, z)
- P(purchase | s, a, z)
- P(exit_intent | s, a, z)

3. Session termination model
- P(done | s, a, z)

Implementation option:
- Start with lookup tables per archetype and stage.
- Later replace with small logistic models if needed.

## 8. Example Probability Intuition by Archetype
Explorer:
- Higher P(widget_click) for trending_carousel and frequently_bought_together.
- Lower immediate P(purchase), but non-trivial delayed conversion after multiple PDP views.

Fast Buyer:
- High baseline P(add_to_cart) and P(checkout_submit) from PDP/cart.
- Increased P(exit_intent) if help_popup appears too early.

Detailed Comparator:
- Higher dwell and repeated PDP transitions.
- Better response to trust_badge than discount_banner in early stages.

Discount Hunter:
- Strong lift from discount_banner on cart and checkout stages.
- Higher abandon probability if no discount is shown near checkout.

Window Shopper:
- High bounce from landing/pdp.
- Low purchase unless multiple high-value signals are aligned.

## 9. Reward Design in Simulator
Mirror real reward logic as much as possible.

Example shaped reward:
- widget_click: +1.2
- add_to_cart: +1.5
- checkout_submit: +2.0
- purchase: +8.0 plus min(order_total / 100, 4.0)
- widget_dismiss: -0.8
- exit_intent: -0.6
- action costs per widget (no-op cost 0)

Design rule:
- Keep sparse conversion reward dominant so policies do not overfit to click farming.

## 10. Simulation Algorithm (Per Session)
1. Sample user type z and static context (device, source).
2. Set initial state at landing.
3. For t in 0..T_max:
- Determine eligible actions for current decision point.
- Query behavior policy to choose action a_t and propensity p_t.
- Sample engagement events and transition using z-specific probabilities.
- Compute reward from sampled events and action cost.
- Emit logged transition row.
- Stop if terminal condition reached.
4. Save trajectory.

## 11. Behavior Policy for Data Generation
To avoid unrealistic deterministic logs, use a behavior policy with exploration:
- Epsilon-greedy contextual bandit.
- Optionally per-archetype randomized policy parameters.

Why:
- Produces action diversity needed for off-policy evaluation.
- Makes synthetic propensities usable for IPS/SNIPS/DR.

## 12. Dataset Output Specification
Generate files compatible with your existing pipeline:
- simulated_transitions.jsonl
- simulated_transitions_flat.csv
- simulated_summary.json

Required columns:
- trajectory_id
- t
- session_id
- decision_point
- state
- action
- propensity
- eligible_actions
- reward
- next_state
- done
- metadata: user_type, generated_events, order_total_if_any

## 13. Calibration Strategy
Calibrate simulator so synthetic stats are plausible:
1. Match high-level real metrics where available:
- average session length,
- conversion rate,
- add-to-cart rate,
- stage-to-stage drop-off,
- widget click-through rates.

2. Tune archetype priors and conditional probabilities until deviations are acceptable.

3. Keep a calibration report with before/after metric gaps.

## 14. Validation and Stress Tests
Validation checks:
- No invalid transitions (for example checkout before cart without explicit rule).
- Probability mass checks sum to 1.
- Reward bounds sanity.
- Propensity always greater than 0 for logged action.

Stress scenarios:
- Increased mobile traffic.
- Sudden discount sensitivity shift.
- Higher bounce regime.

## 15. How to Use Simulated Data in Training
Recommended thesis sequence:
1. Pretrain policy offline on simulated data.
2. Fine-tune on mixed data (simulated plus real) with lower weight on simulated samples over time.
3. Evaluate with OPE on real logged data only for promotion decisions.
4. Deploy under strict guardrails.

Weighting idea:
- At iteration k, set simulated sample weight w_sim(k) decreasing across retraining cycles.

## 16. Thesis Experiment Plan
Suggested experiments:
1. Baselines
- no-op policy
- heuristic policy
- contextual bandit

2. Policies
- offline RL trained only on simulated data
- offline RL trained on mixed simulated and real data

3. Metrics
- OPE estimates (IPS, SNIPS, DR)
- online proxy metrics in controlled rollout
- robustness by device/source/archetype

4. Ablations
- no archetypes vs archetype-aware simulator
- sparse rewards only vs shaped rewards
- low vs high behavior policy exploration

## 17. Limitations to Report
- Simulator bias: synthetic behavior may not fully match real users.
- Hidden confounders not represented in state.
- Overfitting risk to simulation artifacts.

Mitigation:
- Conservative policy learning,
- real-data OPE gates,
- staged online rollout with rollback criteria.

## 18. Practical Next Steps
1. Define archetype priors and initial probability tables in a config file.
2. Implement a trajectory generator script that emits JSONL transitions.
3. Add calibration notebook/script comparing simulated and real summary metrics.
4. Train offline baseline policy on simulated dataset.
5. Run OPE against real logs before any promotion.

---

This concept gives you a structured and thesis-defensible way to create simulated training data while preserving compatibility with your existing data and RL pipeline.