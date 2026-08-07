# Reinforcement Learning vs Sequential Bandits in E-Commerce Customer Journeys

## 1. Thesis Direction

A strong and defensible thesis question is:

> **Under what customer-journey conditions does full sequential reinforcement learning outperform sequential or contextual bandit methods in e-commerce personalization?**

This is preferable to assuming that full RL is always better. The more rigorous framing is that full RL should outperform bandits primarily when customer interactions have meaningful **state transitions**, **delayed rewards**, and **long-term consequences**.

## 2. Core Conceptual Distinction

| Approach | Main assumption | E-commerce interpretation |
|---|---|---|
| A/B testing | Fixed policy comparison | Compare two or more predefined website variants |
| Contextual bandit | One-step decision with context | Choose the best product, banner, recommendation, or offer for the current user/session |
| Sequential bandit | Multiple local bandit decisions across a journey | Choose locally optimal actions at each stage without explicitly modeling long-term state transitions |
| Full reinforcement learning | Actions affect future states and future rewards | Optimize a sequence of recommendations, offers, discounts, and messages across the customer journey |
| Offline RL | Learn policies from logged trajectories | Use historical customer journeys to learn long-term policies without direct online exploration |

The central academic distinction is:

```text
Bandit objective:
Choose the best action now.

RL objective:
Choose the best action now considering how it affects future states and cumulative reward.
```

## 3. Why Bandits Are a Good Stepping Stone

Contextual bandits are a practical intermediate step between static supervised recommendation and full RL because they provide:

```text
controlled exploration
known action propensities
less biased logged feedback
online learning with lower complexity
strong baselines for RL comparison
safer deployment than full sequential RL
```

Bandits are especially appropriate when the decision is:

```text
Given this user/session/product context, which action should I take now?
```

Examples:

| E-commerce use case | Bandit action |
|---|---|
| Product recommendation | Which item/category/banner to show |
| Promotion selection | Which discount or offer to present |
| Email/push personalization | Which message variant to send |
| Ranking module | Which product order or carousel to display |
| Checkout intervention | Which trust/urgency/free-shipping message to show |

However, contextual bandits do **not** fully model long-term customer dynamics. They are best viewed as a **controlled exploration and data-collection layer**, not as a replacement for sequential RL.

## 4. When Full RL Should Outperform Bandits

Full sequential RL is most likely to outperform sequential bandits when:

```text
actions affect future user state
rewards are delayed
short-term and long-term objectives conflict
there are long-term side effects
customer lifetime value matters
```

Examples:

```text
A discount increases immediate conversion but increases future discount expectation.

An irrelevant recommendation reduces future browsing intent.

A premium recommendation may lower immediate click-through rate but increase average order value.

A checkout message may reduce bounce now and increase trust in later sessions.
```

Sequential bandits are likely to remain competitive when:

```text
rewards are immediate
state transitions are weak
decisions are approximately independent
sample efficiency matters more than long-term optimality
interpretability and operational simplicity are priorities
```

## 5. Suggested Hypothesis

A strong thesis hypothesis is:

> **Full sequential reinforcement learning outperforms sequential bandit methods when customer journey decisions exhibit meaningful delayed effects and state dependence. Sequential bandits remain competitive when rewards are immediate, transitions are weak, or the journey can be decomposed into independent local decisions.**

This framing is robust because it allows either outcome to be academically valuable.

## 6. Literature Anchors

### Bandits in E-Commerce

**Liu & Li — “A Map of Bandits for E-commerce”**  
Useful as the main domain-specific source. It maps bandit methods to e-commerce problems such as recommendation, advertising, pricing, revenue management, and inventory decisions.

URL: https://assets.amazon.science/92/9c/e2ab8a7640daabae51f87942a89a/a-map-of-bandits-for-ecommerce.pdf

### Contextual Bandits for Recommendation

**Li et al. — “A Contextual-Bandit Approach to Personalized News Article Recommendation”**  
Classic paper for personalized recommendation using contextual bandits. Not e-commerce-specific, but foundational for online recommendation with contextual bandit feedback.

URL: https://arxiv.org/abs/1003.0146

### Logged Bandit Feedback and Counterfactual Learning

**Swaminathan & Joachims — “Counterfactual Risk Minimization: Learning from Logged Bandit Feedback”**  
Important for logged propensities, inverse propensity scoring, and learning from bandit-collected data.

URL: https://proceedings.mlr.press/v37/swaminathan15.html

### RL-Based Recommender Systems

**Afsar et al. — “Reinforcement Learning based Recommender Systems: A Survey”**  
Useful for positioning RL recommender systems and discussing sequential user-system interaction and long-term engagement.

URL: https://arxiv.org/abs/2101.06286

### Slate Recommendation as a Bridge to RL

**Ie et al. — “Reinforcement Learning for Slate-based Recommender Systems” / SlateQ**  
Relevant because e-commerce recommendations are often slates rather than single items. Slate recommendation is a natural bridge between contextual bandits and sequential RL.

URL: https://arxiv.org/abs/1905.12767

### Long-Term Engagement RL

**Zou et al. — “Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems”**  
Useful for the argument that immediate metrics such as clicks or orders may not capture long-term value.

URL: https://arxiv.org/pdf/1902.05570

### Simulation Environments

**RecoGym**  
E-commerce-style simulation environment for product recommendation and online advertising, based on user traffic patterns and recommendation responses.

URL: https://arxiv.org/abs/1808.00720

**RecSim / RecSim NG**  
Configurable simulator for sequential recommender systems with latent user state, item familiarity, and response models.

URL: https://arxiv.org/abs/1909.04847  
Repository: https://github.com/google-research/recsim

**Virtual-Taobao**  
Large-scale e-commerce simulation environment based on Taobao data. Strong conceptual fit, but less plug-and-play.

URL: https://arxiv.org/abs/1805.10000

**KuaiSim**  
Modern simulator for RL-based recommender systems.

URL: https://openreview.net/forum?id=dJEjgQcbOt

## 7. Recommended Simulation Strategy

For a master’s thesis, avoid simulating an entire website in excessive detail. Instead, simulate a **controlled decision layer** inside the customer journey.

A suitable simplified customer journey:

```text
landing page
→ product/category recommendation
→ product detail page
→ cart
→ checkout intervention
→ purchase or exit
→ optional retention outcome
```

At each step, the agent chooses an action.

Example actions:

| Journey stage | Possible actions |
|---|---|
| Landing page | Show category A/B/C, deal/no deal |
| Product page | Recommend complementary, substitute, premium, or budget item |
| Cart | Offer discount, free shipping, bundle, or no intervention |
| Checkout | Show urgency message, trust message, or no message |
| Post-purchase | Send retention email type A/B/C |

## 8. State Design

The state should include variables that previous actions can affect.

Possible state variables:

```text
user segment
purchase intent
price sensitivity
brand affinity
cart value
fatigue / annoyance
trust
discount expectation
time step
previous action
session history
```

This is important because RL only has a real advantage if actions modify future states.

Example transition logic:

```text
Relevant recommendations increase purchase intent.
Irrelevant recommendations increase fatigue.
Discounts increase short-term purchase probability.
Repeated discounts increase future discount expectation.
Trust messages may reduce checkout bounce.
Premium products may increase average order value but reduce immediate conversion.
```

## 9. Reward Design

Use a cumulative customer-journey reward rather than only click-through rate.

Example reward:

```text
+0.1 for click
+1.0 for add-to-cart
+5.0 for purchase
+margin component
-0.5 for bounce
-1.0 for annoyance
-2.0 for return/refund
-3.0 for excessive discount cost
```

Do not evaluate only clicks, because click-only rewards can make policies myopic and may favor bandits or shallow engagement optimization.

A normalized reward is usually easier to train:

```python
reward = max(-10.0, min(raw_reward, 10.0)) / 10.0
```

## 10. Algorithms to Compare

Recommended baselines:

| Category | Algorithms |
|---|---|
| Non-learning baseline | Random policy, fixed heuristic |
| Contextual bandit | Epsilon-greedy, LinUCB, Thompson sampling |
| Sequential bandit | Independent contextual bandit per journey stage |
| Full RL | Q-learning, DQN, PPO, actor-critic depending on environment complexity |
| Optional oracle | Uses simulator’s true reward model as an upper bound |

The most important comparison is:

```text
Sequential bandit:
Learns locally optimal actions at each customer-journey stage.

Full RL:
Learns actions that maximize cumulative long-term journey reward.
```

## 11. Critical Experimental Scenarios

The thesis should evaluate different environment regimes rather than one fixed environment.

### Scenario A: Immediate Reward, Weak State Transitions

Actions have little effect on future customer state.

Expected result:

```text
Bandits ≈ RL
```

Interpretation: full RL may be unnecessary when decisions are approximately myopic.

### Scenario B: Delayed Reward

Early actions influence later purchase, but immediate feedback is weak.

Expected result:

```text
RL > sequential bandits
```

Interpretation: RL benefits from temporal credit assignment.

### Scenario C: Negative Long-Term Side Effects

Discounts or aggressive interventions improve short-term conversion but harm future value.

Expected result:

```text
RL should learn to avoid overusing harmful actions.
Bandits may over-exploit short-term rewards.
```

Interpretation: RL is useful when actions create future costs.

### Scenario D: Sparse Rewards

Purchases are rare, while clicks are common.

Expected result:

```text
Bandits may learn faster from dense proxy rewards.
RL may need more samples but can optimize long-term return better.
```

Interpretation: compare sample efficiency and final performance separately.

## 12. Evaluation Metrics

Use both algorithmic and business metrics.

### Learning Metrics

```text
average return over time
cumulative regret
sample efficiency
training stability
variance across random seeds
```

### Business Metrics

```text
conversion rate
revenue
margin
average order value
discount cost
bounce rate
retention proxy
refund/return rate
```

### Policy Diagnostics

```text
action distribution by journey stage
discount frequency
performance by user segment
short-term vs long-term reward decomposition
state trajectories under each policy
```

These diagnostics help explain why a method performs better or worse.

## 13. Logged Data Requirements

If using bandits to collect data, log:

```text
timestamp
context/state
available actions
chosen action
selection probability / propensity
observed reward
reward window
policy version
eligible action set
journey/session identifier
```

Propensity logging is essential for:

```text
inverse propensity scoring
doubly robust evaluation
off-policy evaluation
counterfactual policy learning
```

Without propensities, offline comparison becomes much weaker.

## 14. Practical Implementation Notes

### Bandit Mode

A contextual bandit setup uses one decision per interaction:

```text
observe context
choose action
observe reward
update policy
```

This is appropriate for product recommendations, banners, offers, or subject lines when immediate response is the main goal.

### Sequential Bandit Mode

A sequential bandit setup uses independent bandits at each stage:

```text
stage 1 bandit chooses landing page action
stage 2 bandit chooses product page action
stage 3 bandit chooses cart action
stage 4 bandit chooses checkout action
```

Each bandit optimizes a local reward or short-window reward.

### Full RL Mode

A full RL setup treats the customer journey as an MDP:

```text
state_t → action_t → reward_t → state_{t+1}
```

The policy optimizes cumulative return:

```text
G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...
```

This is appropriate when actions affect later states and later rewards.

## 15. Suggested Thesis Structure

```text
1. Introduction
   - E-commerce personalization as sequential decision-making
   - Limits of static recommenders and myopic optimization
   - Research question and contributions

2. Background
   - Multi-armed bandits
   - Contextual bandits
   - Sequential bandits
   - Markov decision processes
   - Reinforcement learning
   - Off-policy evaluation

3. Related Work
   - Bandits in e-commerce
   - RL-based recommender systems
   - Customer journey optimization
   - Simulation environments such as RecoGym and RecSim

4. Methodology
   - Customer journey simulator
   - State/action/reward design
   - Algorithms compared
   - Experimental scenarios

5. Experiments
   - Immediate reward setting
   - Delayed reward setting
   - Discount side-effect setting
   - Sparse reward setting

6. Results
   - Performance comparison
   - Sample efficiency
   - Business metrics
   - Policy diagnostics

7. Discussion
   - When full RL is worth the complexity
   - When sequential bandits are sufficient
   - Deployment risks and limitations

8. Conclusion
   - Summary of findings
   - Practical recommendations
   - Future work
```

## 16. Main Takeaway

The strongest conclusion your thesis can aim for is not simply:

```text
RL is better than bandits.
```

A stronger and more academically credible conclusion is:

```text
Full sequential RL is more effective than sequential bandits when customer-journey decisions have meaningful delayed effects, state dependence, and long-term side effects.

Sequential bandits are often sufficient, simpler, and more sample-efficient when rewards are immediate and journey stages are approximately independent.
```

This makes the thesis balanced, testable, and practically useful for e-commerce personalization.
