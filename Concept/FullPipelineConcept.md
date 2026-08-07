### Phase 1: Exploratory Data Collection via Contextual Bandit (Behavior Policy)
In this initial phase, a Contextual Bandit acts as a safe "behavior policy" to gather the historical data required for full Reinforcement Learning, protecting the shop's baseline revenue while exploring.
* **Action:** Deploy an $\epsilon$-greedy or Thompson Sampling bandit to serve widgets at decision points.
* **Logging Requirement:** The system must log full user trajectories sequentially, rather than isolated events. 
* **Required Data Tuple:** Each log entry must capture $(s_t, a_t, r_t, p_t, s_{t+1})$:
    * **$s_t$ (State):** The complex user session context (e.g., cart value, browsing history).
    * **$a_t$ (Action):** The chosen widget ID.
    * **$r_t$ (Reward):** Immediate interaction (e.g., click or bounce).
    * **$p_t$ (Propensity):** The exact probability that the bandit chose action $a_t$ given state $s_t$ (critical for un-biasing data later).
    * **$s_{t+1}$ (Next State):** How the user's context evolved after the action.

---

### Phase 2: Offline Reinforcement Learning (Policy Training)
This phase bridges the gap between single-step bandit logic and multi-step RL by utilizing the data collected in Phase 1 to safely train an agent without exposing live users to an untrained model.
* **Action:** Utilize Offline RL algorithms (such as Conservative Q-Learning or Offline DQN) on the static dataset of logged trajectories.
* **Objective:** Train an agent to understand delayed credit assignment (e.g., knowing that a discount widget on page 1 influenced a checkout on page 5).
* **Key Mechanism:** Use the recorded propensity scores ($p_t$) to apply Inverse Propensity Scoring (IPS), correcting the inherent bias of the bandit's exploration strategy.

---

### Phase 3: Online Multi-Step RL (Live Optimization)
The fully trained multi-step agent is deployed to the live e-commerce environment to actively manage and optimize the entire customer journey.
* **Action:** Replace the Contextual Bandit with the trained RL agent.
* **Execution:** The agent now selects actions by evaluating the maximum expected cumulative future reward, rather than just immediate clicks.
* **Continuous Learning:** The agent continues to learn online, adapting its policy in real-time to shifting consumer behaviors, seasonal trends, and new inventory.