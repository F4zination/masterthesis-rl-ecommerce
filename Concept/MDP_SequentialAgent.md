# MDP: Sequential Widget Timing Agent

## State Space, Actions, Transitions & Rewards

```mermaid
flowchart TD
    subgraph OBS["Observation Pipeline"]
        O1[Mouse tracking + event log]
        O2[Feature extraction]
        O3[State vector st]
        O1 --> O2 --> O3
    end

    subgraph ENV["User Session States"]
        S0[S0: Session Start]
        S1[S1: Early Browsing]
        S2[S2: Product Exploration]
        S3[S3: High Engagement]
        S4[S4: Cart Consideration]
        S5[S5: Checkout Intent]
        ST[ST: Terminal - Conversion or Abandonment]
    end

    subgraph ACT["Action Space"]
        A0[a0: Wait and observe]
        A1[a1: Show trust badge]
        A2[a2: Show discount nudge]
        A3[a3: Show cart nudge]
        A4[a4: Show frequently bought together]
        A5[a5: Show urgency signal]
    end

    subgraph REW["Reward Signal"]
        R1[r +1.0 : Purchase completed]
        R2[r +0.3 : Widget clicked]
        R3[r +0.1 : Add to cart after widget]
        R4[r -0.1 : Widget shown cost]
        R5[r -0.5 : Abandoned after widget]
    end

    S0 -->|natural navigation| S1
    S1 -->|PDP visit| S2
    S1 -->|idle or exit| ST
    S2 -->|deep scroll or long dwell| S3
    S2 -->|add to cart| S4
    S2 -->|bounce| ST
    S3 -->|add to cart| S4
    S3 -->|continued exploration| S2
    S4 -->|proceed to checkout| S5
    S4 -->|cart abandoned| ST
    S5 -->|order confirmed| ST
    S5 -->|drop-off| ST

    O3 -->|state fed into policy| ACT
    ACT -->|action applied to session| ENV
    ENV -->|next state + reward| OBS

    style S0 fill:#e8f4f8,stroke:#2196F3
    style S1 fill:#e8f4f8,stroke:#2196F3
    style S2 fill:#fff3e0,stroke:#FF9800
    style S3 fill:#fff3e0,stroke:#FF9800
    style S4 fill:#fce4ec,stroke:#E91E63
    style S5 fill:#fce4ec,stroke:#E91E63
    style ST fill:#e8f5e9,stroke:#4CAF50
    style OBS fill:#e0f2f1,stroke:#009688
    style ACT fill:#f3e5f5,stroke:#9C27B0
    style REW fill:#fafafa,stroke:#9E9E9E
```

## MDP Formal Definition

| Component | Definition |
|---|---|
| **State S** | Session trajectory vector: page depth, scroll depth, dwell time, mouse movement cluster, cart total, item count, pages visited, time since last event |
| **Action A** | {wait, trust\_badge, discount\_nudge, cart\_nudge, frequently\_bought\_together, urgency\_signal} |
| **Transition T(s, a, s')** | Stochastic — user behavior in response to widget (or absence of one) |
| **Reward R(s, a, s')** | Shaped: immediate cost on show, positive signal on click/add-to-cart/purchase, negative on post-widget abandonment |
| **Discount γ** | γ < 1 (e.g. 0.95) — later conversions worth slightly less, reflecting session time pressure |
| **Observation Δt** | Periodic polling interval (e.g. every 10–30 seconds) rather than event-triggered |
| **Policy π** | Learned mapping sₜ → aₜ (e.g. DQN, PPO, or recurrent policy for partial observability) |

## Key Difference vs. V2 Bandit

The bandit fires once per fixed trigger point (page load, add-to-cart event) and has no session memory.
The MDP agent **continuously observes** the session state and decides both *when* to act and *what* to show — enabling it to learn that the same widget at different journey stages produces very different outcomes.
