# Theoretical Framework

Based on the idea of incremental evolution, this framework proposes a 4-step approach to integrate a RL-Agent into a Ecommerce Site.
The four steps are:
- **Phase 1**: heuristic Data collection, which serves as baseline for comparison and to mitigate the cold start (open for debate)
- **Phase 2**: Contextual Multi-Armed Bandits, which introduce controlled exploration and optimize the imideate reward
- **Phase 3**: Offline RL Training, which uses data collected by the MABs and trains a RL-Agent to optimize the cumulative reward.
- **Phase 4**: Full Online RL-Agent, which takes actions in real time and adjusts its policy based on the cumulative reward


## Decision Points

In order to optize the cumulative reward, there is a need for multiple points in the customer journey at which the Agent needs to take a action.
These decision points are specific to each e-commerce site but here are some examples:
- **The Landing Page** -> Possible actions: welcome banner, email-newsletter, no-op, etc.
- **Exploration** -> Triggers after x seconds of no interaction (Recommendation, ChatBot, etc.)
- **Product Detail Page** -> Triggers when viewing a specific item (Urgency Timer, Frequently bought together, no-op, etc.)
- **Add to Cart/ Cart View** -> Possible actions: free shipping, upsell, no-op, etc.
- **Exit intent** -> Triggers when the user is about to leave (discount, trust badges, etc.)

## Rewards

The rewards need to be at the same time frequent enough for the agent to understand that actions have an impact but also specific events need to give huge amounts of rewards. This will lead to a optimized policy.

### Sparse Rewards
While these sparse rewards should be the ultimate goal the agent wants to optimize the cumulative reward. They need to be big enough so that the agent does not add often small rewards to more successfully get a higher score. Also the Reward for checkout should scale with the money spent.
- **Successful Checkout** (+100) 
- **Contact Form filled** (+100)
- **Advertisement clicked** (+100)

There are also negative sparse rewards that the agent should avoid at all costs. They should be harsh but not too harsh because then the agent could thinkt it is best to do nothing because the punishment will kill any progress 
- **Session Timeout** (-10)

### Dense Rewards
The agent shall learn that small coversions are also useful. Therefore small rewards are placed to guide the journey.
- **Adding items to cart**
- **Clicking a displayed Widget**
- **Submitting a email** (if not the ultimate goal)

There are also some rewards that grow
- **Dwell Time** staying on page >30 sec
- **Scrolling** scrolling past 50% of page

Not to forget that there are also penalties to teach the agent what not to do.
- **Widget dismissed** teaches the agent not to spam
- **Leave within 5 sec** 
- **Cost of action** for every widget shown (sometimes doing nothing is best)


## Required Data

| Data | Type | How to obtain |
|:-------|:-------|:--------|
| session_id | string | cookie |
| operating system | enum | parsed from browser |
| device type | enum | via screen resolution |
| traffic source | string | document.referrer ? |
| time | timestamp | datetime.now() |
| page_depth | int | session storage |
| time_on_page | float | timer in js |
| user_type | enum | mouse-tracking |
| past_purchases | int | database |

## Wishlist 
| Data | Type | How to obtain |
|:-------|:-------|:--------|
| gender | enum | ? |
| age | int | ? |
