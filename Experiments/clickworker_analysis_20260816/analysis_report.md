# Clickworker frozen-policy analysis

The main analysis is participant-level intention-to-treat. Quality-filtered results are a sensitivity analysis because duration, navigation, dwell and dismissals can be affected by policy.

## Transition-ordering endpoint

Recorded-landing N = 31
Direction of persona differences in conditional funnel-advance proportions, pooled over policy conditions, against the ordering predicted by the locked simulator reference.

| Contrast | Stage | Sim gap | Human higher | Human lower | Human gap | Newcombe 95% CI | One-sided p | Holds |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| **fastbuyer > windowshopper** | landing_to_pdp | 0.3400 | 1.0000 (n=10) | 0.8000 (n=10) | 0.2000 | [-0.1124, 0.5098] | 0.0569 | no |
| **detailedcomparator > windowshopper** | landing_to_pdp | 0.3130 | 0.8182 (n=11) | 0.8000 (n=10) | 0.0182 | [-0.3099, 0.3544] | 0.4579 | no |
| **fastbuyer > windowshopper** | pdp_to_cart | 0.5074 | 0.9000 (n=10) | 1.0000 (n=8) | NA | NA | NA | not tested |
| fastbuyer > detailedcomparator | landing_to_pdp | 0.0270 | 1.0000 (n=10) | 0.8182 (n=11) | 0.1818 | [-0.1248, 0.4770] | 0.0590 | no |
| fastbuyer > detailedcomparator | pdp_to_cart | 0.1784 | 0.9000 (n=10) | 1.0000 (n=9) | NA | NA | NA | not tested |
| detailedcomparator > windowshopper | pdp_to_cart | 0.3289 | 1.0000 (n=9) | 1.0000 (n=8) | NA | NA | NA | not tested |
| fastbuyer > detailedcomparator | cart_to_checkout | 0.1560 | 0.6667 (n=9) | 0.6667 (n=9) | NA | NA | NA | not tested |
| fastbuyer > windowshopper | cart_to_checkout | 0.4194 | 0.6667 (n=9) | 0.8750 (n=8) | NA | NA | NA | not tested |
| detailedcomparator > windowshopper | cart_to_checkout | 0.2633 | 0.6667 (n=9) | 0.8750 (n=8) | NA | NA | NA | not tested |
| fastbuyer > detailedcomparator | checkout_to_purchase | 0.1714 | 0.8333 (n=6) | 0.8333 (n=6) | NA | NA | NA | not tested |
| fastbuyer > windowshopper | checkout_to_purchase | 0.4673 | 0.8333 (n=6) | 0.7143 (n=7) | NA | NA | NA | not tested |
| detailedcomparator > windowshopper | checkout_to_purchase | 0.2959 | 0.8333 (n=6) | 0.7143 (n=7) | NA | NA | NA | not tested |

Bold contrasts are confirmatory (intersection-union, one-sided alpha = 0.05); the rest are descriptive and a null there is uninformative. The confirmatory ordering claim: NOT ANALYZABLE.

## Primary simulator fidelity

Recorded-landing N = 31
Simulator means are standardized to the recorded joint device/traffic distribution within each randomized cell.

| Outcome | Human-simulation gap | Bootstrap 90% CI | Margin | Equivalent | Mean absolute cell gap | Persona Spearman |
|---|---:|---:|---:|:---:|---:|---:|
| conversion | 0.3744 | [0.2403, 0.5135] | [-0.0500, 0.0500] | no | 0.3744 | 0.0000 |
| session_length_steps | 7.5133 | [6.1619, 8.8913] | [-0.5000, 0.5000] | no | 7.5133 | -0.5000 |
| funnel_depth | 1.4473 | [1.0134, 1.8386] | [-0.5000, 0.5000] | no | 1.4473 | 0.5000 |

The global calibration claim requires equivalence for all three endpoints: NOT ESTABLISHED.

## Intention-to-treat

N = 31

| Outcome | V3-V2 estimate | Robust 95% CI | Bootstrap 95% CI | Randomization p |
|---|---:|---:|---:|---:|
| session_reward | 2.3116 | [-3.2409, 7.8641] | [-2.5439, 7.2282] | 0.4379 |
| conversion | 0.1667 | [-0.1971, 0.5304] | [-0.1667, 0.4889] | 0.4293 |

Cell estimates are descriptive; policy-by-persona interactions are exploratory.

## Recorded-landing policy sensitivity

N = 31

| Outcome | V3-V2 estimate | Robust 95% CI | Bootstrap 95% CI | Randomization p |
|---|---:|---:|---:|---:|
| session_reward | 2.3116 | [-3.2409, 7.8641] | [-2.8410, 7.1515] | 0.4384 |
| conversion | 0.1667 | [-0.1971, 0.5304] | [-0.1558, 0.4889] | 0.4339 |

Cell estimates are descriptive; policy-by-persona interactions are exploratory.

## Quality-filtered sensitivity

N = 20

| Outcome | V3-V2 estimate | Robust 95% CI | Bootstrap 95% CI | Randomization p |
|---|---:|---:|---:|---:|
| session_reward | 6.2726 | [1.4465, 11.0987] | [2.1792, 10.4664] | 0.1393 |
| conversion | 0.2444 | [-0.0557, 0.5446] | [0.0000, 0.4889] | 0.4810 |

Cell estimates are descriptive; policy-by-persona interactions are exploratory.

## Audit

- Randomized assignments represented: 31
- Assignments without an observed shop session: 0
- Participants with duplicate observed sessions: 0
- Logged decision-request errors: 0
- Unattributed or mismatched sessions: 27
- Action-level IPS/SNIPS/DR is intentionally not run because deterministic policy logs do not establish overlap.
