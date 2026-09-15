# Clickworker frozen-policy analysis

The main analysis is participant-level intention-to-treat. Quality-filtered results are a sensitivity analysis because duration, navigation, dwell and dismissals can be affected by policy.

Recruitment-window cutoff: data from 2026-08-25T00:00:00 UTC onward excluded (5 assignments, 6 sessions dropped).

## Transition-ordering endpoint

Recorded-landing N = 33
Direction of persona differences in conditional funnel-advance proportions, pooled over policy conditions, against the ordering predicted by the locked simulator reference.

| Contrast | Stage | Sim gap | Human higher | Human lower | Human gap | Newcombe 95% CI | One-sided p | Holds |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| **fastbuyer > windowshopper** | landing_to_pdp | 0.3400 | 1.0000 (n=11) | 0.8182 (n=11) | 0.1818 | [-0.1080, 0.4770] | 0.0590 | no |
| **detailedcomparator > windowshopper** | landing_to_pdp | 0.3130 | 0.8182 (n=11) | 0.8182 (n=11) | 0.0000 | [-0.3227, 0.3227] | 0.5000 | no |
| **fastbuyer > windowshopper** | pdp_to_cart | 0.5074 | 0.8182 (n=11) | 1.0000 (n=9) | NA | NA | NA | not tested |
| fastbuyer > detailedcomparator | landing_to_pdp | 0.0270 | 1.0000 (n=11) | 0.8182 (n=11) | 0.1818 | [-0.1080, 0.4770] | 0.0590 | no |
| fastbuyer > detailedcomparator | pdp_to_cart | 0.1784 | 0.8182 (n=11) | 1.0000 (n=9) | NA | NA | NA | not tested |
| detailedcomparator > windowshopper | pdp_to_cart | 0.3289 | 1.0000 (n=9) | 1.0000 (n=9) | NA | NA | NA | not tested |
| fastbuyer > detailedcomparator | cart_to_checkout | 0.1560 | 0.6667 (n=9) | 0.6667 (n=9) | NA | NA | NA | not tested |
| fastbuyer > windowshopper | cart_to_checkout | 0.4194 | 0.6667 (n=9) | 0.7778 (n=9) | NA | NA | NA | not tested |
| detailedcomparator > windowshopper | cart_to_checkout | 0.2633 | 0.6667 (n=9) | 0.7778 (n=9) | NA | NA | NA | not tested |
| fastbuyer > detailedcomparator | checkout_to_purchase | 0.1714 | 0.8333 (n=6) | 0.8333 (n=6) | NA | NA | NA | not tested |
| fastbuyer > windowshopper | checkout_to_purchase | 0.4673 | 0.8333 (n=6) | 0.7143 (n=7) | NA | NA | NA | not tested |
| detailedcomparator > windowshopper | checkout_to_purchase | 0.2959 | 0.8333 (n=6) | 0.7143 (n=7) | NA | NA | NA | not tested |

Bold contrasts are confirmatory (intersection-union, one-sided alpha = 0.05); the rest are descriptive and a null there is uninformative. The confirmatory ordering claim: NOT ANALYZABLE.

## Primary simulator fidelity

Recorded-landing N = 33
Simulator means are standardized to the recorded joint device/traffic distribution within each randomized cell.

| Outcome | Human-simulation gap | Bootstrap 90% CI | Margin | Equivalent | Mean absolute cell gap | Persona Spearman |
|---|---:|---:|---:|:---:|---:|---:|
| conversion | 0.3519 | [0.2190, 0.4801] | [-0.0500, 0.0500] | no | 0.3519 | -0.8660 |
| session_length_steps | 7.4092 | [6.0455, 8.7196] | [-0.5000, 0.5000] | no | 7.4092 | -0.5000 |
| funnel_depth | 1.3756 | [0.9737, 1.7390] | [-0.5000, 0.5000] | no | 1.3756 | 0.5000 |

The global calibration claim requires equivalence for all three endpoints: NOT ESTABLISHED.

## Intention-to-treat

N = 33

| Outcome | V3-V2 estimate | Robust 95% CI | Bootstrap 95% CI | Randomization p |
|---|---:|---:|---:|---:|
| session_reward | 2.0238 | [-3.5971, 7.6448] | [-3.0965, 7.0321] | 0.4832 |
| conversion | 0.2111 | [-0.1394, 0.5616] | [-0.1111, 0.5111] | 0.3112 |

Cell estimates are descriptive; policy-by-persona interactions are exploratory.

## Recorded-landing policy sensitivity

N = 33

| Outcome | V3-V2 estimate | Robust 95% CI | Bootstrap 95% CI | Randomization p |
|---|---:|---:|---:|---:|
| session_reward | 2.0238 | [-3.5971, 7.6448] | [-3.1802, 6.7695] | 0.4848 |
| conversion | 0.2111 | [-0.1394, 0.5616] | [-0.1111, 0.5222] | 0.3215 |

Cell estimates are descriptive; policy-by-persona interactions are exploratory.

## Quality-filtered sensitivity

N = 21

| Outcome | V3-V2 estimate | Robust 95% CI | Bootstrap 95% CI | Randomization p |
|---|---:|---:|---:|---:|
| session_reward | 5.4556 | [0.5136, 10.3975] | [1.2014, 9.5799] | 0.2015 |
| conversion | 0.2556 | [-0.0358, 0.5469] | [0.0111, 0.4889] | 0.5051 |

Cell estimates are descriptive; policy-by-persona interactions are exploratory.

## Audit

- Randomized assignments represented: 33
- Assignments without an observed shop session: 0
- Participants with duplicate observed sessions: 0
- Logged decision-request errors: 0
- Unattributed or mismatched sessions: 34
- Action-level IPS/SNIPS/DR is intentionally not run because deterministic policy logs do not establish overlap.
