# Clinical Design

> How this app decides what to say to someone who is struggling, which
> psychological constructs each decision corresponds to, and where the design
> deliberately stops.
>
> Companion to [`PRIVACY.md`](../PRIVACY.md). That document defines the data
> trust boundary; this one defines the clinical boundary.

---

## 1. Scope, and what this is not

This is a journaling assistant. It reflects a person's own writing back to them
with structure and warmth. That is the entire claim.

**It is not:**

| Not this | Why the line matters |
|---|---|
| A diagnostic tool | Diagnosis requires a clinician, a history, and a differential. A language model pattern matching on one paragraph has none of those. The app is instructed never to label the person (`generator_prompts.py:18`, `verifier_prompts.py:27`). |
| A therapy substitute | There is no working alliance, no continuity of care, no accountability, and no one to notice if things get worse. |
| A crisis service | The crisis path routes **outward** to human help. It does not attempt to manage risk itself (`app.py:218`). |
| A validated instrument | No efficacy study, no psychometric validation, no clinical trial. Nothing here has been tested on outcomes. |

Every claim below is a description of **implemented behavior**, verifiable at the
cited line. Where a construct is named, it names the established idea the
behavior corresponds to. It is not a claim of formal clinical implementation.

---

## 2. The three tier response model

The central design decision: **not all difficulty is the same difficulty**, and
answering all of it identically causes harm in both directions.

```
                        ┌─────────────────────────────────────┐
  journal entry ───────►│  tier resolution (deterministic)    │
                        └──────────────┬──────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
   ┌─────────┐                  ┌─────────────┐                ┌──────────┐
   │ CRISIS  │                  │  DISTRESS   │                │  NORMAL  │
   └────┬────┘                  └──────┬──────┘                └────┬─────┘
        │                              │                            │
  reframe CLEARED             reframe KEPT                    reframe kept
  support message,            steadying message              analysis only
  points outward              above the analysis
        │                              │                            │
   never softened              never medicalised            never inflated
```

Implemented in `_apply_reframe_gate` (`app.py:501`). The tiers are mutually
exclusive and ordered by severity: crisis wins over distress, always.

### Why three tiers and not two

Collapsing distress into crisis is the failure mode this design exists to
prevent. From the verifier instructions (`verifier_prompts.py:20`):

> Those are painful self judgment and the app answers them on a separate
> supportive path; marking them as crisis pushes emergency resources at someone
> who is simply having a hard time, which is its own harm.

That is **over triage**, and treating it as a harm rather than as safe
conservatism is the load bearing idea in this document. A person who writes "I
feel like a failure" every week and receives a crisis hotline number every week
learns that the tool does not read them, and stops writing. The safety feature
destroys the thing it was protecting.

Collapsing crisis into distress is the opposite failure and is worse. Hence the
asymmetry in section 4.

---

## 3. Construct map

Each row: a behavior that exists in the code, and the established construct it
corresponds to.

| Implemented behavior | Location | Corresponding construct |
|---|---|---|
| Coaching to notice absolutist words such as "always" and "never" | `generator_prompts.py:29` | Dichotomous or all or nothing thinking, one of the cognitive distortions catalogued in Beck's cognitive therapy tradition |
| `reframe`: one grounded alternative perspective, explicitly forbidden from dismissing or minimising the feeling | `generator_prompts.py:30`, `schemas/analysis.py:45` | Cognitive restructuring. The non minimising constraint is what separates restructuring from thought stopping or forced positivity |
| "Offer, never order." Suggestions must read as optional invitations, never commands | `generator_prompts.py:19`, `verifier_prompts.py:28` | Autonomy support, and avoidance of the righting reflex, as described in motivational interviewing (Miller and Rollnick) |
| Crisis / distress / normal routing with different response registers | `app.py:501` | Stepped care: match response intensity to presentation severity |
| Refusal to answer self criticism with emergency resources | `verifier_prompts.py:20` | Avoidance of iatrogenic harm through over triage |
| `_HARSH_OUTPUT_PATTERNS` blocks blame, dismissal, and character judgment in assistant output | `app.py:190`, `app.py:541` | Invalidation, and therapeutic rupture as a response failure mode |
| `_REPORTED_SPEECH` guard: "my friend said I am worthless" does not count as self judgment | `app.py:175` | Attribution. Separates a self schema from a reported appraisal by another person |
| "Coach the writing process, not the person" | `generator_prompts.py:16` | The expressive writing paradigm (Pennebaker), where benefit comes from the act and structure of writing rather than from being evaluated |
| Hedged language required under uncertainty; confidence score emitted | `generator_prompts.py:11`, `generator_prompts.py:15` | Epistemic humility as a clinical stance. Overconfident interpretation of a single entry is itself a failure |
| Explicit prohibition on diagnosis, with referral suggested instead | `generator_prompts.py:14` | Scope of practice |
| Retrieved past entries may never drive crisis classification | `verifier_prompts.py:19` | Present state assessment. Risk is a current state, not a permanent label |

That last row deserves emphasis. It is the point where the retrieval
architecture and the clinical model constrain each other:

> Someone who had a terrible night last week and today writes "had a good day"
> is not in crisis, and treating them as though they are means every entry they
> ever write is answered with emergency resources.
> `verifier_prompts.py:19`

A naive RAG design would let retrieved context inform every judgment. Here,
retrieval is deliberately **excluded** from one specific decision, because
importing a past crisis into a present assessment would permanently mark the
user. Grounding and risk assessment draw on different evidence sets by design.

---

## 4. Asymmetric risk, stated explicitly

The crisis gate is deliberately not balanced, and the reasoning is recorded at
`app.py:80`:

- **False positive** (fires on a non crisis entry): the user sees one supportive
  message they did not need. Mildly imprecise, recoverable.
- **False negative** (misses a real crisis): positivity or a reframe reaches
  someone in genuine danger. Not recoverable.

So the crisis floor is tuned toward catching more real phrasing at the cost of
occasional unnecessary support messages. One accepted tradeoff is documented
rather than hidden: "jumping off a bridge" also matches benign bungee jumping
entries, and was kept anyway.

The distress tier runs the same logic at lower stakes, and can therefore afford
to be broader: firing only adds one steadying sentence and never suppresses the
analysis (`app.py:141`).

The harsh output filter runs the asymmetry in the **opposite** direction, and
this is intentional (`app.py:186`). A false positive there silently deletes a
useful suggestion, so over broad patterns would quietly gut the analysis. It
matches blame, dismissal, and diagnosis only, never ordinary directive advice.

**Design rule that follows:** each filter's breadth is set by the cost of its own
false positive, not by a single global sensitivity preference.

---

## 5. Deterministic floor beneath model judgment

Every safety decision has a code level floor underneath the model's judgment.
The model classifies; code acts.

| Check | Floor | Why not trust the model |
|---|---|---|
| Crisis detection | `_CRISIS_PATTERNS` regex (`app.py:99`) plus `_HARM_TO_OTHERS_PATTERNS` (`app.py:132`) | Fires even when the verifier call failed entirely or returned a wrong verdict. `_is_crisis` (`app.py:458`) returns true if **any** signal fires |
| Quote fabrication | `_strip_ungrounded_quotes` (`app.py:581`) drops any quote not an exact substring of the entry | Live confirmed: a fabricated quote with zero grounding passed verification with `groundedness_score=0.95` |
| Assistant tone | `_find_harsh_content` (`app.py:541`) forces a rewrite, `_strip_harsh_items` (`app.py:556`) drops what survived | Tone is exactly what a small local model misses quietly |

The principle: **whether a property is mechanically checkable determines whether
it belongs in code or in the model.** "Is this quote actually in the entry" needs
no judgment at all, so it must not depend on any. "Is this reframe compassionate"
genuinely needs judgment, so it stays with the model and gets a code level
backstop.

---

## 6. Known gaps, stated rather than implied

These are real limitations, encoded as `XFAIL` tests so they cannot be quietly
forgotten. Each is a deliberate stop, not an oversight.

| Gap | Where recorded | Reasoning |
|---|---|---|
| Vague departure phrasing ("I won't be here much longer") is not caught by the regex floor | `tests/test_crisis_gate_adversarial.py` | Too easily benign (moving, changing jobs, retiring). Regexing it would raise false positives sharply. Left to verifier judgment |
| Non English self harm phrasing is not caught by the floor | `tests/test_crisis_gate_adversarial.py` | The floor is English only. The verifier prompt carries the mitigation |
| Street addresses and full names are not redacted | `tests/test_privacy_adversarial.py`, `privacy/redact.py` | Needs semantic understanding a regex cannot provide. A regex that gives false confidence is worse than one honest about its coverage |
| `RAG_NAMESPACE_MODE=user` trusts an unauthenticated header | `tests/test_privacy_adversarial.py` | Not exploitable at the shipped default (`session`). Must not be enabled without real authentication |

Two crisis patterns were tried and **removed** after live false positive testing
(`app.py:90`): bare "kms" collides with the AWS KMS acronym, which is a real
collision for a work stress journaling app, and bare "overdose" collides with the
ordinary idiom "an overdose of X".

---

## 7. Escalation

When the crisis tier fires, the app clears the reframe and shows a message that
points outward to human support (`app.py:218`). It does not attempt to assess
lethality, make a safety plan, or keep the person in conversation.

The message is operator editable via `AIHJ_CRISIS_MESSAGE` so the resource line
can be localised. If that variable is set to an empty string, the built in
message is used rather than rendering a crisis entry with no support text at all
(`app.py:214`).

**Anyone deploying this beyond personal use should replace the default message
with region appropriate crisis resources.**

---

## 8. What is not yet measured

Honest accounting of the gap between what is built and what is evidenced:

**Now measured** (`make crisis-eval`, `evals/crisis_safety_eval.py`, 56 labeled
cases, deterministic floor only, no LLM involved):

| tier | sensitivity | specificity | PPV | NPV |
|---|---|---|---|---|
| crisis | **1.000** | 0.971 | 0.957 | 1.000 |
| distress | 1.000 | 1.000 | — | — |

Confusion on the crisis tier: TP=22, **FN=0**, FP=1, TN=33. The single false
positive is the documented bungee-jumping case from section 4, kept on purpose.
Crisis-handled-as-distress misroutes: 0.

Read these numbers with two caveats:

- **The case set was authored with the patterns in `app.py` visible**, so
  sensitivity is optimistic. Specificity is the more trustworthy half, since the
  neutral cases are ordinary journaling idioms rather than reverse-engineered
  negatives. An unbiased estimate needs a held-out set.
- **The three documented gaps from section 6 are excluded from the scores** and
  reported separately. All three are still missed, as designed. Folding them in
  would be dishonest in the other direction, since they are deliberate
  non-coverage rather than regressions.

The eval was mutation-tested rather than trusted on a green: deleting one
euphemistic branch from `_CRISIS_PATTERNS` drops sensitivity to 0.909, names both
missed entries, and exits non-zero. A safety metric that cannot go red is not a
safety metric.

**Now measured** (`make reframe-eval`, `evals/reframe_quality.py`, 15 labeled
exemplars validating a five-detector rubric): minimising, toxic positivity,
commanding language, ungrounded genericness, and invalidating pivots. Each of
the five detectors was proven load-bearing by mutation testing (disabling any
one drops recall from 1.000 to 0.800, missing exactly its own two cases and no
others). Scored against real `qwen3:8b` output with `--live`: 5/5 acceptable,
after fixing one rubric gap the live run exposed (the acknowledgement lexicon
missed "it's natural to feel X", a phrasing this model uses as house style).
Full account, including a mutation test that was itself broken and caught before
being trusted, in `docs/IMPROVEMENTS.md` section 12.

This measures the mechanically checkable half only: whether a reframe minimises,
commands, or dismisses. It cannot judge insight or whether a reframe is *true*.

**Still not measured:**

- Verifier-assisted crisis detection. These numbers are the deterministic floor
  with an empty verdict, which is the guarantee that holds when the model fails.
  The verifier adds coverage on top and is not scored here.
- No outcome data of any kind. Nothing here demonstrates that using this app
  helps anyone.

See [`IMPROVEMENTS.md`](IMPROVEMENTS.md) for the work closing these gaps.

---

## 9. Disclaimer

This tool is for personal reflection. It is not medical advice, not therapy, and
not a crisis service. If you are struggling, please contact a licensed
professional or a crisis line in your area. If you are in immediate danger,
contact local emergency services.
