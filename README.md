# Negotiation prototype

This research prototype studies a structured negotiation among Alice, Bob, and Carl over a hazardous-waste handling scenario. I began with an explicit, LLM-free deterministic baseline to check the negotiation mechanics, then added a Bayesian particle model that estimates opponents' issue weights instead of assuming they are known.

The scenario is still synthetic: utility weights, option values, reservation values, and system constraints are configured manually. That keeps the protocol and inference model testable while preference elicitation and empirical evaluation remain future work. Runs end in either an agreement accepted by all three agents or a deadlock, with decisions and belief updates recorded as JSON traces.

## What is implemented

- A configurable six-issue offer space covering handler, timing, cost responsibility, employment protection, work/pay protection, and disposal documentation.
- Declarative system constraints that apply to every offer and filter infeasible offers before they can be proposed. A simple knowledge graph stores scenario entities and relationships.
- Additive stakeholder utilities with configured issue weights, option values, and reservation values.
- Exhaustive search over feasible offers. The current offer score is the proposer's surplus multiplied by the other agents' predicted acceptance probabilities. Ties follow the configured issue and option order.
- A rotating Alice -> Bob -> Carl protocol with propose, counter, accept, reject, agreement, and deadlock outcomes. A new proposal counts as the proposer's acceptance; the other agents must accept that same offer.
- Two opponent-belief settings: known preferences for a deterministic baseline, and separate Dirichlet-particle models for each observer-opponent pair. The particle models infer issue weights from explicit accept/reject events and expose posterior predictions, mean, variance, and effective sample size.
- Structured traces containing offers considered, scores, actions, reasons, observations, and posterior updates.

`InferenceCore` and `OfferSearch` are interfaces between the controller and the inference/search implementations. The controller does not depend on a particular opponent model.

## Run

Requires Python 3.11 or newer. Install the package from the repository root:

```powershell
python -m pip install -e .
```

Run the deterministic baseline or the particle-based model:

```powershell
python -m negotiation configs/toxic_waste.json > deterministic_trace.json
python -m negotiation.run_bayesian configs/toxic_waste.json configs/bayesian_particles.json > bayesian_trace.json
```

Each command writes a JSON outcome and turn traces to standard output. Example outputs are in [`examples/`](examples/).

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

The tests cover offer validation, constraint filtering, utility calculations, exhaustive search, protocol outcomes, particle likelihoods and normalisation, reproducibility, resampling, public event routing, and a multi-round synthetic negotiation in which a rejection changes a later counteroffer.

## Interpretation and current limits

The scenario rules and stakeholder preferences are illustrative configuration values. They are not legal findings or preferences measured from Alice, Bob, or Carl. The deterministic baseline uses known configured utilities to check exact behavior, including structural deadlock. Bayesian runs do not use that perfect-information structural check.

The particle model infers issue weights only. Option values, reservation values, and the logistic response parameter remain fixed. Counteroffers appear in the public trace but do not yet contribute a pairwise-preference likelihood. Inference cores receive a public scenario view and their own utility. Bayesian opponent beliefs receive fixed option values and reservations, but not the synthetic true opponent issue weights. The controller retains full profiles for outcome diagnostics; the public view is an API boundary, not process-level security isolation.

Stakeholder-specific hard constraints are not modeled. The configured system constraints apply equally to all stakeholders; personal non-negotiables would require a separate design for feasibility and acceptance.

This repository does not include a human review loop, language model, UI, evolutionary search, Coppélia, quantum model, or real-user evaluation.
