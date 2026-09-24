# Agentic negotiation research prototype

This is the **first coding assignment** from `SPEC_TOTAL_HANDOFF.md`: a structured, LLM-free standard negotiation baseline plus a continuous Dirichlet-particle opponent model. It uses synthetic configuration only. No empirical claim, legal conclusion, or Coppélia implementation is implied.

## Run

Requires Python 3.11+ and NumPy. From this directory:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python -m negotiation configs/toxic_waste.json > deterministic_trace.json
python -m negotiation.run_bayesian configs/toxic_waste.json configs/bayesian_particles.json > bayesian_trace.json
```

Both commands emit a JSON outcome and complete turn traces. The Bayesian command explicitly disables the perfect-information structural-deadlock oracle and records that fact in its output.

## Structure

- `configs/toxic_waste.json`: issue schema, synthetic profiles, hard rules, graph, turn order, objective, and turn limit.
- `configs/bayesian_particles.json`: separate prior, seed, response-noise, ESS, and resampling configuration for each observer–opponent pair.
- `src/negotiation/domain.py`: typed offers, utilities, rules, graph, and scenario loading.
- `src/negotiation/inference.py`: common inference/search interfaces, deterministic core, and exhaustive search.
- `src/negotiation/protocol.py`: controller, agreement/deadlock results, and inspectable decisions.
- `src/negotiation/bayesian_particles.py`: vectorised particle posterior and posterior predictions.
- `tests/fixtures`: hand-checked deterministic case and hidden synthetic Bayesian case.

## Explicit development assumptions

- All six issue option sets, both hard rules, stakeholder weights and values, reservations, `beta = 12`, `ESS threshold = 0.4`, turn limit, and seeds are **synthetic development parameters**, not measured preferences. The rule requiring a specialist is a fixture constraint, not a legal finding.
- The configured action objective is proposer surplus multiplied by other parties' predicted acceptance probabilities. It is not a fairness criterion. Exhaustive-search ties use configured issue and option order. The controller treats a proposal/counteroffer as the proposer's own acceptance, then requires each other agent to accept that exact offer.
- Under perfect information, a preflight scan can certify structural deadlock. Bayesian runs cannot use true hidden opponent weights for that decision; they report strategic or procedural deadlock under the configured protocol. Whether and how to infer structural deadlock under uncertainty remains open.
- Bayesian v1 infers **only issue weights**. Option-level values, reservations, and logistic response noise are fixed inputs. Only explicit `ACCEPT`/`REJECT` events update the posterior. `PROPOSE`/`COUNTER` are public trace events but do not yet use a pairwise preference likelihood.
- The in-process synthetic scenario retains true utilities for outcome diagnostics. Bayesian belief objects are constructed without opponent weight vectors. Strict isolation of private profiles across agents remains a later multi-agent integration task.

## Research decisions left open

Preference elicitation, needs versus wants, Bob's dual role, fairness/social objectives, strategic behaviour, richer counteroffer evidence, and the eventual Coppélia/Q-Coppélia relationship are **not** answered by this prototype. `InferenceCore` and `OfferSearch` are the extension points. Human review, language, UI, evolutionary search, Coppélia, quantum logic, and real-user testing are outside this assignment.
