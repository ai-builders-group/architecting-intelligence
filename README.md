# Architecting Intelligence: Systems Engineering for Production AI Agents and Workflows

Companion code for production AI agents and workflows, covering validation, orchestration, retrieval, and evaluation.

[The book and updates](https://arizenai.com/architecting-intelligence/?ref=gh-b2#book-signup)

[![Companion checks](https://github.com/ai-builders-group/architecting-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-builders-group/architecting-intelligence/actions/workflows/ci.yml)

## Quickstart

Use a Python 3.12 virtual environment. Then run these three commands:

```sh
git clone https://github.com/ai-builders-group/architecting-intelligence.git && cd architecting-intelligence
pip install -r requirements/quickstart.txt
python examples/quickstart.py
```

Output:

```text
Accepted: user_message='Hello' context_limit=2000
Invalid input:
1 validation error for ChatbotInput
user_message
  String should have at least 1 character [type=string_too_short, input_value='', input_type=str]
```

The quickstart is a companion example, not printed in the book; it uses [Validated chatbot input and output (lst-1.8)](lst/lst-1.8.md).

## Browse by problem

Find every listing in [LISTINGS.md](LISTINGS.md).

| Chapter | Problem | Listings |
| --- | --- | --- |
| Chapter 1 | Control agent actions through typed contracts, approval, and guardrails | [Typed result with reliability checks](lst/lst-1.3.md); [Validated chatbot input and output](lst/lst-1.8.md); [Human approval for tool calls](lst/lst-1.9.md); [Refund validation and escalation](lst/lst-1.11.md); [Bounded tool execution with correction](lst/lst-1.12.md) |
| Chapter 2 | Define agent boundaries and keep requirements aligned with code | [Agent contract interface](lst/lst-2.5.md); [Refund agent contract sketch](lst/lst-2.10.md); [Agent requirements document template](lst/lst-2.18.md); [Agent requirements compliance workflow](lst/lst-2.19.md) |
| Chapter 3 | Allocate reasoning budgets and verify candidate answers | [Reasoning budgets and trace schemas](lst/lst-3.13.md); [Reasoning engine with verifier scoring](lst/lst-3.14.md); [Stubbed reasoning strategy comparison](lst/lst-3.15.md) |
| Chapter 4 | Route requests across model tiers with costs, confidence, and fallbacks | [Model resource and routing schemas](lst/lst-4.1.md); [Confidence-gated tier routing](lst/lst-4.2.md); [Net routing benefit calculator](lst/lst-4.4.md); [Circuit breaker fallback sketch](lst/lst-4.6.md); [Semantic routing with model fallback](lst/lst-4.10.md) |
| Chapter 5 | Manage graph state, critique loops, retries, and checkpoints | [Validation routing with retry limits](lst/lst-5.3.md); [Reflection-enhanced agent state](lst/lst-5.8.md); [ReAct graph with bounded retries](lst/lst-5.10.md); [State reducers for graph updates](lst/lst-5.17.md); [PostgreSQL checkpointer with optimistic locking](lst/lst-5.19.md) |
| Chapter 6 | Keep workflows durable through retries, compensation, and human approval | [Temporal activities for external calls](lst/lst-6.4.md); [Transaction approval workflow](lst/lst-6.5.md); [Typed model activity with retries](lst/lst-6.11.md); [Saga orchestration with reverse compensation](lst/lst-6.14.md); [Human approval for saga compensation](lst/lst-6.18.md) |
| Chapter 7 | Route uncertain actions through approval, checkpoints, and review capacity limits | [Confidence based escalation policy](lst/lst-7.1.md); [Hashed proposals and approval decisions](lst/lst-7.6.md); [Interruptible human approval tool](lst/lst-7.14.md); [PostgreSQL checkpoints with resume tokens](lst/lst-7.20.md); [Adaptive supervision circuit breaker](lst/lst-7.26.md) |
| Chapter 8 | — | No code listings in this chapter |
| Chapter 9 | Extract documents and fit retrieval and conversation memory into context budgets | [Spatial table extraction schemas](lst/lst-9.2.md); [Adaptive semantic text splitting](lst/lst-9.7.md); [Confidence gated metadata filter extraction](lst/lst-9.17.md); [Event driven document embedding processor](lst/lst-9.22.md); [Cost based context strategy selection](lst/lst-9.40.md); [Conversation summaries with fact verification](lst/lst-9.45.md) |
| Chapter 10 | Retrieve relevant context with vector, text, and graph queries | [Parent and child retrieval schemas](lst/lst-10.1.md); [Retrieval relevance grader stub](lst/lst-10.4.md); [Hybrid retrieval configuration and result schemas](lst/lst-10.14.md); [Hybrid retrieval with reciprocal rank fusion](lst/lst-10.15.md); [Typed graph triple writes](lst/lst-10.21.md); [Leiden graph community detection](lst/lst-10.22.md) |
| Chapter 11 | Detect rule drift and retrieve recent or archived event signals | [Ontology drift and guarded rule updates](lst/lst-11.9.md); [Event signals and summary batches](lst/lst-11.12.md); [In-memory event vector buffer](lst/lst-11.13.md); [Event batch summarization](lst/lst-11.14.md); [Recent and historical event retrieval](lst/lst-11.15.md) |
| Chapter 12 | Bind evaluation evidence to comparable subjects and release decisions | [Evaluation identity and comparison receipts](lst/lst-12.5.md); [Evidence-bound evaluator results](lst/lst-12.6.md); [Typed metric evaluation protocol](lst/lst-12.7.md); [Evidence-bound release authorization](lst/lst-12.9.md); [Evaluation identity regression tests](lst/lst-12.10.md) |
| Chapter 13 | Authorize effects and check synthetic attacks for unauthorized changes | [Effect authorization with single-use capabilities](lst/lst-13.1.md); [Synthetic effect discovery contracts](lst/lst-13.3.md); [Canary leakage and task utility gate](lst/lst-13.4.md) |
| Chapter 14 | — | No code listings in this chapter |
| Chapter 15 | Choose retrieval or fine-tuning under privacy, update, and cost constraints | [Knowledge substrate cost router](lst/lst-15.1.md); [Fine-tuning breakeven cost model](lst/lst-15.4.md); [Fine-tuning architecture decision record](lst/lst-15.5.md); [Retrieval versus fine-tuning decision tree](lst/lst-15.6.md); [SQL dialect fine-tuning decision example](lst/lst-15.7.md) |
| Chapter 16 | Prepare, validate, activate, and monitor model adapters | [Shared-support KL divergence monitor](lst/lst-16.2.md); [Synthetic training data rejection sampling](lst/lst-16.3.md); [Adapter composition compiler pseudocode](lst/lst-16.7.md); [Adapter admission and activation receipts](lst/lst-16.8.md); [Model adapter release pipeline](lst/lst-16.9.md); [Model card from pipeline artifacts](lst/lst-16.11.md) |
| Chapter 17 | Validate metric requests and preserve source-specific meanings | [Typed MCP metrics tool](lst/lst-17.9.md) |
| Chapter 18 | Declare an agent interface, authentication, and metrics summary capability | [Metrics agent capability card](lst/lst-18.2.md) |
| Chapter 19 | — | No code listings in this chapter |
| Chapter 20 | — | No code listings in this chapter |

## Running a listing

Each listing page says what it needs and how to run it. Most files define code to import rather than printing output. Install the requirements file the page names. `pip install -r requirements.txt` installs the common packages most listings use, so it is the usual full install. Listings marked as needing an earlier listing build on that one's definitions.

## What is checked

CI checks that every pointer resolves and every Python file compiles on Python 3.12; the code is not tested against live services. Details: [MAINTAINING.md](docs/MAINTAINING.md).

## Errata

Report a difference from the book or a failing listing with the [errata form](https://github.com/ai-builders-group/architecting-intelligence/issues/new?template=errata.yml).

## Licenses

Code is released under the MIT license (`LICENSE`). Diagrams and other non-code content are released under CC BY 4.0 (`LICENSE-CONTENT`). Copyright 2026 AI Builders Group.
