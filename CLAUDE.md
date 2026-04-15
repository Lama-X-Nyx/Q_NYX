# Q_NYX — Operating System for Claude Code

## Mission

You are working on **one integrated trading system**, not a collection of disconnected experiments.

Your job is to **understand, preserve, and improve the existing project as a coherent multi-timeframe system**.

You must optimize for:
1. **architectural coherence**
2. **end-to-end integration**
3. **TDD-first development**
4. **multi-timeframe consistency**
5. **clear documentation at every iteration**

You must **not** optimize for local elegance, isolated module quality, or parallel prototype creation.

---

## Non-negotiable rules

### 1) Integration-first, always
This repository is a **single product**.
Do not create side architectures, parallel pipelines, duplicate implementations, or isolated “better versions” unless explicitly requested.

Before writing code, you must identify:
- the current execution path
- the current owner module of the behavior
- how the new logic fits into the existing runtime path
- how the change will be verified end-to-end

### 2) TDD only
You must work **only in TDD**.

For every change:
1. identify the behavior to change
2. write or update a failing test first
3. implement the smallest patch that makes the test pass
4. run the relevant tests
5. refactor only if integration remains unchanged and tests stay green

Never implement large code changes first and “add tests later”.

### 3) Multi-timeframe is mandatory
This is a **multi-timeframe system**.
You must never treat components as isolated single-timeframe toys if they affect the real pipeline.

Every relevant change must be reasoned through the MTF stack:
- 1D context
- 4H structure
- 1H regime
- 15M setup / entry
- meta orchestration / execution / risk

Any feature, model, rule, validator, filter, or refactor that ignores the MTF nature of the system is incomplete by default.

### 4) Document every iteration
At **every iteration**, you must document what happened.

For each meaningful step:
- what problem was addressed
- what hypothesis was tested
- what files were touched
- what test was added or changed
- what passed / failed
- what remains uncertain
- what the next smallest step is

Documentation is part of the work, not an optional cleanup step.

### 5) Minimal surface area
Patch the smallest possible surface area.
Prefer modifying the natural owner of the behavior over creating a new module.

### 6) No architectural improvisation
Do not rename, move, split, merge, or replace architectural components unless explicitly asked.

Do not invent:
- `*_v2.py`
- `*_new.py`
- `*_clean.py`
- `*_refactor.py`
- `experimental_*`
- alternative pipelines
- duplicate orchestration layers

unless the user explicitly asks for that.

---

## Project worldview

This project is not a bag of utilities.
It is a **connected decision system** with:
- data ingestion
- feature generation
- multi-timeframe interpretation
- agent coordination
- filtering
- execution realism
- validation
- feedback

When you touch one part, you must think about:
- who produces its inputs
- who consumes its outputs
- how it affects the rest of the pipeline
- whether tests prove that it still works together

Local correctness without system coherence is failure.

---

## Required workflow for every task

For every non-trivial task, follow this exact order.

### Step 1 — Read before acting
Read the relevant existing files first.
Do not jump into implementation.

You must summarize:
- the current execution path
- the relevant modules already involved
- the owner module that should be changed first
- the current test coverage related to the request

### Step 2 — State the integration path
Before coding, explicitly state:
- where the change begins
- where it flows
- what modules are affected
- what remains unchanged

You must think in terms of **call path**, not isolated code fragments.

### Step 3 — Define the TDD target
Before implementation, define:
- the expected behavior
- the failing test that proves the need for change
- whether the test is unit, integration, or MTF validation

### Step 4 — Implement the smallest patch
Implement the minimal change required to satisfy the failing test.
Do not “improve nearby code” unless strictly necessary.

### Step 5 — Verify together
Run the smallest relevant test set first, then any necessary broader checks.

Verification priority:
1. targeted test
2. surrounding module tests
3. integration tests
4. MTF/path-level validation if the change affects cross-timeframe logic

### Step 6 — Document the iteration
After each iteration, document:
- what changed
- why
- which tests proved it
- what risk remains
- what next step is still needed

---

## TDD policy

TDD is mandatory.

### TDD rules
- Every bugfix must begin with a failing test reproducing the bug.
- Every new behavior must begin with a test describing the intended behavior.
- Every refactor must preserve existing tests and behavior.
- Every integration change must include a test proving compatibility with the current execution path.
- No “test later”.
- No “manual validation only”.
- No “works in theory”.

### Preferred test order
1. narrow failing test
2. minimal implementation
3. passing targeted test
4. surrounding regression tests
5. integration or MTF validation if required

### Test types
Use the right level:
- **unit tests** for local logic
- **integration tests** for module interaction
- **MTF tests** for timeframe coordination
- **regression tests** for previously broken behavior

### What is not acceptable
- implementing first and planning tests afterward
- adding only happy-path tests
- ignoring runtime path compatibility
- claiming confidence without executed tests

---

## Multi-timeframe policy

All meaningful logic must be evaluated in the context of the full MTF architecture.

### Mandatory MTF questions
When changing anything relevant, ask:
- Does this affect 1D context interpretation?
- Does this affect 1H regime classification?
- Does this affect 15M setup or entry timing?
- Does this affect cross-agent agreement?
- Does this affect orchestrator behavior?
- Does this affect execution or risk downstream?

### MTF consistency rule
No change is complete if it improves one timeframe view while silently breaking another.

### MTF-aware implementation
When designing or modifying behavior:
- preserve timeframe responsibilities
- preserve cross-timeframe data flow
- preserve alignment logic
- preserve the distinction between context, regime, setup, entry, and orchestration
- avoid collapsing MTF logic into simplistic single-layer shortcuts

### MTF-aware testing
If a change affects cross-timeframe behavior, you must include a test that verifies the interaction, not only the isolated subcomponent.

---

## Documentation policy

You must document every iteration.

### Required documentation outputs
At minimum, keep documentation updated through the work.

Use these files when they exist:
- `docs/SESSION_LOG.md` for chronological iteration logs
- `docs/CHANGELOG.md` for meaningful versioned behavior changes
- task-specific docs if already part of the repository structure

If the repository already has a documentation convention, follow it.
Do not invent a parallel documentation system unless necessary.

### Every iteration log must include
- date / iteration title
- task being addressed
- current hypothesis
- files read
- files changed
- tests added/updated
- test results
- architectural impact
- open questions
- next step

### Documentation style
Be concrete.
Do not write vague progress theater.
Document facts:
- what was proven
- what failed
- what remains unknown

---

## Source-of-truth policy

The existing repository is the source of truth.

Priority order:
1. current runtime path
2. current tests
3. current documented architecture
4. new ideas

If there is tension between elegance and compatibility, prefer compatibility unless explicitly asked otherwise.

---

## Reuse-before-create policy

Before creating any new file, class, helper, abstraction, or pipeline, you must answer:

1. Which existing module should own this behavior?
2. Why can it not live there?
3. Who will import the new thing?
4. Who will call it?
5. What existing path does it extend or replace?
6. What test proves it works with the rest of the system?

If you cannot answer these clearly, do not create the new artifact.

### New files are allowed only when
- responsibility truly does not belong to an existing module
- reuse would make the existing owner incorrect or overloaded
- the import/call path is explicit
- tests prove the new artifact works inside the current architecture

---

## Forbidden default behaviors

Unless the user explicitly asks otherwise, do **not**:

- create parallel pipelines
- create duplicate versions of existing logic
- create “clean” rewrites
- create speculative modules
- migrate architecture while solving a local bug
- split logic into many helpers just to satisfy typing
- refactor broad areas because one local change felt messy
- introduce a new abstraction without proving need
- optimize only for pyright/lint at the expense of runtime clarity
- treat documentation as optional

---

## Pyright / lint / typing policy

Typing improvements must preserve runtime behavior and architecture.

When fixing pyright:
- prefer local annotations
- prefer narrowing and guard clauses
- prefer minimal fixes
- do not restructure the project to make types easier
- do not create abstraction layers just for typing cleanliness

The goal is:
- safer code
- same architecture
- same behavior
- better verified integration

---

## Change-size policy

Prefer:
- small PR-sized patches
- reversible changes
- one behavior change at a time
- one integration proof at a time

Avoid:
- giant sweeps
- touching many files without necessity
- bundling unrelated cleanups together
- “while I’m here” changes

---

## Definition of done

A change is done only when all of the following are true:

1. the current execution path is understood
2. the correct owner module was patched first
3. a failing test existed first
4. the smallest fix was implemented
5. relevant tests pass
6. multi-timeframe coherence was considered
7. the iteration was documented
8. no unnecessary parallel module or architecture was introduced

Code that looks clean but is not integrated is **not done**.

---

## Required response style during work

When working on a task, communicate like an integration engineer.

Always provide:
- current path summary
- files involved
- TDD plan
- smallest patch plan
- test plan
- documentation update plan

When finishing an iteration, summarize:
- what changed
- what did not change
- what was proven
- what remains risky
- what the next smallest step should be

---

## Default mindset

Think like this:

- “What already owns this behavior?”
- “How does this fit into the existing MTF system?”
- “What test proves this integration works?”
- “What is the smallest patch?”
- “What do I need to document before moving on?”

Do **not** think like this:

- “I can build a cleaner version next to it”
- “I’ll isolate this in a fresh module”
- “I’ll make it elegant first and integrate later”
- “I’ll add tests once the architecture feels right”

---

## Final rule

**Do not optimize for local cleanliness.  
Optimize for architectural coherence, TDD discipline, multi-timeframe integrity, and documented iteration-by-iteration progress.**
