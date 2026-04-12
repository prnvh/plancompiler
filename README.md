# PlanCompiler

PlanCompiler is a typed graph compiler for data workflows. An LLM plans over a fixed node library, the system validates the plan deterministically, and the compiler emits executable Python from pre-written node templates.

This repository was previously released as **LLM Code Graph Compiler v1.0.0**. The current repository name and system name are **PlanCompiler**.

## Paper

- [Paper PDF](./paper/PlanCompiler.pdf)
- [Technical Writeup](https://prnvh.github.io/compiler.html)

## Current Status

The current system is not the original single-call planner described in older drafts. The codebase now includes:

- a **two-stage planner** in [core/planner.py](./core/planner.py)
- **canonical node contracts** in [nodes/contracts.py](./nodes/contracts.py)
- **strict deterministic validation** in [core/validator.py](./core/validator.py)
- **deterministic compilation** with in-memory `result` output support in [core/compiler.py](./core/compiler.py)
- a built **DS-1000 evaluation path** under [benchmark/ds1000](./benchmark/ds1000) with loader, frozen subset manifest, comparator, and multi-case runner

The DS-1000 path is already built and working end to end. We are intentionally not publishing the current score in the README yet.

## Why

Free-form LLM code generation often fails structurally before it fails syntactically. Long multi-step workflows drift:

- later steps reference the wrong columns
- intermediate formats stop matching earlier assumptions
- the model uses stale or unavailable APIs
- runs diverge even when the prompt and temperature stay fixed

PlanCompiler reduces that failure surface by asking the LLM to plan with system primitives instead of writing the whole program directly.

## How It Works

PlanCompiler runs a fixed pipeline:

1. **Stage 1 planning:** choose the graph architecture
2. **Stage 2 planning:** fill parameters for the chosen nodes only
3. **Normalization:** clean up representation differences only
4. **Validation:** reject invalid plans deterministically
5. **Compilation:** emit Python from verified node templates
6. **Execution:** run the compiled artifact

Important constraint: normalization is **not** structural auto-repair. It does things like:

- assign missing node ids
- normalize edge syntax
- resolve legacy node-type references into node ids
- canonicalize parameter aliases through node contracts

It does **not** invent nodes, repair broken graphs, or rewrite invalid plans into valid ones.

## Architecture

### 1. Node Registry

The registry in [nodes/registry.py](./nodes/registry.py) is the ground truth for available primitives. Each node declares:

- stable internal id
- human-facing label and description
- input and output types
- required parameters
- template path
- function name
- planner visibility
- canonical parameter schema and examples

The planner can only choose nodes that exist in the registry.

### 2. Node Contracts

[nodes/contracts.py](./nodes/contracts.py) defines the canonical parameter shapes for nodes, especially transformer nodes with operation lists.

This is where PlanCompiler standardizes things like:

- top-level parameter names
- allowed operation types
- required and optional fields per operation
- alias normalization for planner output

The goal is to make execution strict without making the planner guess from scratch every time.

### 3. Two-Stage Planner

The planner in [core/planner.py](./core/planner.py) is the only place where the LLM is called.

**Stage 1: architecture planning**

The first planner call returns only:

```json
{
  "nodes": [
    {"id": "n1", "type": "CSVParser"},
    {"id": "n2", "type": "DataFilter"},
    {"id": "n3", "type": "CSVExporter"}
  ],
  "edges": [["n1", "n2"], ["n2", "n3"]],
  "flags": []
}
```

**Stage 2: parameter planning**

The second planner call receives the fixed architecture plus the exact contracts for the selected nodes and returns only:

```json
{
  "parameters": {
    "n1": {"file_path": "data.csv"},
    "n2": {"condition": "salary > 35000"},
    "n3": {"output_path": "filtered.csv"}
  }
}
```

Then the system merges both stages and normalizes the final plan.

The same planner entrypoint is used by:

- the normal CLI / harness flow
- the benchmark harness
- the DS-1000 runner

There is not a separate DS-1000-specific planner.

### 4. Validator

The validator in [core/validator.py](./core/validator.py) is deterministic. Invalid plans stop here and do not compile.

Current checks include:

- node existence
- edge validity
- type compatibility
- acyclicity
- orphan detection
- input arity
- required parameter presence
- parameter contract validity
- single terminal sink

### 5. Compiler

The compiler in [core/compiler.py](./core/compiler.py) performs:

- topological ordering
- node template emission
- deterministic execution block generation

The planner no longer supplies `glue_code` as part of the normal planning contract. Plans are compiled from nodes, edges, and parameters.

The compiler supports two main output modes:

- `emit_mode="print"` for script-style workflows
- `emit_mode="result"` for in-memory workflows such as DS-1000 evaluation

### 6. Runtime Adapters

The repository has multiple runners built on the same core:

- [benchmark/harness.py](./benchmark/harness.py) for the custom benchmark sets
- [benchmark/ds1000/runner.py](./benchmark/ds1000/runner.py) for DS-1000 evaluation

The benchmark code is responsible for loading tasks and scoring outputs. The planning, validation, and compilation logic stays in `core/`.

## Type System

Node edges are typed. A plan is valid only when adjacent node types are compatible.

Core types currently used in the system include:

- `FilePath`
- `DataFrame`
- `Series`
- `Scalar`
- `DBHandle`
- `HTTPResponse`
- `ANY`

`ANY` is reserved for true pass-through or flexible boundary nodes such as logging or output reduction. It is not a license for arbitrary unsafe plans.

## Node Library

The current planner-visible library is broader than the original v1 set and now includes more explicit semantic transformer families.

### Sources / Ingestion

- `CSVParser`
- `JSONParser`
- `ExcelParser`
- `DataFrameInput`
- `SQLiteReader`

### DataFrame Processing

- `SchemaValidator`
- `ColumnTransformer`
- `ValueTransformer`
- `DatetimeTransformer`
- `DataFilter`
- `ColumnSelector`
- `NullHandler`
- `DataSorter`
- `TypeCaster`
- `DataDeduplicator`
- `Aggregator`
- `StatsSummary`
- `ReduceOutput`
- `DataTransformer` as a generic fallback

### Storage / Query

- `SQLiteConnector`
- `PostgresConnector`
- `QueryEngine`

### Export / API / Adapters

- `CSVExporter`
- `JSONExporter`
- `RESTEndpoint`
- `AuthMiddleware`
- `ErrorHandler`
- `Logger`
- `HTTPToDataFrame`

`DataFrameJoin` exists in the codebase but is currently hidden from planner use because the system does not yet honestly support general multi-input planning.

## Setup

**Requirements:** Python 3.11

```bash
git clone https://github.com/prnvh/plancompiler
cd plancompiler
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Set environment variables:

```bash
OPENAI_API_KEY=your_key_here
```

Optional planner configuration:

```bash
PLANNER_MODEL=gpt-4.1
```

The current default planner model is `gpt-4.1`. The planner model is configurable through `PLANNER_MODEL`, and the pricing table used by the harness lives in [core/planner.py](./core/planner.py).

If you want to reproduce baseline comparisons, you will also need:

```bash
ANTHROPIC_API_KEY=your_key_here
```

## Usage

### Natural-Language Planning

```bash
python cli.py --task "Read a CSV, keep rows where salary > 35000, and export the result"
```

### Manual Node Selection

```bash
python cli.py --nodes CSVParser DataFilter CSVExporter
```

### Load a Prewritten Plan

```bash
python cli.py --plan plans/my_plan.json
```

Each run goes through:

1. plan resolution
2. validation
3. compilation
4. `output/app.py` emission
5. execution or inspection depending on the caller

## Benchmarks

### Custom Benchmark Harness

The original task-suite harness is still available:

```bash
python benchmark/harness.py --tasks benchmark/tasks/tasks_set_a.json --output benchmark/results/results_set_a.json
```

Baseline runner:

```bash
python benchmark/run_baseline.py --tasks benchmark/tasks/tasks_set_a.json --results benchmark/results/results_set_a.json
```

### DS-1000

The repository now includes a working DS-1000 pipeline under [benchmark/ds1000](./benchmark/ds1000):

- upstream dataset snapshot in [benchmark/ds1000/upstream](./benchmark/ds1000/upstream)
- loader in [benchmark/ds1000/loader.py](./benchmark/ds1000/loader.py)
- frozen linear Pandas subset manifest in [benchmark/ds1000/linear_pandas_subset_manifest.json](./benchmark/ds1000/linear_pandas_subset_manifest.json)
- multi-case runner in [benchmark/ds1000/runner.py](./benchmark/ds1000/runner.py)
- comparator in [benchmark/ds1000/comparator.py](./benchmark/ds1000/comparator.py)

Freeze or refresh the subset manifest:

```bash
python -m benchmark.ds1000.freeze_subset --skip-failures --output benchmark/ds1000/linear_pandas_subset_manifest.json
```

Run a smoke sample:

```bash
python -m benchmark.ds1000.run_sample --limit 15 --skip-failures --output benchmark/results/ds1000_sample_results.json
```

Run the full frozen subset:

```bash
python -m benchmark.ds1000.run_sample --limit 158 --skip-failures --output benchmark/results/ds1000_full_results.json
```

The DS-1000 runner executes all declared test cases for each retained task. The README intentionally does not publish the current score yet.

## Limitations

PlanCompiler is stronger than the original v1 system, but it is not feature-complete.

- **No true fan-in / branching execution yet.** The current compiler and validator assume single-predecessor execution for planner-visible graphs.
- **`DataFrameJoin` is not planner-enabled.** Multi-input dataframe workflows are not honestly supported yet.
- **`Aggregator` is still narrower than full pandas groupby/agg semantics.** This is one of the major remaining capability gaps.
- **Fallback planning still exists.** `DataTransformer` remains available as a generic fallback when no more specific transformer fits, and the planner can still under-use more specific nodes.
- **Planner quality is still a real bottleneck.** A plan can validate and execute while still producing the wrong answer.
- **The DS-1000 integration currently targets a frozen linear Pandas subset, not all DS-1000 task shapes.**

## Development

Run the test suite:

```bash
python -m unittest discover -s tests
```

The tests cover:

- plan normalization and validation
- node contracts
- compiler behavior
- semantic transformer nodes
- benchmark adapters including DS-1000 support

---

*April 2026 · Pranav H.*
