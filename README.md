# LLM Code Graph Compiler

A constraint-guided program synthesis system that uses large language models to generate typed execution graphs over a fixed node library, then deterministically compiles them into executable Python artifacts.

Instead of generating free-form code, the system restricts synthesis to a validated DAG composed of template primitives. This improves structural reliability and first-pass execution success.

---

## Overview

Traditional LLM code generation produces unconstrained source code, which often fails due to schema drift, structural inconsistencies, or minor semantic deviations.

This system:

1. Converts a natural language task into a structured graph plan  
2. Validates graph structure and type compatibility  
3. Topologically schedules nodes  
4. Deterministically compiles template implementations  
5. Produces executable Python artifacts  

---

## System Architecture

Task Description  
↓  
Planner (LLM-based structured graph generator)  
↓  
Graph Validator (DAG + type + parameter checks)  
↓  
Topological Scheduler  
↓  
Deterministic Compiler  
↓  
Executable Artifact  

---

## Repository Structure

```
core/
    compiler.py
    planner.py
    validator.py
    reliability.py

nodes/
    registry.py
    types.py
    templates/

benchmark/
    tasks.json
    harness.py
    run_baseline.py

cli.py
requirements.txt
```

---

## Node Registry

All computation is performed using a fixed library of nodes.

Each node defines:

- Input type
- Output type
- Required parameters
- Path to implementation template

Nodes are declared in:

- `nodes/registry.py`
- `nodes/types.py`
- `nodes/templates/`

The planner is strictly restricted to this registry.

---

## Planner Output Format

Planner output must be strictly valid JSON:

```json
{
  "nodes": [],
  "edges": [],
  "parameters": {},
  "flags": [],
  "glue_code": ""
}
```

Constraints:

- Only registered nodes may be used
- Graph must be acyclic
- Edges must be type-compatible
- Required parameters must be present

---

## Setup

### 1. Clone Repository

```bash
git clone <repo-url>
cd llm-code-graph-compiler
```

### 2. Create Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

---

## CLI Usage

### Manual Node Composition

```bash
python cli.py --nodes CSVParser QueryEngine
```

### From Plan JSON

```bash
python cli.py --plan tests/test_plan.json
```

### From Natural Language Task (LLM Planner)

```bash
python cli.py --task "Read CSV, store into SQLite, and query data"
```

Generated artifacts are written to:

```
output/app.py
output/requirements.txt
```

---

## Benchmarking

Benchmark tasks are defined in:

```
benchmark/tasks.json
```

Each task includes:

- Natural language description
- Expected output artifacts
- Evaluation criteria

---

## Running Full Benchmark (Compiler + Baseline)

```bash
python benchmark/harness.py --tasks benchmark/tasks.json --output benchmark/results.json
```

This will:

- Generate structured plans
- Validate graphs
- Compile executable artifacts
- Execute pipelines
- Evaluate criteria
- Run baseline LLM generation
- Store results in `benchmark/results.json`

---

## Running Baseline Only

After generating compiler results:

```bash
python benchmark/run_baseline.py --tasks benchmark/tasks.json --results benchmark/results.json
```

This augments the existing results file with baseline metrics.

---

## Reproducing Benchmark Results

To reproduce reported metrics:

1. Delete any runtime artifacts:
   - `benchmark/results.json`
   - Generated CSV files
   - Generated JSON files
   - SQLite `.db` files  

2. Ensure `.env` is configured  

3. Run:

```bash
python benchmark/harness.py --tasks benchmark/tasks.json --output benchmark/results.json
```

4. (Optional) Augment baseline:

```bash
python benchmark/run_baseline.py --tasks benchmark/tasks.json --results benchmark/results.json
```

Primary reported metric:

```
first_pass_success_rate
```

---

## Metrics Collected

For each task:

- Plan success
- Validation success
- Compilation success
- Execution success
- Criteria pass/fail
- First-pass success
- Runtime duration
- Baseline success
- Baseline duration

---

## Reliability Module

A structural reliability estimator exists at:

```
core/reliability.py
```

This module computes graph connectivity and type coherence metrics.  
It is currently not integrated into the main execution loop.

---

## Requirements

Core dependencies:
pydantic
pandas
openai
sqlalchemy
psycopg2-binary

---

## Current Scope

The system demonstrates:

- Structured DAG-constrained planning
- Deterministic compilation
- Type-safe execution
- Measurable improvement over free-form LLM code generation
- Reproducible benchmarking
