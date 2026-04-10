# Changelog

## Unreleased

- Synchronized `README.md` with the current registry size: 25 nodes across seven categories
- Corrected `main.review.tex` failure-count wording to distinguish failed tasks from failed runs
- Refreshed the `main.review.tex` appendix artifact so it matches the current node templates

## v1.0 - March 2026

### 21/02/2026
- Set up repository and defined architecture
- Created README and project skeleton
- Implemented core `NodeType` enum
- Created `nodes/registry.py`
- Created 10 initial nodes in `nodes/templates/`

### 26/02/2026
- Fixed virtual environment and installed dependencies
- Wrote code for all 10 initial nodes
- Created `core/` and `tests/` directories
- Implemented `compiler.py`, `validator.py`, `planner.py`, `cli.py`
- Topological ordering inside validator
- Added execution stub and compiler output normalization
- Added `reliability.py` and flag signalling (unused, later removed)
- Updated README

### 28/02/2026
- Normalized LLM output handling
- Fixed OpenAI integration
- Updated `registry.py` to store explicit function names so the compiler doesn't have to derive them
- Compiler now auto-generates glue code
- Added 13 new nodes, registry now at 26 nodes total

### 01/03/2026
- Built the benchmarking system (`harness.py`, `criteria.py`)

### 02/03/2026
- Updated benchmarking system
- Added input arity check to validator (CHECK 6)
- Fixed compiler bugs
- Added GNU AGPL v3 license

### 03/03/2026
- Ran initial tests across ~50 tasks

### 04/03/2026
- Expanded to 180 tasks across 6 sets (n=30 per set)
- Updated criteria, planner, and registry
- Stabilized baseline harness, added execution timeout

### 05/03/2026
- Added 20 probe tasks per set, now n=50 per set, 300 tasks total
- Probe tasks target the two main failure patterns: QueryEngine evasion and Logger placement
- Added ablation harness (not yet run)
- Updated README

### 06/03/2026 to 11/03/2026
- Switched from 5 runs to N=3 all-must-pass (task only passes if all 3 runs pass independently)
- Added Claude Sonnet 4.6 as a second baseline
- Upgraded GPT-4o baseline to GPT-4.1 (GPT-4o had patchy SQLAlchemy 2.0 compatibility, GPT-4.1 is a stronger comparison)
- Re-ran the full benchmark across all six sets with corrected criteria
- Final results: compiler 278/300 (93%), GPT-4.1 202/300 (67%), Claude Sonnet 4.6 187/300 (62%)
- Cleaned up repo for v1.0
- Updated README to match the finished system

# v1.1.0 - PlanCompiler 

### 18/03/2026
- Changed name from LLM Code Graph Compiler to PlanCompiler 
- Updated README.md to reflect changes
- Released as v1.1.0


# v1.2.0 - 
### 09/04/2026 
- Added repeat node support 
