### 21/02/2026
- Setting up repository and defining architecture.
- Create README
- Initialize Project Skeleton
- Implemented core NodeType enum
- Created nodes/registry.py 
- Created 10 test nodes in nodes/templates/

### 26/02/2026
- Fixed virtual environment
- install project dependencies
- Created code for all 10 nodes
- created core/ and tests/
- implemented compiler.py
- implemented validator.py
- implemented topological ordering inside validator
- implemented planner
- implemented cli
- added execution stub 
- added normalization for compiler output 
- added reliability.py and flag signalling
- Updated README.md

### 28/02/2026
- Normalized LLM output
- Fixed OpenAI integration
- Updated registry.py to have function names for easier recall
- Updated compiler to auto generate glue code.
- Added 13 new nodes in same domain (data pipeline) and updated registry

### 1/03/2026
- Created benchmarking system

### 2/03/2026
- Updated benchmarking system
- Updated validator to now include an extra test for input arity
- Fixed compiler issues
- Ran sample test, achieved 40% increase from gpt-4o (Compiler=9/10, GPT=5/10)
- added GNU AGPL v3 license 

### 3/03/2026
- Ran initial tests with ~50 tasks.
- Added benchmark tests for ~70 tests over 7 sets (n=10) 

### 4/03/2026
- Ran tests for 180 tasks, over 6 sets (n=30)
- Updated criteria, planner, registry
- Stabilized baseline test, added deadline.

### 5/03/2026
- Added 20 new tasks per set. n=50 per set and total n=300
- Altered harness and run_baseline to run 5 times for accurate testing.
