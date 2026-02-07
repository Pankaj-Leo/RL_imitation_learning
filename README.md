# Imitation Learning: Theory to Practice

A complete, reproducible implementation of fundamental imitation learning algorithms with detailed explanations of why they work (or fail catastrophically).

## The Core Problem

**Goal**: Teach a robot to act by observing human demonstrations.

**Challenge**: Behavior Cloning (BC) can achieve low training error but compounds to catastrophic test-time failure due to distribution shift. One mistake leads to unseen states, leading to more mistakes - a death spiral.

**Solution**: This repo implements the algorithms that actually work, with clear explanations of *why* BC fails and *how* the fixes address the root cause.

![RL_immitation](RL_immitation.png)

## What You'll See

### Complete Comparison (`getting_started.py`)

This is the **recommended starting point**. It runs all three algorithms side-by-side:

```bash
python getting_started.py
```

**Example output (varies by seed/environment):**
```
ALGORITHM 1: BEHAVIOR CLONING
  Training loss: low ← Looks great on expert states
  Success rate: low ← Often fails under covariate shift
  
ALGORITHM 2: DAGGER  
  Iteration 1: Success rate improves
  Iteration N: Success rate typically improves with aggregation
  
ALGORITHM 3: AGGRAVATE-INSPIRED (COST-SENSITIVE DAGGER)
  Iteration N: cost-sensitive variant can reduce high-risk mistakes (toy setting)
```

Plus generates `complete_comparison.png` with visualizations.

### Experiment 1: BC Fails Spectacularly

Run `demo_bc_failure.py` to see:
- Training error: 0.001 (excellent!)
- Crash rate: 95% (catastrophic!)
- **Output**: `bc_failure.png` showing crash trajectory

### Experiment 2: DAgger Actually Works

Run `demo_dagger_success.py` to see:
- Interactive learning over 10 iterations
- Crash rate drops from 94% → 2%
- **Output**: `dagger_success.png` and `dagger_trajectory.png`

## Algorithms Implemented

| Algorithm | Error Bound | Key Insight | Requirements |
|-----------|-------------|-------------|--------------|
| **BC** | O(εT²) | Supervised learning | Expert demos only |
| **DAgger** | O(εT) | Query expert on learner states | Interactive expert |
| **AggreVaTe** | O(εT) | Cost-sensitive classification | Expert Q-function |

## Project Structure

```
imitation_learning/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
│
├── getting_started.py           # 🎯 START HERE - Complete demo
├── demo_bc_failure.py          # BC failure demonstration
├── demo_dagger_success.py      # DAgger success demonstration
│
├── algorithms/
│   ├── __init__.py
│   ├── bc.py                   # Behavior Cloning (broken baseline)
│   ├── dagger.py               # DAgger (O(εT) solution)
│   └── aggravate.py            # AggreVaTe (value-aware learning)
│
├── core/
│   ├── __init__.py
│   ├── policies.py             # Neural network policies
│   ├── value_functions.py      # Q-networks, Value networks
│   └── expert.py               # Simulated expert demonstrators
│
└── environments/
    ├── __init__.py
    └── racetrack.py            # Cliff racetrack environment
```

## Key Files Explained

### Algorithms

**`algorithms/bc.py`** - Behavior Cloning
- Standard supervised learning approach
- Demonstrates O(εT²) catastrophic failure
- Includes `BehaviorCloningWithFeedback` showing latching behavior

**`algorithms/dagger.py`** - DAgger (Dataset Aggregation)
- Interactive learning with proper data aggregation
- Achieves O(εT) error bound
- Includes `AdaptiveDAgger` with uncertainty-based querying

**`algorithms/aggravate.py`** - AggreVaTe
- Advantage-based imitation learning
- Cost-sensitive classification
- Q-network training for advantage estimation

### Core Components

**`core/policies.py`** - Neural Network Policies
- `MLPPolicy`: Standard feedforward (deterministic/stochastic)
- `FeedbackAwarePolicy`: Includes previous action (shows feedback loops)
- `EnsemblePolicy`: Multiple models for uncertainty

**`core/value_functions.py`** - Value Functions
- `QNetwork`: Action-value function Q(s,a)
- `ValueNetwork`: State-value function V(s)
- `CostNetwork`: Discriminator for IRL
- `AdvantageEstimator`: Compute advantages from Q-functions

**`core/expert.py`** - Expert Demonstrators
- `OptimalRacecarDriver`: PID-controlled expert
- `SuboptimalDriver`: Intentionally suboptimal
- `HumanInterventionSimulator`: Simulates human takeover

### Environment

**`environments/racetrack.py`** - Simulation Environment
- `CliffRacetrack`: Demonstrates covariate shift
- Gymnasium-compatible interface
- Configurable track width and cliff penalty

## Key Concepts

### 1. Covariate Shift (Why BC Fails)

**Problem**: Training distribution ≠ test distribution

```
Training:  ████████████████  (expert on track)
Testing:   ████░░░░....      (robot drifts → crashes)
              ↑ No training data!
```

One mistake → new states → more mistakes → **death spiral**

Error compounds **quadratically**: O(εT²)

### 2. Data Aggregation (Why DAgger Works)

**The Critical Line**:
```python
# WRONG (catastrophic forgetting)
self.dataset = new_data

# RIGHT (aggregation)
self.dataset.append(new_data)
```

This single difference: 10x performance improvement

DAgger queries expert on learner's states → error is O(εT) not O(εT²)

### 3. Advantages (Why AggreVaTe Is Better)

Not all mistakes are equal:

```
Mistake at track center: advantage = 0.1 (safe)
Mistake at cliff edge: advantage = 100.0 (fatal)
```

AggreVaTe weights errors by consequences → better safety

## Implementation Highlights

### 1. Proper Data Aggregation
```python
# In algorithms/dagger.py
def add_data(self, states, actions):
    self.states_buffer.append(states)  # Aggregate!
    self.actions_buffer.append(actions)
```

### 2. Cost-Sensitive Classification
```python
# In algorithms/aggravate.py
advantage = Q(state, predicted) - Q(state, expert)
loss = advantage * mse(predicted, expert)  # Weight by cost
```

### 3. No-Regret Learning
```python
# Follow-The-Leader: minimize cumulative loss
policy = argmin_π Σ loss_i(π)
```

## Performance

| Algorithm | Training Error | Test Error | Real Numbers (ε=1%, T=100) |
|-----------|---------------|------------|---------------------------|
| BC | ε | εT² | 100 (catastrophic) |
| DAgger | ε | εT | 1 (manageable) |
| AggreVaTe | ε | εT/M | 0.01 (excellent) |





**Pankaj Somkuwar** - AI Engineer / AI Product Manager / AI Solutions Architect

- LinkedIn: [Pankaj Somkuwar](https://www.linkedin.com/in/pankaj-somkuwar/)
- GitHub: [@Pankaj-Leo](https://github.com/Pankaj-Leo)
- Website: [Pankaj Somkuwar](https://www.pankajsomkuwarai.com)
- Email: [pankaj.som1610@gmail.com](mailto:pankaj.som1610@gmail.com)

```

## License

MIT - use freely, cite appropriately.

---

