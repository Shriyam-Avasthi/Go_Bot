# Go_Bot: Simplified Go Game with Advanced AI Agents

A Python-based implementation of a simplified Go game featuring multiple AI agents with increasing sophistication. This project demonstrates various game-playing algorithms including minimax with alpha-beta pruning, Monte Carlo Tree Search (MCTS), and iterative deepening search.

## Overview

This is a streamlined version of Go with simplified rules and winning conditions, implemented with a GUI for human play and multiple AI agent implementations for competitive play. The project serves as a testbed for exploring different game-tree search algorithms and move evaluation strategies.

## My Contributions

This repository contains several AI agent implementations that I've developed, showcasing a progression of algorithmic complexity:

### AI Agent Implementations (in `group1.py`)

1. **Agent1** - Basic random agent that selects from legal actions
2. **Agent1v1** - Minimax with depth-limited search and board state snapshots
3. **Agent1v2** - Enhanced minimax with Zobrist hashing for transposition tables
4. **Agent1v3** - Minimax with alpha-beta pruning for improved performance
5. **Agent1v4** - Advanced minimax with move ordering and strategic evaluation weights
6. **Agent1v5** - Monte Carlo Tree Search (MCTS) implementation with:
   - Intelligent rollout policy prioritizing capture/save moves
   - Weighted move selection
   - Evaluation function for non-terminal positions
7. **Agent1v6** - **Primary agent** featuring:
   - Iterative deepening search with time management
   - Quiescence search for tactical positions
   - Transposition tables with alpha-beta pruning
   - Move history and best move tracking
   - Time-based search control
   - Intelligent move ordering based on:
     - Capture moves (highest priority)
     - Save moves for endangered groups
     - Threatening moves (reducing opponent liberties)

### Key Features

- **LightweightBoardHandler**: Optimized board representation using Union-Find for group management and NumPy arrays for faster computation
- **Zobrist Hashing**: Fast board state hashing for transposition tables
- **Move Evaluation**: Multi-factor evaluation considering:
  - Group liberation (liberties available)
  - Atari threats (groups with single liberty)
  - Piece mobility
  - Terminal win/loss states
- **Performance Profiling**: Multiple profiling outputs included for optimization analysis

### Evaluation and Testing

- `Evaluate.py` - Benchmarking script to test agent performance with detailed win rate statistics
- Supports single or multiple match evaluations
- Tests various agent configurations and time/depth parameters

## Technical Highlights

### Algorithm Features
- **Alpha-Beta Pruning**: Reduces search space for minimax algorithm
- **Transposition Tables**: Caches board positions to avoid redundant calculations
- **Iterative Deepening**: Progressively deeper searches within time constraints
- **Quiescence Search**: Extended search for tactical positions to avoid horizon effects
- **MCTS with Policy**: Monte Carlo Tree Search with move prioritization strategies

### Performance Optimizations
- NumPy-based board representation for fast operations
- Union-Find data structure for group connectivity
- Zobrist hashing for O(1) position comparison
- Lazy evaluation and early termination strategies

## Game Rules

This simplified Go variant includes:

### Legal Actions
- Players must place stones on opponent's liberties (when no endangered groups exist)
- **Capture Priority**: If opponent has groups with 1 liberty, those are forced moves
- **Defensive Priority**: If player has groups with 1 liberty, must defend them
- **Suicide Prevention**: Cannot place stones that would immediately lose your group

### Winning Conditions
- Remove any opponent's group (capture their stones)
- Leave opponent with no legal actions (rare occurrence ~1.6% in random play)

### Setup
- BLACK always moves first
- First move is fixed at board center (10, 10)
- 20x20 board

## Usage

### Setup

```bash
conda create -n ai_assign python=3.11
pip install -r requirements.txt
```

### Run a Match

**Interactive mode** (play against Agent1v6):
```bash
python match.py
```

**Agent vs Agent** (edit `match.py`):
```python
agent_black = Agent1v6('black', max_time=2.0)
agent_white = Agent1v6('white', max_time=2.0)
match = Match(agent_black=agent_black, agent_white=agent_white, gui=False)
match.start()
```

### Evaluate Agents

Run `Evaluate.py` to benchmark agent performance:
```bash
python Evaluate.py
```

This tests agents over multiple matches and reports:
- Win rates for each agent
- Draw rates
- Performance metrics

## Project Structure

```
Go_Bot/
├── game/
│   ├── go.py          # Core board logic and rules
│   ├── ui.py          # GUI visualization
│   └── util.py        # Utility classes
├── group1.py          # My AI agent implementations
├── group2.py          # Alternative agent
├── match.py           # Main game runner
├── Evaluate.py        # Benchmarking script
├── output*.prof       # Performance profiling data
└── requirements.txt
```

## Dependencies

- **pygame** - GUI rendering
- **numpy** - Numerical computations and array operations
- **tqdm** - Progress bars for evaluation

## Performance Analysis

Included `.prof` files (output.prof, output2.prof, etc.) contain profiling data from various agent implementations. Agent1v6 shows the best balance of strength and computational efficiency through:

- Time-managed iterative deepening
- Strategic move ordering
- Effective transposition table utilization
- Quiescence search for capturing tactics

## Future Improvements

- Neural network evaluation function training
- AlphaGo-style policy and value networks
- Parallelized MCTS (parallel playouts)
- Opening book implementation
- Dynamic time allocation based on position complexity

## References

- Minimax with Alpha-Beta Pruning: Classic game theory algorithm
- Monte Carlo Tree Search: Upper Confidence bounds applied to Trees (UCT)
- Zobrist Hashing: Fast incremental hashing for game positions

---

**Author**: Shriyam Avasthi  
**Last Updated**: October 2025
