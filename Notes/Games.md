# Game and Puzzle Taxonomy

A categorized list of games and puzzles, organized by type and domain.

---

## 1. Single Player Games

### 1.1 Linguistic

- **Crossword**
- **Hangman**
- **Wordle**
- **Cryptogram**  
  _Example:_ `"YVCCF, RJSY!" → "HELLO, JAKE!"`
- **Word Ladder**  
  _Example:_ `COLD → CORD → CARD → WARD → WARM`
- **Rebus Puzzle**  
  _Example:_ `👁️🐝M`, `🌧️🏹`

### 1.2 Spatial

- **Sudoku**
- each row, column, and subsquare added up to the same number.
- **Futoshiki**
- **Killer Sudoku**
- **Minesweeper**
- **Tetris**
- **Snake**
- **Breakout**
- **Rubik's Cube**  
  _All variations, including 2×2 – 17×17_
- **Jigsaw Puzzles**
- **Sliding-Tile Puzzles**  
  _Examples: 15-puzzle, Rush Hour_
- **Towers of Hanoi**
- **Nonograms**
- **Solitaire**
- **Str8ts**


### 1.3 Strategic

- **PAC-MAN**

---

## 2. Multiplayer Games

### 2.1 Linguistic

- **Scrabble**
- **Boggle**
- **Bananagrams**
- **Taboo**
- **Codenames**
- **Scattegories**
- **Cards Against Humanity**
- **Pictionary**

### 2.2 Spatial

- **Tic-Tac-Toe**
- **Connect Four**
- **Chess**
- **Checkers**
- **Go**
- **Battleship**
- **Memory**
- **Pong**

### 2.3 Strategic

- **Settlers of Catan**
- **Monopoly**
- **Pokemon**
- **Keep Talking and Nobody Explodes**
- **Among Us**




https://www.puzzle-nonograms.com/
https://www.puzzle-sudoku.com/
https://www.puzzle-minesweeper.com/minesweeper-5x5-easy/
https://www.puzzle-jigsaw-sudoku.com/
https://www.puzzle-yin-yang.com/


DAT/PAT:

#
Subtest & what it asks you to do
Typical time-savers
1
Keyholes / Apertures – Decide which oddly-shaped “keyhole” an unfamiliar 3-D object will fit through without turning.
Picture the widest dimension first; mentally “drop” the object straight in. (eruditionprep.com)
2
Top-Front-End (TFE) / Orthographic Projection – Given two views of an object, pick its missing third view.
Trace corners that “line up” across views to avoid mis-counting hidden edges. (dat-prep.com)
3
Angle Ranking – Rank four closely spaced angles from smallest to largest.
Compare just the apex region; draw an imaginary arc to judge curvature. (bootcamp.com)
4
Hole-Punching (Paper Folding) – A square sheet is folded up to 3× and hole-punched; you choose the final unfolded pattern.
Work backwards: mirror the punch with every unfold, one fold at a time. (shemmassianconsulting.com)
5
Cube Counting – A block of cubes is painted on some faces, then broken apart. Count cubes with 0, 1, 2, or 3 painted faces.
Make a quick 3-column tally as you scan; memorize corner/edge/face counts. (bootcamp.com)
6
Pattern (3-D) Folding / Spatial Relations – Choose which 3-D solid results when a complex 2-D net is folded into a box-like shape.
Identify an “anchor” face, then see where a distinctive edge or dot must land. (shemmassianconsulting.com)


# Apple Paper Excerpt

3.1 Puzzle Environments

We evaluate LRM reasoning on four controllable puzzles spanning compositional depth, planning complexity, and distributional settings. The puzzles are defined below and illustrated in Fig. 3.

**Tower of Hanoi** is a puzzle featuring three pegs and $n$ disks of different sizes stacked on the first peg in size order (largest at bottom). The goal is to transfer all disks from the first peg to the third peg. Valid moves include moving only one disk at a time, taking only the top disk from a peg, and never placing a larger disk on top of a smaller one. The difficulty in this task can be controlled by the number of initial disks as the minimum number of required moves with $n$ initial disks will be $2^n-1$. However, in this work we do not grade for optimality of final solution and only measuring the correctness of each move and reaching the target state.

**Checker Jumping** is a one-dimensional puzzle arranging red checkers, blue checkers, and a single empty space in a line. The objective is to swap the positions of all red and blue checkers, effectively mirroring the initial configuration. Valid moves include sliding a checker into an adjacent empty space or jumping over exactly one checker of the opposite color to land in an empty space. No checker can move backward in the puzzle process. The complexity of this task can be controlled by the number of checkers: with $2 n$ checkers, the minimum number of moves required will be $(n+1)^2-1$.

**River Crossing** is a constraint satisfaction planning puzzle involving $n$ actors and their corresponding $n$ agents who must cross a river using a boat. The goal is to transport all $2 n$ individuals from the left bank to the right bank. The boat can carry at most $k$ individuals and cannot travel empty. Invalid situations arise when an actor is in the presence of another agent without their own agent present, as each agent must protect their client from competing agents. The complexity of this task can also be controlled by the number of actor/agent pairs present. For $n=2, n=3$ pairs, we use boat capacity of $k=2$ and for larger number of pairs we use $k=3$.

**Blocks World** is a block-stacking puzzle requiring rearrangement of blocks from an initial configuration into a specified goal configuration. The objective is to find the minimum number of moves needed for this transformation. Valid moves are restricted to the topmost block of any stack, which can be placed either on an empty stack or on top of another block. The complexity in this task can be controlled by the number of blocks present.