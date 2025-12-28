## 1. Basic Metadata

- Title: The Hidden Logic of Sudoku (Second Edition).
  Evidence: "The Hidden Logic of Sudoku (Second Edition)" (p. 1)

- Authors: Denis Berthier.
  Evidence: "Denis Berthier" (p. 1)

- Year: November 2007.
  Evidence: "November 2007" (p. 1)

- Venue (conference/journal/arXiv): Book.
  Evidence: "Book" (p. 1)

## 2. One-Sentence Contribution Summary

The book presents a predicate-logic framework of Sudoku resolution rules and introduces extended chain rules (including 3D chains) that "allow to solve almost all the puzzles." (p. 12, p. 13)

Evidence:
> The first edition of this book (May 2007) introduced a conceptual framework for
> Sudoku solving, where "resolution rules" played a central role. All the concepts
> were formalised in Predicate Logic (FOL), which (surprisingly) was a new idea:
> (p. 12)

> the new 3D chains, even limited to short lengths, allow to solve almost all the puzzles
> (99% of the random minimal puzzles with chains of length no more than five and
> 99.9% with chains of length no more than seven);
> (p. 13)

## 3. Tasks Evaluated

Task 1: Sudoku puzzle completion (solve a given grid).
- Task type: Reasoning / relational; Other (constraint satisfaction).
- Dataset(s) used: Royle17 collection; Sudogen0 collection; Sudogen17 collection.
- Domain: Sudoku number grids (9x9).
- Evidence:
> Given a 9x9 grid, partially filled with numbers from 1 to 9 (the "entries" of the
> problem, also called the "clues" or the "givens"), complete it with numbers from 1 to
> 9 so that in every of the nine rows, in every of the nine columns and in every of the
> nine disjoint blocks of 3x3 contiguous cells, the following property holds:
> (p. 18)

> there is at most one occurrence of each of these numbers.
> (p. 18)

> All our examples rely on three large puzzle collections:
> (p. 28)

> the first, hereafter named the Royle17 collection, has been assembled by the
> graph theorist Gordon Royle; it consists of the 36,628 known (non essentially equi-
> valent) minimal grids with a unique solution;
> (p. 28)

> the second, hereafter named the Sudogen0 collection, consists of 10,000
> puzzles randomly generated with the C generator suexg
> (p. 29)

> the third, hereafter named the Sudogen17 collection, consists of 10,000
> puzzles randomly generated with the same software as above, but using a different
> seed (17);
> (p. 29)

Task 2: Puzzle classification (classification results produced by the solver).
- Task type: Other (classification / benchmarking).
- Dataset(s) used: Puzzles with unique solutions (>56,000); based on the same puzzle collections.
- Domain: Sudoku number grids.
- Evidence:
> the current release of our solver (SudoRules 13) has been tested on
> more than 56,000 puzzles known to have a unique solution (and this produced the
> classification results in chapters XXI and XXIII);
> (p. 410)

## 4. Domain and Modality Scope

- Single domain? Yes, Sudoku puzzles on 9x9 number grids.
  Evidence:
> Given a 9x9 grid, partially filled with numbers from 1 to 9 (the "entries" of the
> problem, also called the "clues" or the "givens"), complete it with numbers from 1 to
> 9 so that in every of the nine rows, in every of the nine columns and in every of the
> nine disjoint blocks of 3x3 contiguous cells, the following property holds:
> (p. 18)

- Multiple domains within the same modality? Not specified in the paper.
- Multiple modalities? Not specified in the paper.
- Domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Sudoku puzzle completion | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | See quote below (p. 12). |
| Puzzle classification | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | See quote below (p. 12). |

Evidence:
> The first edition of this book (May 2007) introduced a conceptual framework for
> Sudoku solving, where "resolution rules" played a central role. All the concepts
> were formalised in Predicate Logic (FOL), which (surprisingly) was a new idea:
> (p. 12)

## 6. Input and Representation Constraints

- Fixed grid size and symbol set; fixed 3x3 blocks (constraint definition).
  Evidence:
> Given a 9x9 grid, partially filled with numbers from 1 to 9 (the "entries" of the
> problem, also called the "clues" or the "givens"), complete it with numbers from 1 to
> 9 so that in every of the nine rows, in every of the nine columns and in every of the
> nine disjoint blocks of 3x3 contiguous cells, the following property holds:
> (p. 18)

> there is at most one occurrence of each of these numbers.
> (p. 18)

- Padding/resizing requirements: Not specified in the paper.
- Fixed number of tokens: Not specified in the paper.
- Fixed dimensionality beyond the grid statement: Not specified in the paper.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified in the paper.
- Fixed or variable sequence length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Mechanisms for computational cost (windowing/pooling/pruning): Not specified in the paper.

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Not specified in the paper.
- Where it is applied: Not specified in the paper.
- Fixed/modified/ablated: Not specified in the paper.

## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: Not specified in the paper.
- Multiple positional encodings compared: Not specified in the paper.
- Claims that PE choice is not critical: Not specified in the paper.

## 10. Evidence of Constraint Masking

- Model size(s): Not specified in the paper.
- Dataset size(s):
  Evidence:
> All our examples rely on three large puzzle collections:
> (p. 28)

> the first, hereafter named the Royle17 collection, has been assembled by the
> graph theorist Gordon Royle; it consists of the 36,628 known (non essentially equi-
> valent) minimal grids with a unique solution;
> (p. 28)

> the second, hereafter named the Sudogen0 collection, consists of 10,000
> puzzles randomly generated with the C generator suexg
> (p. 29)

> the third, hereafter named the Sudogen17 collection, consists of 10,000
> puzzles randomly generated with the same software as above, but using a different
> seed (17);
> (p. 29)

> the current release of our solver (SudoRules 13) has been tested on
> more than 56,000 puzzles known to have a unique solution (and this produced the
> classification results in chapters XXI and XXIII);
> (p. 410)

- Performance gains attributed to architectural hierarchy (new chain rules), not model/data scaling:
  Evidence:
> the new 3D chains, even limited to short lengths, allow to solve almost all the puzzles
> (99% of the random minimal puzzles with chains of length no more than five and
> 99.9% with chains of length no more than seven);
> (p. 13)

- Training tricks: Not specified in the paper.

## 11. Architectural Workarounds

- 3D chains as more general extensions over 2D chains to increase solving power.
  Evidence:
> the newest topic is "3D" chains (chapters XXII and XXIII); these are the fully
> super-symmetric extensions, or the 3D counterparts, of all the 2D chains introduced
> in the first edition (which could be spotted as sequences of cells in either of the rc-,
> rn-, cn-, and bn- spaces); as 3D chains are more general but also more complex and
> more difficult to spot on a real grid than the 2D chains, they take place above them
> in a complexity hierarchy; they do not replace them; this is why Parts One to Three
> have been kept unchanged, apart from the presentation details mentioned above; the
> new 3D chains, even limited to short lengths, allow to solve almost all the puzzles
> (99% of the random minimal puzzles with chains of length no more than five and
> 99.9% with chains of length no more than seven);
> (p. 13)

- Extended Sudoku board for representation and practical use.
  Evidence:
> the extended Sudoku board defined in chapter II, the way to build and use it in
> practice, which were previously only available on my Web pages, have been fully
> integrated into the book;
> (p. 12)

- Compact notation for solution paths.
  Evidence:
> a new notation, the "nrc notation", is now used for displaying the solution
> paths of all the examples; being more compact, it allowed the introduction of Part
> Four without significantly increasing the total number of pages.
> (p. 13)

- Resolution techniques (e.g., coloring/tagging) to find rule patterns on grids.
  Evidence:
> chapter XXIV introduces the idea that one or more resolution technique(s) can be
> the implementation of a resolution rule and can thus help find the occurrences of its
> underlying pattern on a real grid; it gives precise examples for chains, building on
> classical ideas of colouring and tagging
> (p. 13)

## 12. Explicit Limitations and Non-Claims

- The book does not review all advanced rules proposed on the Web.
  Evidence:
> The first thing that has not been done in this book is a review of all the advanced
> rules that have been proposed, mainly on the Web (under varying names, in more or
> less clear formulations, with more or less defined scopes).
> (p. 412)

- No complete resolution theory without Trial and Error is achieved.
  Evidence:
> The second thing that has not been achieved in this book is the discovery of a
> complete resolution theory that would make no uniqueness assumption and that
> would not use Trial and Error.
> (p. 413)

- Some puzzles remain beyond known resolution rules.
  Evidence:
> no set of resolution rules is known that would allow to solve such exceptionally complex
> puzzles as Easter Monster.
> (p. 413)

- Open question about detecting no-solution puzzles.
  Evidence:
> does our strongest resolution theory (L13, or its weak extension L16 or the 3D theory M28) detect all the puzzles that have no
> solution? We have found no example that could not be detected.
> (p. 413)
