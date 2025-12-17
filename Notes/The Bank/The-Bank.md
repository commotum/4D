We're trying to train a model that operates over a live, typed, graph‑structured world (entities + attributes + values + history) that is exposed to the model as a stream of transactions.

The data is naturally:
- event-sourced (history is a log),
- nonlinear (entities/blocks/pixels form a graph / tree / hyperdoc structure),
- multi-author (system vs self vs user), and
- multi-write per step (a tx writes many datoms at once).

The hardest part is not predicting values; it's:
1. creating new objects, and
2. referring back to them reliably over time (Parent pointers, entity IDs, block ownership, pixel ownership, etc.)

## World representation as a Short-Term-Memory (STM) bank

Let the model's "working context" at logical time $t$ be a bank of $L$ slots:

$$
B_t=\left\{b_{t, 1}, \ldots, b_{t, L}\right\}
$$

Each bank slot stores a segmented token:

$$
b_{t, j} \equiv x_{t, j}=\left[x_{t, j}^{(A)}\left|x_{t, j}^{(P)}\right| x_{t, j}^{(S)}\left|x_{t, j}^{(T)}\right| x_{t, j}^{(V)}\right]
$$

where:
- $x^{(A)}=$ Author (system/self/user/etc.)
- $x^{(P)}$ = Parent pointer (points to another entity / slot / handle)
- $x^{(S)}=$ Position (row/col index, block order, attribute slot, etc.)
- $x^{(T)}$ = Type (a finite categorical "value type" / schema type)
- $x^{(V)}=$ Value (decoded by a type-conditioned decoder)

