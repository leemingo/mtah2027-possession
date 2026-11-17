"""3-state HMM decoding (H, A, D) with Viterbi, one graded half at a time.

Maximises  sum_t S[t, s_t]  -  sum_t C[s_{t-1}, s_t]  over label sequences. A change of team costs
C_SWITCH, so the other team has to accumulate that much control evidence before possession moves
(a touch or a deflection does not); the change is then placed where that evidence starts, not
after it, which a moving-window vote cannot do.
"""

import numpy as np

H, A, D = 0, 1, 2


def transition_costs(p):
    c = np.zeros((3, 3))
    c[H, A] = c[A, H] = p["C_SWITCH"]
    c[H, D] = c[A, D] = p["C_DEAD"]
    c[D, H] = c[D, A] = p["C_RESTART"]
    return c


def viterbi(scores, costs):
    """Best state path for scores (T, 3) and transition costs (3, 3). Deterministic tie-break."""
    T, K = scores.shape
    back = np.zeros((T, K), dtype=np.int8)
    value = scores[0].copy()
    neg_costs = -costs
    for t in range(1, T):
        cand = value[:, None] + neg_costs          # cand[i, j]: come from i, go to j
        best = np.argmax(cand, axis=0)             # first max wins -> deterministic
        back[t] = best
        value = cand[best, np.arange(K)] + scores[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = int(np.argmax(value))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(scores, blocks, costs, n_frames):
    """Labels 'H'/'A'/'D' on the graded blocks, 'X' elsewhere."""
    labels = np.full(n_frames, "X", dtype="<U1")
    letters = np.array(["H", "A", "D"])
    for _, first, last in blocks:
        path = viterbi(scores[first:last + 1], costs)
        labels[first:last + 1] = letters[path]
    return labels
