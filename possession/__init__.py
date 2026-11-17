"""Team possession estimation for More than a Hack 2027 (BAR-ATH).

Pipeline: ball source -> per-frame evidence (ball control, dead ball, Smart Tagging) ->
3-state (H/A/D) HMM decoded with Viterbi per half -> frame,label CSV.

    python -m possession.run --out evaluation/results/A.csv
"""
