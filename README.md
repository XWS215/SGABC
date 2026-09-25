# SGABC

Supporting artifacts for the manuscript, in five parts:

1. **Scenario generation algorithm** — MATLAB implementations of SGABC, the baselines (ABC, PSO), the PSO + surrogate combination, and two ablations.
2. **Space reduction** — the dimension-sensitive space compression module, with its 2-D and 3-D sampling statistics.
3. **Surrogate model** — comparison of random, Latin hypercube and dynamic sampling, and of batch random forest, incremental stratified random forest and a neural network.
4. **Automated testing** — two complete PreScan + Simulink + CarSim environments (highway and urban intersection), each with `autotest.m`.
5. **Expert evaluation data** — the rating materials: 200 trajectory, 200 time-history and 200 risk-field plots, plus summary figures of the two-round ratings.

Parts 1–3 require MATLAB, Part 4 requires PreScan, MATLAB/Simulink and CarSim, and Part 5 needs nothing.
