# Game-Theoretic Modeling of Simulated Gambling

A Python project exploring the interaction between simulated gambling
game design and user decisions using **game theory, backward induction,
and stochastic simulation**.

## Overview

The project compares two Return-to-Player (RTP) conditions:

-   **Standard RTP:** 90%
-   **Inflated RTP:** 150% (hypothetical)

It models a user's STOP/CONTINUE decisions over a finite number of
rounds, examines overrides of STOP recommendations, and simulates
hypothetical user profiles to calculate a Behavioral Risk Score (BRS).

## Methods

-   **Backward induction:** Solves the finite-horizon stopping problem
    and calculates the optimal user policy.
-   **STOP-override model:** Calculates the probability of three
    consecutive decisions to continue despite a STOP recommendation.
-   **Stochastic simulation:** Simulates 750 sessions across three
    hypothetical user profiles, two RTP conditions, and five session
    lengths.
-   **BRS analysis:** Uses sensitivity, specificity, Youden's J, and
    ROC-AUC to explore classification thresholds.

## Key Results

-   Under the assumed parameters, STOP is optimal initially at 90% RTP;
    CONTINUE is optimal initially at 150% RTP.
-   At 150% RTP, the initial value rises from **35.76** at 10 rounds to
    **42.65** at 30 rounds, then remains unchanged through 70 rounds.
-   The BRS classification AUC ranges from **0.683 to 0.740**,
    indicating moderate separation of the simulated profiles.

## Repository Structure

``` text
├── README.md
├── Codes
    └── repeated_gambling_simulation_variable_N.py
    └── repeated_game_impaired_control.py
├── results/
```


## Run

Requires Python 3 and the packages used by the scripts.

``` bash
git clone https://github.com/<your-username>/game-theoretic-simulated-gambling.git
cd game-theoretic-simulated-gambling/Results
python repeated_game_impaired_control.py
```

Run the profile simulation using its script filename.

## Limitations

The results depend on assumed parameters and hypothetical user profiles.
The 150% RTP condition is hypothetical. The impaired-control event and
BRS are model-defined measures, **not clinical addiction measures or
validated predictions of real-world behavior**.

## Contributors

-   Samyak P. Chadaga
-   Divyansh Tiwari
-   Samhith B. S.
-   Gurram Janakiram Naga Venkata Sai


Department of Electronics, SVNIT Surat, India.
