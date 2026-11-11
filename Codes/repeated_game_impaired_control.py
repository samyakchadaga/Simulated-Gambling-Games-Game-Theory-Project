"""
Repeated Game with Backward Induction and an Impaired-Control Extension

The script:
1. Solves the finite-horizon repeated game by backward induction.
2. Derives the optimal STOP/CONTINUE policy for every reachable state.
3. Adds a simple behavioural rule: when STOP is optimal, the player overrides
   that decision and continues with probability q_override.
4. Calculates the probability of three consecutive STOP overrides.
5. Exports summary/state CSVs and graphs.

Important:
- q_override values and the three-override event are illustrative assumptions.
- The output is a probability of a model-defined impaired-control pattern,
  NOT a clinically validated probability or diagnosis of gambling addiction.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# 1. Model parameters
PARAMS = {
    "W": 10.0,
    "e": 2.0,
    "c": 1.0,
    "h0": 0.3,
    "h1": 0.125,
    "delta": 0.95,
    "p_win": 0.30,
    "p_loss": 0.70,
    "beta_continue": 0.02,
    "beta_loss": 0.10,
    "lambda_risk": 2.0,
}

RTPS = [0.90, 1.50]
HORIZONS = [10, 20, 30, 50, 70]
Q_VALUES = [0.10, 0.30, 0.50]
K = 3  # Required consecutive STOP overrides

OUTPUT_DIR = Path("repeated_game_results")
OUTPUT_DIR.mkdir(exist_ok=True)



# 2. Backward induction
def solve_base_game(rtp, T):
    """
    State (n, losses):
      n      = number of rounds already played
      losses = cumulative number of losses

    Returns:
      V      = optimal value function
      policy = optimal STOP/CONTINUE decision
      Q      = value of choosing CONTINUE
    """
    p_win = PARAMS["p_win"]
    p_loss = PARAMS["p_loss"]
    beta_c = PARAMS["beta_continue"]
    beta_l = PARAMS["beta_loss"]
    delta = PARAMS["delta"]

    V = {}
    policy = {}
    Q = {}

    # Terminal condition: no value after the final round.
    for losses in range(T + 1):
        V[(T, losses)] = 0.0
        policy[(T, losses)] = "END"
        Q[(T, losses)] = None

    # Work backwards from round T to round 1.
    for n in range(T - 1, -1, -1):
        for losses in range(n + 1):
            accumulated_risk = beta_c * n + beta_l * losses
            round_number = n + 1

            risk_cost = (
                PARAMS["h0"] + PARAMS["h1"] * (round_number - 1)
            )

            immediate_utility = (
                PARAMS["W"] * (rtp - 1.0)
                + PARAMS["e"]
                - PARAMS["c"]
                - risk_cost
                - PARAMS["lambda_risk"] * accumulated_risk
            )

            expected_future_value = (
                p_win * V[(n + 1, losses)]
                + p_loss * V[(n + 1, losses + 1)]
            )

            continue_value = (
                immediate_utility + delta * expected_future_value
            )
            Q[(n, losses)] = continue_value

            if continue_value > 0:
                V[(n, losses)] = continue_value
                policy[(n, losses)] = "CONTINUE"
            else:
                V[(n, losses)] = 0.0
                policy[(n, losses)] = "STOP"

    return V, policy, Q



# 3. Probability of impaired-control pattern
def solve_model(rtp, T, q_override):
    """
    If backward induction says CONTINUE:
        player continues; override streak resets to zero.

    If backward induction says STOP:
        player stops with probability (1 - q_override);
        player continues despite STOP with probability q_override.

    Event:
        K consecutive decisions to continue when STOP is optimal.

    Probability is calculated by backward recursion over:
        (n, losses, current_override_streak)
    """
    p_win = PARAMS["p_win"]
    p_loss = PARAMS["p_loss"]

    V, policy, Q = solve_base_game(rtp, T)

    # P[n, losses, streak] = probability of eventually reaching K
    # consecutive STOP overrides from the given state.
    P = {}

    for losses in range(T + 1):
        for streak in range(K):
            P[(T, losses, streak)] = 0.0

    for n in range(T - 1, -1, -1):
        for losses in range(n + 1):
            for streak in range(K):

                if policy[(n, losses)] == "CONTINUE":
                    # A rational continuation breaks the override streak.
                    P[(n, losses, streak)] = (
                        p_win * P[(n + 1, losses, 0)]
                        + p_loss * P[(n + 1, losses + 1, 0)]
                    )

                else:
                    # Player may override the optimal STOP decision.
                    if streak + 1 == K:
                        probability_after_override = 1.0
                    else:
                        probability_after_override = (
                            p_win * P[(n + 1, losses, streak + 1)]
                            + p_loss * P[(n + 1, losses + 1, streak + 1)]
                        )

                    P[(n, losses, streak)] = (
                        q_override * probability_after_override
                    )

  
    mass = {(0, 0): 1.0}  # (losses, override streak)
    expected_rounds = 0.0
    expected_losses = 0.0
    probability_stop_before_event = 0.0

    for n in range(T):
        next_mass = {}

        for (losses, streak), state_probability in mass.items():
            if policy[(n, losses)] == "CONTINUE":
                expected_rounds += state_probability

                next_mass[(losses, 0)] = (
                    next_mass.get((losses, 0), 0.0)
                    + state_probability * p_win
                )
                next_mass[(losses + 1, 0)] = (
                    next_mass.get((losses + 1, 0), 0.0)
                    + state_probability * p_loss
                )

            else:
                # Player follows STOP.
                stop_probability = state_probability * (1 - q_override)
                probability_stop_before_event += stop_probability
                expected_losses += stop_probability * losses

                # Player overrides STOP and plays another round.
                override_probability = state_probability * q_override
                expected_rounds += override_probability

                if streak + 1 == K:
                    # The event has occurred; treat it as an endpoint.
                    expected_losses += override_probability * losses
                else:
                    next_mass[(losses, streak + 1)] = (
                        next_mass.get((losses, streak + 1), 0.0)
                        + override_probability * p_win
                    )
                    next_mass[(losses + 1, streak + 1)] = (
                        next_mass.get((losses + 1, streak + 1), 0.0)
                        + override_probability * p_loss
                    )

        mass = next_mass

    # Remaining probability mass reached the horizon.
    for (losses, streak), state_probability in mass.items():
        expected_losses += state_probability * losses

    summary = {
        "RTP": rtp,
        "horizon_T": T,
        "q_override": q_override,
        "required_consecutive_overrides": K,
        "initial_game_value": V[(0, 0)],
        "probability_impaired_control_pattern": P[(0, 0, 0)],
        "expected_rounds_until_stop_or_event": expected_rounds,
        "expected_losses_until_stop_or_event": expected_losses,
        "probability_stopping_before_event": probability_stop_before_event,
    }

    # Export all reachable backward-induction states.
    state_rows = []
    beta_c = PARAMS["beta_continue"]
    beta_l = PARAMS["beta_loss"]

    for n in range(T):
        for losses in range(n + 1):
            accumulated_risk = beta_c * n + beta_l * losses

            for streak in range(K):
                state_rows.append({
                    "RTP": rtp,
                    "horizon_T": T,
                    "q_override": q_override,
                    "round": n + 1,
                    "rounds_already_played": n,
                    "losses_so_far": losses,
                    "accumulated_risk": accumulated_risk,
                    "current_override_streak": streak,
                    "continue_value_from_backward_induction": Q[(n, losses)],
                    "optimal_action": policy[(n, losses)],
                    "probability_of_event_from_state": P[(n, losses, streak)],
                })

    return summary, state_rows



# 4. Run all scenarios
def main():
    summaries = []
    all_state_rows = []

    for rtp in RTPS:
        for T in HORIZONS:
            for q_override in Q_VALUES:
                summary, state_rows = solve_model(rtp, T, q_override)
                summaries.append(summary)
                all_state_rows.extend(state_rows)

    summary_df = pd.DataFrame(summaries)
    states_df = pd.DataFrame(all_state_rows)

    summary_df.to_csv(
        OUTPUT_DIR / "impaired_control_summary.csv", index=False
    )
    states_df.to_csv(
        OUTPUT_DIR / "backward_induction_and_behaviour_policy.csv",
        index=False,
    )

    parameters = [
        {"parameter": name, "value": value}
        for name, value in PARAMS.items()
    ]
    parameters.extend([
        {"parameter": "q_override_scenarios", "value": str(Q_VALUES)},
        {"parameter": "consecutive_override_threshold_K", "value": K},
    ])
    pd.DataFrame(parameters).to_csv(
        OUTPUT_DIR / "model_parameters.csv", index=False
    )

    # Graphs: probability vs horizon, separately for each RTP.
    for rtp in RTPS:
        plt.figure(figsize=(8, 5))

        for q_override in Q_VALUES:
            subset = summary_df[
                (summary_df["RTP"] == rtp)
                & (summary_df["q_override"] == q_override)
            ]
            plt.plot(
                subset["horizon_T"],
                subset["probability_impaired_control_pattern"],
                marker="o",
                label=f"q = {q_override:.1f}",
            )

        plt.xlabel("Maximum number of rounds")
        plt.ylabel("Probability of 3 consecutive STOP overrides")
        plt.title(f"Impaired-control pattern probability (RTP = {rtp:.2f})")
        plt.ylim(-0.03, 1.03)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"pattern_probability_RTP_{rtp:.2f}.png",
            dpi=180,
        )
        plt.close()

    # Graphs: compare RTP values for each q.
    for q_override in Q_VALUES:
        plt.figure(figsize=(8, 5))

        for rtp in RTPS:
            subset = summary_df[
                (summary_df["RTP"] == rtp)
                & (summary_df["q_override"] == q_override)
            ]
            plt.plot(
                subset["horizon_T"],
                subset["probability_impaired_control_pattern"],
                marker="o",
                label=f"RTP = {rtp:.2f}",
            )

        plt.xlabel("Maximum number of rounds")
        plt.ylabel("Probability of 3 consecutive STOP overrides")
        plt.title(f"RTP comparison (q = {q_override:.1f})")
        plt.ylim(-0.03, 1.03)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"RTP_comparison_q_{q_override:.1f}.png",
            dpi=180,
        )
        plt.close()

    # Backward-induction policy heatmaps for T=20.
    for rtp in RTPS:
        _, policy, _ = solve_base_game(rtp, 20)
        matrix = []

        for n in range(20):
            row = []
            for losses in range(20):
                if losses <= n:
                    row.append(
                        1 if policy[(n, losses)] == "CONTINUE" else 0
                    )
                else:
                    row.append(float("nan"))
            matrix.append(row)

        plt.figure(figsize=(8, 6))
        plt.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(
            ticks=[0, 1],
            label="Action (0 = STOP, 1 = CONTINUE)",
        )
        plt.xlabel("Losses accumulated")
        plt.ylabel("Rounds already played")
        plt.title(f"Backward-induction policy (RTP = {rtp:.2f}, T = 20)")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"BI_policy_RTP_{rtp:.2f}.png",
            dpi=180,
        )
        plt.close()

    print("\nSUMMARY RESULTS\n")
    print(summary_df.to_string(index=False))
    print(f"\nAll results saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
