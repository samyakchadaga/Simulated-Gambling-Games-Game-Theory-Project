"""\
Repeated simulated-gambling experiment with variable maximum session length.

Experiment:
    Maximum rounds N = 10, 20, 30, 50, 70
    25 sessions per (N, game design, user profile) combination
    5 x 2 x 3 x 25 = 750 sessions total

Fixed across all sessions and test cases:
    entertainment utility e = 2.0
    initial risk cost h0 = 0.3
    risk-cost growth alpha = 0.125
    cost per round = 1.0
    win probability = 0.30

Randomised:
    initial virtual wager (8 to 12 credits)
    win/loss outcomes
    continue/stop decisions
    wager-escalation decisions

BRS is a model-defined behavioural proxy, not a clinical diagnosis.
The user profiles are hypothetical parameter settings, not clinical groups.
The designer payoff is an illustrative accounting value; this script does NOT
calculate a strategic equilibrium or Nash equilibrium.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. EXPERIMENT SETTINGS
SEED = 42
N_SESSIONS = 25
N_VALUES = [10, 20, 30, 50, 70]

# Fixed utility parameters.
ENTERTAINMENT_UTILITY = 2.0
INITIAL_RISK_COST = 0.3
RISK_COST_GROWTH = 0.125
COST_PER_ROUND = 1.0
WIN_PROBABILITY = 0.30

GAME_DESIGNS = {
    "Standard_RTP": 0.90,
    "Inflated_RTP": 1.50,
}

# Assumed user decision profiles, not clinical categories.
USER_PROFILES = {
    "Controlled": {
        "p_continue_if_negative": 0.05,
        "p_continue_if_positive": 0.95,
        "p_escalate_after_loss": 0.05,
        "p_escalate_without_loss": 0.01,
        "wager_increase_fraction": 0.05,
    },
    "Moderately_Persistent": {
        "p_continue_if_negative": 0.25,
        "p_continue_if_positive": 0.95,
        "p_escalate_after_loss": 0.25,
        "p_escalate_without_loss": 0.05,
        "wager_increase_fraction": 0.10,
    },
    "Highly_Persistent": {
        "p_continue_if_negative": 0.60,
        "p_continue_if_positive": 0.95,
        "p_escalate_after_loss": 0.60,
        "p_escalate_without_loss": 0.10,
        "wager_increase_fraction": 0.20,
    },
}

# Equal BRS weights are modelling assumptions.
W_L = 0.50
W_E = 0.50
assert np.isclose(W_L + W_E, 1.0)

OUTPUT_DIR = Path(__file__).resolve().parent



# 2. SIMULATE ONE SESSION

def simulate_session(rng, rtp, profile_name, profile, max_rounds):
    initial_wager = rng.uniform(8.0, 12.0)
    wager = initial_wager
    previous_round_was_loss = False

    L = 0  # Continued rounds where expected utility <= 0.
    E = 0  # Wager increases on rounds that are actually played.
    rounds_played = 0
    cumulative_net_gambling_result = 0.0
    cumulative_user_utility = 0.0

    # Illustrative designer accounting payoff, not a strategic equilibrium.
    engagement_value_per_round = 1.0
    monetization_value_per_round = 0.0
    design_cost = 2.0 if np.isclose(rtp, 0.90) else 3.0
    designer_payoff = -design_cost

    round_records = []

    for t in range(1, max_rounds + 1):
        wager_before_escalation = wager

        # Wager escalation is more likely after a previous loss.
        if t > 1:
            p_escalate = (
                profile["p_escalate_after_loss"]
                if previous_round_was_loss
                else profile["p_escalate_without_loss"]
            )
            if rng.random() < p_escalate:
                wager *= 1.0 + profile["wager_increase_fraction"]

        # Risk cost increases with the round number, using fixed h0 and alpha.
        h_t = INITIAL_RISK_COST + RISK_COST_GROWTH * (t - 1)

        # Expected utility before the stochastic outcome.
        Q_t = (
            wager * (rtp - 1.0)
            + ENTERTAINMENT_UTILITY
            - COST_PER_ROUND
            - h_t
        )

        p_continue = (
            profile["p_continue_if_negative"]
            if Q_t <= 0
            else profile["p_continue_if_positive"]
        )
        continues = rng.random() < p_continue

        if not continues:
            round_records.append({
                "round": t,
                "wager": wager,
                "expected_utility_if_played": Q_t,
                "continued": 0,
                "won": np.nan,
                "gross_return": 0.0,
                "net_gambling_result": 0.0,
                "risk_cost": h_t,
                "wager_increased_this_round": int(wager > wager_before_escalation),
            })
            break

        rounds_played += 1
        wager_increased = int(wager > wager_before_escalation)
        E += wager_increased

        # Stochastic win/loss outcome. Payout is calibrated so
        # E[gross return] = wager * RTP.
        won = rng.random() < WIN_PROBABILITY
        gross_return = wager * (rtp / WIN_PROBABILITY) if won else 0.0
        net_gambling_result = gross_return - wager
        previous_round_was_loss = not won

        if Q_t <= 0:
            L += 1

        realised_round_utility = (
            net_gambling_result
            + ENTERTAINMENT_UTILITY
            - COST_PER_ROUND
            - h_t
        )
        cumulative_net_gambling_result += net_gambling_result
        cumulative_user_utility += realised_round_utility
        designer_payoff += engagement_value_per_round + monetization_value_per_round

        round_records.append({
            "round": t,
            "wager": wager,
            "expected_utility_if_played": Q_t,
            "continued": 1,
            "won": int(won),
            "gross_return": gross_return,
            "net_gambling_result": net_gambling_result,
            "risk_cost": h_t,
            "wager_increased_this_round": wager_increased,
        })

    # Normalise by the opportunities available for this particular N.
    L_norm = L / max_rounds
    E_norm = E / (max_rounds - 1) if max_rounds > 1 else 0.0
    brs = 100.0 * (W_L * L_norm + W_E * E_norm)

    session_result = {
        "max_session_rounds_N": max_rounds,
        "rtp": rtp,
        "user_profile": profile_name,
        "entertainment_e_fixed": ENTERTAINMENT_UTILITY,
        "initial_risk_cost_h0_fixed": INITIAL_RISK_COST,
        "risk_growth_alpha_fixed": RISK_COST_GROWTH,
        "cost_per_round_fixed": COST_PER_ROUND,
        "win_probability_fixed": WIN_PROBABILITY,
        "initial_wager_randomised": initial_wager,
        "rounds_played": rounds_played,
        "L_nonpositive_expected_utility_continuations": L,
        "E_wager_increases": E,
        "L_normalised": L_norm,
        "E_normalised": E_norm,
        "cumulative_net_gambling_result": cumulative_net_gambling_result,
        "user_cumulative_realised_utility": cumulative_user_utility,
        "designer_payoff_assumption": designer_payoff,
        "BRS": brs,
    }
    return session_result, round_records


# 3. RUN ALL COMBINATIONS

def run_experiment():
    rng = np.random.default_rng(SEED)
    session_rows = []
    round_rows_all = []

    for max_rounds in N_VALUES:
        for design_name, rtp in GAME_DESIGNS.items():
            for profile_name, profile in USER_PROFILES.items():
                test_case = f"{design_name}__{profile_name}"
                print(f"Running N={max_rounds}: {test_case} ...")

                for session_id in range(1, N_SESSIONS + 1):
                    result, round_rows = simulate_session(
                        rng, rtp, profile_name, profile, max_rounds
                    )
                    result.update({
                        "game_design": design_name,
                        "test_case": test_case,
                        "session_id": session_id,
                    })
                    session_rows.append(result)

                    for row in round_rows:
                        row.update({
                            "max_session_rounds_N": max_rounds,
                            "game_design": design_name,
                            "user_profile": profile_name,
                            "test_case": test_case,
                            "session_id": session_id,
                        })
                        round_rows_all.append(row)

    sessions = pd.DataFrame(session_rows)
    rounds = pd.DataFrame(round_rows_all)

    summary = (
        sessions.groupby(
            ["max_session_rounds_N", "test_case", "game_design", "user_profile"],
            as_index=False,
        )
        .agg(
            sessions=("session_id", "count"),
            mean_rounds_played=("rounds_played", "mean"),
            mean_L=("L_nonpositive_expected_utility_continuations", "mean"),
            mean_E=("E_wager_increases", "mean"),
            mean_L_normalised=("L_normalised", "mean"),
            mean_E_normalised=("E_normalised", "mean"),
            mean_net_gambling_result=("cumulative_net_gambling_result", "mean"),
            mean_user_realised_utility=("user_cumulative_realised_utility", "mean"),
            mean_designer_payoff=("designer_payoff_assumption", "mean"),
            mean_BRS=("BRS", "mean"),
            median_BRS=("BRS", "median"),
            std_BRS=("BRS", "std"),
            p90_BRS=("BRS", lambda x: x.quantile(0.90)),
            p95_BRS=("BRS", lambda x: x.quantile(0.95)),
        )
    )

    sessions_path = OUTPUT_DIR / "variable_N_session_results.csv"
    rounds_path = OUTPUT_DIR / "variable_N_round_results.csv"
    summary_path = OUTPUT_DIR / "variable_N_test_case_summary.csv"
    sessions.to_csv(sessions_path, index=False)
    rounds.to_csv(rounds_path, index=False)
    summary.to_csv(summary_path, index=False)

    create_graphs(sessions, summary)

    expected_sessions = len(N_VALUES) * len(GAME_DESIGNS) * len(USER_PROFILES) * N_SESSIONS
    print("\nEXPERIMENT COMPLETE")
    print(f"Total sessions: {len(sessions):,} (expected {expected_sessions:,})")
    print(f"Recorded round decisions: {len(rounds):,}")
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nCreated CSV files:")
    print(sessions_path)
    print(rounds_path)
    print(summary_path)
    print("\nCreated graph files in the same directory:")
    print("variable_N_BRS_vs_N.png")
    print("variable_N_rounds_vs_N.png")
    print("variable_N_L_rate_vs_N.png")
    print("variable_N_E_rate_vs_N.png")
    print("variable_N_BRS_distributions.png")

    return sessions, rounds, summary



# 4. PLOT RESULTS

def create_graphs(sessions, summary):
    # Plot one line for each of the six design/profile combinations.
    metrics = [
        ("mean_BRS", "Mean BRS", "variable_N_BRS_vs_N.png"),
        ("mean_rounds_played", "Mean rounds played", "variable_N_rounds_vs_N.png"),
        ("mean_L_normalised", "Normalised L (loss-continuation rate)", "variable_N_L_rate_vs_N.png"),
        ("mean_E_normalised", "Normalised E (wager-escalation rate)", "variable_N_E_rate_vs_N.png"),
    ]

    for metric, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        for test_case, group in summary.groupby("test_case"):
            group = group.sort_values("max_session_rounds_N")
            ax.plot(
                group["max_session_rounds_N"],
                group[metric],
                marker="o",
                linewidth=1.8,
                label=test_case.replace("__", " / "),
            )
        ax.set_xlabel("Maximum session length (N)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} versus maximum session length")
        ax.set_xticks(N_VALUES)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=200)
        plt.close(fig)

    # Distribution plot: BRS for each N, separated by test case.
    fig, ax = plt.subplots(figsize=(12, 7))
    test_cases = list(sessions["test_case"].drop_duplicates())
    positions = []
    data = []
    labels = []
    pos = 1
    for test_case in test_cases:
        for n_value in N_VALUES:
            vals = sessions.loc[
                (sessions["test_case"] == test_case)
                & (sessions["max_session_rounds_N"] == n_value),
                "BRS",
            ].dropna().to_numpy()
            data.append(vals)
            positions.append(pos)
            labels.append(str(n_value))
            pos += 1
        pos += 1  # visual gap between test cases

    ax.boxplot(data, positions=positions, widths=0.65, showfliers=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Maximum session length N (groups separated by test case)")
    ax.set_ylabel("BRS")
    ax.set_title("BRS distributions across N and test cases")
    ax.grid(True, axis="y", alpha=0.3)
    # Label the six test-case blocks above the plot.
    for i, test_case in enumerate(test_cases):
        centre = i * (len(N_VALUES) + 1) + (len(N_VALUES) + 1) / 2
        ax.text(centre, 1.02, test_case.replace("__", " / "),
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "variable_N_BRS_distributions.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    sessions, rounds, summary = run_experiment()
