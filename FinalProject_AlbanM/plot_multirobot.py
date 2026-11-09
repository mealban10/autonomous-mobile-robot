import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

CSV_FILE = Path(__file__).resolve().parent / "results" / "loc" / "scenario_1_localization_PF.csv"

# Folder for all generated figures.
OUTPUT_DIR = "results"

# Create the output folder if it does not exist.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output figure paths.
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "multi_robot_trajectories_PF.png",
)

ERROR_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "multi_robot_localization_errors_PF.png",
)

ROBOT_COLORS = {
    "Robot 1": "dodgerblue",
    "Robot 2": "darkorange",
}


# ============================================================
# Load localization data
# ============================================================

def load_localization_data():
    """Load and validate the multi-robot localization CSV."""

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"Could not find {CSV_FILE}. "
            "Run the multi-robot simulation first."
        )

    df = pd.read_csv(CSV_FILE)

    required_columns = [
        "sim_time_s",
        "robot_id",
        "true_x",
        "true_y",
        "odom_x",
        "odom_y",
        "pf_x",
        "pf_y",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing CSV columns: {missing}"
        )

    # Sort samples chronologically for each robot.
    df = df.sort_values(
        ["robot_id", "sim_time_s"]
    ).reset_index(drop=True)

    # Calculate the Euclidean position error for each estimate.
    df["odom_error"] = np.sqrt(
        (df["odom_x"] - df["true_x"]) ** 2
        + (df["odom_y"] - df["true_y"]) ** 2
    )

    df["pf_error"] = np.sqrt(
        (df["pf_x"] - df["true_x"]) ** 2
        + (df["pf_y"] - df["true_y"]) ** 2
    )

    return df


# ============================================================
# Plot individual robot trajectories
# ============================================================

def plot_robot_trajectory(ax, robot_df, robot_id):
    """Plot ground truth, odometry and PF for one robot."""

    ax.plot(
        robot_df["true_x"],
        robot_df["true_y"],
        color="blue",
        linewidth=2,
        label="Ground truth",
    )

    ax.plot(
        robot_df["odom_x"],
        robot_df["odom_y"],
        color="red",
        linestyle="--",
        linewidth=1.8,
        label="Odometry",
    )

    ax.plot(
        robot_df["pf_x"],
        robot_df["pf_y"],
        color="green",
        linestyle="-.",
        linewidth=1.8,
        label="Particle Filter",
    )

    # Use the first recorded position as the starting point.
    start_x = robot_df["true_x"].iloc[0]
    start_y = robot_df["true_y"].iloc[0]

    ax.scatter(
        start_x,
        start_y,
        color="black",
        marker="o",
        s=70,
        label="Start position",
        zorder=5,
    )

    # Mark the last recorded position.
    end_x = robot_df["true_x"].iloc[-1]
    end_y = robot_df["true_y"].iloc[-1]

    ax.scatter(
        end_x,
        end_y,
        color="purple",
        marker="X",
        s=85,
        label="Last position",
        zorder=5,
    )

    ax.set_title(f"{robot_id} - Localization Trajectories")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


# ============================================================
# Plot combined ground-truth trajectories
# ============================================================

def plot_combined_trajectories(ax, df):
    """Plot all robots' ground-truth trajectories together."""

    for robot_id, robot_df in df.groupby("robot_id"):

        color = ROBOT_COLORS.get(robot_id, "gray")

        ax.plot(
            robot_df["true_x"],
            robot_df["true_y"],
            color=color,
            linewidth=2.2,
            label=robot_id,
        )

        # Starting position.
        ax.scatter(
            robot_df["true_x"].iloc[0],
            robot_df["true_y"].iloc[0],
            color=color,
            marker="o",
            edgecolors="black",
            s=85,
            zorder=5,
        )

        # Last recorded position.
        ax.scatter(
            robot_df["true_x"].iloc[-1],
            robot_df["true_y"].iloc[-1],
            color=color,
            marker="X",
            edgecolors="black",
            s=100,
            zorder=5,
        )

    ax.set_title("Multi-Robot Ground Truth Trajectories")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend()


# ============================================================
# Plot localization errors over time
# ============================================================

def plot_localization_errors(df):
    """Plot odometry and PF position errors for each robot."""

    robot_ids = df["robot_id"].unique()

    fig, axes = plt.subplots(
        len(robot_ids),
        1,
        figsize=(12, 4 * len(robot_ids)),
        squeeze=False,
    )

    for index, robot_id in enumerate(robot_ids):

        ax = axes[index, 0]

        robot_df = df[
            df["robot_id"] == robot_id
        ]

        ax.plot(
            robot_df["sim_time_s"],
            robot_df["odom_error"],
            color="red",
            linewidth=1.6,
            label="Odometry error",
        )

        ax.plot(
            robot_df["sim_time_s"],
            robot_df["pf_error"],
            color="green",
            linewidth=1.6,
            label="Particle Filter error",
        )

        ax.set_title(f"{robot_id} - Position Error")
        ax.set_xlabel("Simulation time [s]")
        ax.set_ylabel("Position error [m]")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()

    fig.savefig(
        ERROR_OUTPUT_FILE,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"Error plot saved to: "
        f"{os.path.abspath(ERROR_OUTPUT_FILE)}"
    )


# ============================================================
# Print localization error statistics
# ============================================================

def print_error_statistics(df):
    """Print position error metrics for every robot."""

    print("\n" + "=" * 65)
    print("MULTI-ROBOT LOCALIZATION ERROR STATISTICS")
    print("=" * 65)

    for robot_id, robot_df in df.groupby("robot_id"):

        odom_rmse = np.sqrt(
            np.mean(robot_df["odom_error"] ** 2)
        )

        pf_rmse = np.sqrt(
            np.mean(robot_df["pf_error"] ** 2)
        )

        odom_mae = robot_df["odom_error"].mean()
        pf_mae = robot_df["pf_error"].mean()

        odom_max = robot_df["odom_error"].max()
        pf_max = robot_df["pf_error"].max()

        print(f"\n{robot_id}")
        print("-" * 45)

        print(f"Samples: {len(robot_df)}")

        print("\nOdometry:")
        print(f"  Mean position error: {odom_mae:.4f} m")
        print(f"  Position RMSE:       {odom_rmse:.4f} m")
        print(f"  Maximum error:       {odom_max:.4f} m")

        print("\nParticle Filter:")
        print(f"  Mean position error: {pf_mae:.4f} m")
        print(f"  Position RMSE:       {pf_rmse:.4f} m")
        print(f"  Maximum error:       {pf_max:.4f} m")

        if odom_rmse > 0:
            improvement = (
                (odom_rmse - pf_rmse) / odom_rmse
            ) * 100

            print(
                f"\nPF improvement over odometry: "
                f"{improvement:+.2f}%"
            )

    print("\n" + "=" * 65)


# ============================================================
# Main
# ============================================================

def main():

    df = load_localization_data()

    robot_ids = df["robot_id"].unique()

    # Create one trajectory plot per robot plus one combined plot.
    fig, axes = plt.subplots(
        1,
        len(robot_ids) + 1,
        figsize=(7 * (len(robot_ids) + 1), 7),
        squeeze=False,
    )

    axes = axes[0]

    # Plot individual robot trajectories.
    for index, robot_id in enumerate(robot_ids):

        robot_df = df[
            df["robot_id"] == robot_id
        ]

        plot_robot_trajectory(
            axes[index],
            robot_df,
            robot_id,
        )

    # Plot all robots' ground-truth trajectories together.
    plot_combined_trajectories(
        axes[-1],
        df,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FILE,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"Trajectory plot saved to: "
        f"{os.path.abspath(OUTPUT_FILE)}"
    )

    print(f"Total samples: {len(df)}")

    # Create the separate error plot.
    plot_localization_errors(df)

    # Print quantitative localization metrics.
    print_error_statistics(df)

    plt.show()


if __name__ == "__main__":
    main()