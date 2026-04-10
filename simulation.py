import numpy as np
import matplotlib.pyplot as plt


def simulate_approach(START_HEIGHT=0.15, START_DIST=0.15):
    # Setup Plot
    plt.figure(figsize=(8, 8))
    plt.title("Imitation Learning Data: Varied Approach Trajectories")
    plt.xlabel("Lateral Distance (XY Error) [m]")
    plt.ylabel("Height (Z) [m]")
    plt.axhline(0, color="black", linestyle="--", label="Button Height")
    plt.axvline(0, color="black", linestyle="--")

    # Simulation Constants
    BUTTON_Z = 0.0

    BUTTON_PRE_Z = 0.01

    MIN_STEP = 0.002
    MAX_STEP = 0.03
    BASE_SPEED_GAIN = 0.1
    BASE_LATERAL_GAIN = 0.1  # Lower base to visualize curves better

    # --- Generate Multiple Trajectories ---
    # We create a random "style" parameter for each episode
    # style < 1.0: Aggressive Z (Go down first)
    # style > 1.0: Aggressive XY (Align first)
    styles = np.logspace(np.log10(0.1), np.log10(10.0), 10)

    for style in styles:
        # Initial State
        curr_z = START_HEIGHT
        curr_dist = START_DIST

        # Adjust Gains based on style
        # We preserve the original gains but skew them with the style parameter
        k_xy = BASE_LATERAL_GAIN * np.sqrt(style)
        k_z = BASE_SPEED_GAIN / np.sqrt(style)

        history_x = [curr_dist]
        history_z = [curr_z]

        # Control Loop Simulation
        steps = 0
        while (abs(curr_dist) > 0.002) and steps < 1000:
            steps += 1

            # 1. Calculate Errors
            dist_error = 0 - curr_dist  # Target is 0
            z_error = BUTTON_PRE_Z - curr_z

            # 2. XY Action (Lateral)
            # Apply gain, clamp for max speed safety (optional in sim)
            step_xy_mag = abs(dist_error) * k_xy
            step_xy_mag = np.clip(step_xy_mag, MIN_STEP, MAX_STEP)
            step_xy = np.sign(dist_error) * step_xy_mag
            # step_xy = dist_error * k_xy

            # 3. Z Action (Approach)
            step_z_mag = abs(z_error) * k_z
            step_z_mag = np.clip(step_z_mag, MIN_STEP, MAX_STEP)
            if abs(z_error)<=MIN_STEP:
                step_z_mag = 0
            step_z = np.sign(z_error) * step_z_mag
            # 4. Update State
            curr_dist += step_xy
            curr_z += step_z

            # # Stop if we hit the button
            #     curr_z = BUTTON_PRE_Z
            #     history_x.append(curr_dist)
            #     history_z.append(curr_z)
            #     break

            history_x.append(curr_dist)
            history_z.append(curr_z)
        print(f"style: { style}, steps = {steps}")
        ## now go down slowly
        while curr_z > BUTTON_Z:
            step_z = -0.001
            curr_dist += 0
            curr_z += step_z
            history_x.append(curr_dist)
            history_z.append(curr_z)
        # Plot result
        label = f"Style {style:.2f}"
        if style == styles[0]:
            label += " (Down First)"
        if style == styles[-1]:
            label += " (Align First)"
        plt.plot(history_x, history_z, label=label, alpha=0.7)

    plt.legend()
    plt.grid(True)
    plt.title(f"xy: {START_DIST}, z:{START_HEIGHT}")
    plt.savefig(f"simulation_figures/xy: {START_DIST}, z:{START_HEIGHT}.png")


if __name__ == "__main__":
    simulate_approach(0.15,-0.15)
    simulate_approach(0.10,-0.15)
    simulate_approach(0.05,-0.15)
    simulate_approach(0.15,-0.05)
    simulate_approach(0.10,-0.05)
    simulate_approach(0.05,-0.05)
    simulate_approach(0.05,-0.01)
    simulate_approach(0.05,-0)

