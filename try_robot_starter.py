from rtde_control import RTDEControlInterface
try:
    rtde_control = RTDEControlInterface("10.42.0.163")
except RuntimeError as e:
    print(
        f"Failed to connect to RTDE. Error:\n{e}"
    )