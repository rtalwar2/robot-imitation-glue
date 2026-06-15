from collections import deque
import time

import matplotlib.pyplot as plt

from airo_robots.manipulators.hardware.ur_rtde import URrtde

ur = URrtde(ip_address="10.42.0.162")

data = deque()

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(-5, 0)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Fz [N]")
ax.set_title("UR TCP Fz (last 5 seconds)")
fig.tight_layout()

try:
    while True:
        now = time.time()
        tcp_force = ur.rtde_receive.getActualTCPForce()
        fz = tcp_force[2]
        data.append((now, fz))

        while data and (now - data[0][0]) > 5:
            data.popleft()

        xs = [t - now for t, _ in data]
        ys = [v for _, v in data]
        line.set_data(xs, ys)

        if ys:
            y_min = min(ys)
            y_max = max(ys)
            pad = max(0.1, (y_max - y_min) * 0.1)
            ax.set_ylim(y_min - pad, y_max + pad)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
