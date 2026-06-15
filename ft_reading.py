from collections import deque
import time

import matplotlib.pyplot as plt

from robot_imitation_glue.hardware.ipc_ft import FTSubscriber

sub = FTSubscriber("FT")
time.sleep(5)  # wait for subscriber to connect

data = deque()

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(-5, 0)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Fz")
ax.set_title("Fz (last 5 seconds)")
fig.tight_layout()

while True:
    now = time.time()
    fz = sub.get_FT()[2]
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