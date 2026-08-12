import asyncio
import collections
import struct
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bleak import BleakClient, BleakScanner

DEVICE_NAME = "CaptainHook"
CHARACTERISTIC_UUID = "1A3AC130-31EE-758A-BC50-54A61958EF81"

# Data store for 3 channels
MAX_POINTS = 200
times = collections.deque(maxlen=MAX_POINTS)
v0_data = collections.deque(maxlen=MAX_POINTS)
v1_data = collections.deque(maxlen=MAX_POINTS)
v2_data = collections.deque(maxlen=MAX_POINTS)

start_time = time.time()
client_ref = None  # Reference to perform clean disconnect on Ctrl+C


def notification_handler(sender, data: bytearray):
    v0, v1, v2 = struct.unpack("<3f", data)
    elapsed_time = time.time() - start_time

    times.append(elapsed_time)
    v0_data.append(v0)
    v1_data.append(v1)
    v2_data.append(v2)

    print(f"Sensors -> A0: {v0:.2f}V | A1: {v1:.2f}V | A2: {v2:.2f}V")


async def run_ble():
    global client_ref
    print(f"Scanning for device '{DEVICE_NAME}'...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)

    if not device:
        print(f"Could not find device '{DEVICE_NAME}'.")
        return

    print(f"Found device: {device.name} [{device.address}]. Connecting...")

    async with BleakClient(device) as client:
        client_ref = client
        print("Connected! Listening for notifications... (Press Ctrl+C to exit)")
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            print("\nDisconnecting BLE gracefully...")
            await client.stop_notify(CHARACTERISTIC_UUID)


# Setup matplotlib live graph
fig, ax = plt.subplots()
(line0,) = ax.plot([], [], "r-", lw=1.5, label="Sensor A0")
(line1,) = ax.plot([], [], "g-", lw=1.5, label="Sensor A1")
(line2,) = ax.plot([], [], "b-", lw=1.5, label="Sensor A2")

ax.set_title("Real-Time 3-Channel Sensor Readings")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
ax.set_ylim(-0.2, 3.5)
ax.grid(True)
ax.legend(loc="upper right")


def update_plot(frame):
    if times:
        line0.set_data(times, v0_data)
        line1.set_data(times, v1_data)
        line2.set_data(times, v2_data)
        ax.set_xlim(max(0, times[-1] - 10), max(10, times[-1]))
    return line0, line1, line2


ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=False)


# Main execution handling Ctrl+C gracefully
if __name__ == "__main__":
    loop = asyncio.new_event_loop()

    # Function to stop the loop on close
    def on_close(event):
        plt.close("all")

    fig.canvas.mpl_connect("close_event", on_close)

    # Run BLE task asynchronously alongside matplotlib
    try:
        ble_task = loop.create_task(run_ble())

        # Start matplotlib GUI in main thread
        plt.show(block=False)

        while plt.fignum_exists(fig.number):
            loop.run_until_complete(asyncio.sleep(0.05))
            plt.pause(0.001)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Stopping program...")
    finally:
        if "ble_task" in locals():
            ble_task.cancel()
            try:
                loop.run_until_complete(ble_task)
            except asyncio.CancelledError:
                pass
        loop.close()
        print("Done. You can now rerun the script immediately!")
        sys.exit(0)