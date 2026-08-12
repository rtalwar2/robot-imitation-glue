import asyncio
import collections
import struct
import threading
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bleak import BleakClient, BleakScanner

# Configuration matching the Arduino sketch
DEVICE_NAME = "CaptainHook"
CHARACTERISTIC_UUID = "1A3AC130-31EE-758A-BC50-54A61958EF81"

# Data store for plotting (stores up to 200 data points)
MAX_POINTS = 200
times = collections.deque(maxlen=MAX_POINTS)
voltages = collections.deque(maxlen=MAX_POINTS)
start_time = time.time()


def notification_handler(sender, data: bytearray):
    """Callback function triggered when new BLE data arrives."""
    # Unpack 4-byte float (little-endian '<f')
    voltage = struct.unpack("<f", data)[0]
    elapsed_time = time.time() - start_time

    times.append(elapsed_time)
    voltages.append(voltage)
    print(f"Received: {voltage:.3f} V")


async def run_ble():
    """Scans, connects, and listens to notifications from the BLE device."""
    print(f"Scanning for device with name '{DEVICE_NAME}'...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)

    if not device:
        print(f"Could not find device '{DEVICE_NAME}'. Make sure it's powered on.")
        return

    print(f"Found device: {device.name} [{device.address}]")
    print("Connecting...")

    async with BleakClient(device) as client:
        print("Connected! Starting data notifications...")
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

        # Keep connection alive while the plot is open
        while True:
            await asyncio.sleep(1)


def start_ble_loop():
    """Runs the asyncio BLE event loop in a separate thread."""
    asyncio.run(run_ble())


# Start BLE thread in the background
ble_thread = threading.Thread(target=start_ble_loop, daemon=True)
ble_thread.start()

# Setup matplotlib live graph
fig, ax = plt.subplots()
(line,) = ax.plot([], [], "r-", lw=1.5, label="Sensor Voltage (A0)")

ax.set_title("Real-Time BLE Sensor Voltage")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
ax.set_ylim(-0.2, 3.5)
ax.grid(True)
ax.legend(loc="upper right")


def update_plot(frame):
    """Updates the graph animation."""
    if times:
        line.set_data(times, voltages)
        # Scroll the x-axis to follow new data
        ax.set_xlim(max(0, times[-1] - 10), max(10, times[-1]))
    return (line,)


# Animate plot every 50ms
ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=False)

plt.show()