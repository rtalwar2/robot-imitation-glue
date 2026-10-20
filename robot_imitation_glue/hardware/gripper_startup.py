"""Cold-start-safe construction of the Schunk grippers.

Every bottle-rig entrypoint (the collection launcher, eval, the recovery wrapper) needs a
SchunkGripperProcess, and none of them may trust the plain constructor: airo-mono's __init__
spawns the child that does all MODBUS I/O, sleeps 1.0 s and never checks the child came up. The
child's first act -- BKSModule(port) then MakeReady(), BEFORE the main loop's per-command
try/except -- sometimes gets no answer (USB/RS485 cold start, or the port briefly still held by
the previous process's dying child); the child dies right there, and the script's first real
gripper command fails as a misleading `_queue.Empty` on the parent's result-queue timeout. Seen
2026-10-07: in collection it detonated mid-episode, after the arms had already moved; the same
flake reproduces by launching airo-mono's schunk_process.py test repeatedly.

So grippers are constructed via make_gripper() only. The per-arm by-path strings live here too,
retiring the per-script literals; they name otter's physical topology (usb-0:2/12 = right arm,
"#rechts", the bottle hand; usb-0:7/11 = left arm, "#links") -- re-check with `ls -l
/dev/serial/by-path/` if an adapter is replugged.
"""

import time
from queue import Empty

from airo_robots.grippers.hardware.schunk_process import SchunkGripperProcess

GRIPPER_PORT_RIGHT = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:2:1.0-port0,12,115200,8E1"
GRIPPER_PORT_LEFT = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:7:1.0-port0,11,115200,8E1"


def make_gripper(port: str, attempts: int = 5, delay: float = 1.5) -> SchunkGripperProcess:
    """Construct the SchunkGripperProcess, retrying until its child survives the MODBUS init.

    Liveness is probed with a real queue round-trip (`speed`), not just `is_alive()`: a dead child
    is briefly a zombie that still reports alive, and the round-trip is what catches that window.
    A plain re-run of any of these scripts usually works -- that is exactly the behaviour this
    retries through, so a handshake that would have succeeded a second later never reaches the
    robot moves. Exhausting the attempts raises with the two real causes to check, not a queue
    stacktrace.
    """
    for attempt in range(1, attempts + 1):
        gripper = SchunkGripperProcess(usb_interface=port)
        healthy = False
        if gripper._process.is_alive():
            try:
                _ = gripper.speed  # completes only while the child's main loop is serving
                healthy = True
            except (Empty, RuntimeError):
                healthy = False
        if healthy:
            return gripper
        gripper.shutdown()
        if attempt < attempts:
            print(
                f"[gripper] child died during MODBUS init on {port} "
                f"(attempt {attempt}/{attempts}) -- retrying in {delay}s"
            )
            time.sleep(delay)
    raise RuntimeError(
        f"gripper never answered the MODBUS handshake after {attempts} attempts on {port} -- "
        "check it is powered, then that no other process holds the port (another rig script, "
        "airo-mono's standalone schunk_process.py test, or a lingering zombie of either)"
    )
