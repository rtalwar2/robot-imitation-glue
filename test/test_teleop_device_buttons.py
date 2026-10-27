"""the record buttons of a teleop device (e.g. the airo-teleop phone web UI) drive the collect_data state machine."""

from types import SimpleNamespace

from robot_imitation_glue.collect_data import Event, read_teleop_device_buttons


def agent_with_buttons(**raw_inputs):
    return SimpleNamespace(teleop_device=SimpleNamespace(last_raw_inputs=raw_inputs))


def test_button_press_sets_event():
    event = Event()
    read_teleop_device_buttons(agent_with_buttons(startRecording=True), event, {})
    assert event.start_recording and not event.stop_recording and not event.cancel_recording


def test_held_button_triggers_once():
    event = Event()
    pressed = read_teleop_device_buttons(agent_with_buttons(stopRecording=True), event, {})
    assert event.stop_recording

    event.clear()
    pressed = read_teleop_device_buttons(agent_with_buttons(stopRecording=True), event, pressed)
    assert not event.stop_recording

    # released and pressed again
    pressed = read_teleop_device_buttons(agent_with_buttons(stopRecording=False), event, pressed)
    read_teleop_device_buttons(agent_with_buttons(stopRecording=True), event, pressed)
    assert event.stop_recording


def test_cancel_button():
    event = Event()
    read_teleop_device_buttons(agent_with_buttons(cancelRecording=True), event, {})
    assert event.cancel_recording


def test_devices_without_buttons_are_ignored():
    event = Event()
    read_teleop_device_buttons(SimpleNamespace(), event, {})
    read_teleop_device_buttons(SimpleNamespace(teleop_device=SimpleNamespace()), event, {})
    read_teleop_device_buttons(agent_with_buttons(), event, {})
    assert not (event.start_recording or event.stop_recording or event.cancel_recording)
