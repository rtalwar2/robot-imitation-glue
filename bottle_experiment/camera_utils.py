import pyrealsense2 as rs


def freeze_auto_exposure(camera, warmup_frames=30):
    """
    Let the RealSense's auto-exposure converge for a few frames, then lock it by reading
    back the settled exposure/gain and disabling auto-exposure.

    Without this, a tool that grabs just one frame right after opening the camera (e.g.
    lid_touch_point_annotator.py) gets it before auto-exposure has converged -- dark and
    inconsistent -- while a tool that runs longer before capturing (e.g.
    calibrate_and_hover_bottle.py, which moves the robot first) sees a well-exposed image.
    Locking removes that inconsistency and makes captures repeatable across runs.
    """
    for _ in range(warmup_frames):
        camera.get_rgb_image_as_int()

    color_sensor = camera.pipeline.get_active_profile().get_device().first_color_sensor()
    exposure = color_sensor.get_option(rs.option.exposure)
    gain = color_sensor.get_option(rs.option.gain)
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    color_sensor.set_option(rs.option.exposure, exposure)
    color_sensor.set_option(rs.option.gain, gain)
    print(f"[camera] auto-exposure locked at exposure={exposure} gain={gain}")
