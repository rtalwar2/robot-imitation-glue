import time


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.time):
    """_summary_

    taken from: https://github.com/tlpss/diffusion_policy/blob/main/diffusion_policy/common/precise_sleep.py#L16
    """
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return
