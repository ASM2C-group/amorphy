from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        from system_description import SystemInfo
        system = SystemInfo()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        log = f'\nTime elapsed for the execution of Function "{func.__name__}{args} {kwargs}" took {total_time:.4f} seconds. \n'
        print(log)
        system.additional_info(log)
        return result
    return timeit_wrapper
