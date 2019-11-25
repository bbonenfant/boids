from time import sleep, time
from typing import Any, Callable, Iterable, TypeVar

N = TypeVar("N")


def mean(data: Iterable[N]) -> N:
    """ Compute the mean of some collection of data defined with addition and multiplication operators. """
    if iter(data) is data:
        data = list(data)
    return sum(data) * (1 / len(data))


def set_framerate(rate: float) -> Callable:
    """ Set the framerate for a function. """
    frame_interval = 1 / rate if rate > 0 else 0.03

    def limiter(func: Callable) -> Callable:
        """ Initialize the limiter. """
        last = time()

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """ Perform the limiting. """
            nonlocal last

            # Perform the frame limiting.
            last, timestep = time(), time() - last
            sleep_time = 0 if timestep > frame_interval else frame_interval - timestep
            sleep(sleep_time)

            # Call the function.
            return func(*args, **kwargs, timestep=timestep)
        return wrapper
    return limiter
