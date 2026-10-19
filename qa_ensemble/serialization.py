from functools import wraps
from threading import RLock

MODEL_LOCK = RLock()


def serialized(function):
    @wraps(function)
    def call(*args, **kwargs):
        with MODEL_LOCK:
            return function(*args, **kwargs)
    return call
