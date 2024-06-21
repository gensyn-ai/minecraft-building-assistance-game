import _thread
import traceback


def wrap_thread_to_handle_exceptions(thread_func):
    def wrapper(*args, **kwargs):
        try:
            thread_func(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            _thread.interrupt_main()

    return wrapper
