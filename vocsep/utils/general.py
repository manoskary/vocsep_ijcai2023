from __future__ import print_function
import sys
import threading
try:
    import thread
except ImportError:
    import _thread as thread


def quit_function(fn_name):
    print('{} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    thread.interrupt_main() # raises KeyboardInterrupt


def exit_after(s):
    '''
    use as decorator to exit process if
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                # print("TimeoutException: \n Skipping.")
                timer.cancel()
            return result
        return inner
    return outer


