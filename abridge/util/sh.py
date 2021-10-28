import os
import signal
import atexit


child_processes = []


def sh(command):
    child_processes.append(os.system(command))


def register_postactions():
    def cleanup():
        for pid in child_processes:
            os.kill(pid, signal.SIGTERM)

    atexit.register(cleanup)
