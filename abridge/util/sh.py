import os
import signal
import atexit


child_processes = []


def get_file_dir():
    return os.path.dirname(os.path.realpath(__file__))


def sh(command, kill=False):
    pid = os.system(command)
    if kill:
        child_processes.append(pid)


def start_web(wait=True):
    command = 'python3 ' + get_file_dir() + '/../../web/manage.py runserver'
    if not wait:
        command += ' &'
    sh(command, True)


def bootstrap():
    sh('python3 ' + get_file_dir() + '/../../web/manage.py migrate')


def register_postactions():
    def cleanup():
        for pid in child_processes:
            os.kill(pid, signal.SIGTERM)

    atexit.register(cleanup)
