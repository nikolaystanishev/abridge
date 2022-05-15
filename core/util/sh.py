import atexit
import os
import signal
import subprocess
from threading import Event

child_processes = []

current_file_dir = os.path.dirname(os.path.realpath(__file__))


def sh(command, kill=False):
    pid = subprocess.Popen(command).pid
    if kill:
        child_processes.append(pid)


def start_web(wait=False):
    sh(['python3', current_file_dir + '/../../web/manage.py', 'runserver'], True)
    if wait:
        Event().wait()


def bootstrap():
    sh(['python3', current_file_dir + '/../../web/manage.py', 'migrate'])

    sh(['cd', current_file_dir + '/../../web/frontend/templates/frontend'])
    sh(['rm', '-rf', 'node_modules'])
    sh(['yarn', 'install'])
    sh(['cd', '-'])


def register_postactions():
    def cleanup():
        for pid in child_processes:
            os.kill(pid, signal.SIGTERM)

    atexit.register(cleanup)
