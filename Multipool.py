# credit: https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic

import multiprocessing

import multiprocessing.pool


class NonDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class Multipool(multiprocessing.pool.Pool):
    Process = NonDaemonProcess