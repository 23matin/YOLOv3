import time


class Timer(object):
    """
    Timer
    """
    def __init__(self):
        self.start = time.time()
        self.end = time.time()

    def start(self):
        self.start = time.time()

    def stop(self):
        self.end = time.time()

    def getMsec(self):
        return 1000.*(self.end - self.start)

    def getSec(self):
        return self.end - self.start

