import time


class Timer(object):
    """
    Timer
    """
    def __init__(self):
        self.t1 = time.time()
        self.t2 = time.time()

    def start(self):
        self.t1 = time.time()

    def stop(self):
        self.t2 = time.time()

    def getMsec(self):
        return 1000.*(self.t2 - self.t1)

    def getSec(self):
        return self.t2 - self.t1

