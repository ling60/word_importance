import time


class Timer:
    def __init__(self, func_str=''):
        print('starting: {}'.format(func_str))
        self.start = time.time()

    def stop(self):
        print("finished in {:.2f} sec.".format(time.time() - self.start), flush=True)
