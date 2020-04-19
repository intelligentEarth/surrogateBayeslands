import multiprocessing
manager = multiprocessing.Manager()
q = manager.Queue()
q.put([1])

