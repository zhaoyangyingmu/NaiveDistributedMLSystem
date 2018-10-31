from server.worker import *


worker = Worker('', 5555, 5)
worker.start()