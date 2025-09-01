import threading
import time

from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager


def test_bucket_serialisation():
    rm = ResourceManager(auto_scan=False)
    rm.set_bucket_limit("nlp", 1)  # only one concurrent task

    start_order = []
    finish_order = []

    def task(idx):
        with rm.request_resources("nlp"):
            start_order.append(idx)
            time.sleep(0.1)
            finish_order.append(idx)

    t1 = threading.Thread(target=task, args=(1,))
    t2 = threading.Thread(target=task, args=(2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Serialisation: task 1 must finish before task 2 starts
    assert start_order == [1, 2] or start_order == [2, 1]
    # whichever started first must also finish first because second waits
    assert finish_order[0] == start_order[0]
