
class Controller(object):

    def __init__(self, lock=None, update_event=None, rolling_event=None, num_for_update=10, thread_num=1):
        self.lock = lock
        self.update_event = update_event
        self.rolling_event = rolling_event
        self.update_num = num_for_update
        self.thread_num = thread_num

        self.counter = 0
        self.waiting_counter = 0
        self.set_rolling()

    def add_counter(self):
        with self.lock:
            self.counter += 1
            if self.rolling_event.is_set():
                if self.counter % self.update_num == 0:
                    self.rolling_event.clear()

    def check_update(self, agent_index):
        if self.rolling_event.is_set():
            return False
        elif self.thread_num == 1:
            return True
        elif agent_index != 0:
            with self.lock:
                self.waiting_counter += 1
                if self.waiting_counter == self.thread_num - 1:  # wait for all the workers stop
                    self.update_event.set()
            self.rolling_event.wait()
            return False
        else:            # agent_index == 0
            self.update_event.wait()
            return True

    def set_rolling(self):
        self.update_event.clear()
        self.waiting_counter = 0
        self.rolling_event.set()
