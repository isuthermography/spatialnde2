import time
import spatialnde2 as snde

class measurement_time_monotonic(snde.measurement_time):
    value=None
    def __init__(self,value):
        super().__init__("")
        self.value = value
        pass
    def seconds_since_epoch(self):
        return value
    def __eq__(self,other):
        return self.value==other.value
    def __ne__(self,other):
        return self.value!=other.value
    def __lt__(self,other):
        return self.value<other.value
    def __le__(self,other):
        return self.value<=other.value
    def __gt__(self,other):
        return self.value>other.value
    def __ge__(self,other):
        return self.value>=other.value
    pass

class measurement_clock_monotonic(snde.measurement_clock):
    def __init__(self):
        super().__init__("")
        pass

    def get_current_time(self):
        return measurement_time_monotonic(time.monotonic_ns())
    pass

clock = measurement_clock_monotonic()
time1 = clock.get_current_time()
time.sleep(1)
time2 = clock.get_current_time()
print("should be true",time1==time1)
print("should be false",time1>time2)
