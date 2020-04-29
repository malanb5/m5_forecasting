import datetime

def time_this(func, **kwargs):
    prev_time = datetime.datetime.now()
    func(kwargs["device_type"])
    cur_time = datetime.datetime.now()
    print("function took: %s sec" % (cur_time - prev_time))

def get_timestamp_str():
    d = datetime.datetime.now()
    return d.strftime("%Y-%m-%d_%H-%M")