from yj.environment import WORKING_DIR
import logging

def init(level):
    import yj.Timer as Timer
    import sys

    ds = Timer.get_timestamp_str()

    # https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
    logPath = WORKING_DIR + "/logs"
    fileName = "%s" % ds

    logFormatter = logging.Formatter("%(asctime)s [%(name)-5.5s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(level)

    return rootLogger