from math import floor

def readtime(text):
    words = len(str(text.split(" ")))
    avg_read_time = 225
    return floor(words * 1 / avg_read_time)
