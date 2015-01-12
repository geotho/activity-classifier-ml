import struct
import math
from datetime import datetime


class DataRow:
    def __init__(self, time, x, y, z):
        self.time = time
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return str(self.time) + " " + str(self.x) + " " + str(self.y) + " " + str(self.z)

    @property
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    @classmethod
    def from_row(self, row):
        timestamp, x, y, z = struct.unpack('>qfff', row)
        timestamp = datetime.fromtimestamp(timestamp)
        return DataRow(timestamp, x, y, z)
