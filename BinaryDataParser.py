__author__ = 'George'

import struct
from datetime import datetime


class DataRow:
    def __init__(self, time, x, y, z):
        self.time = time
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return str(self.time) + " " + str(self.x) + " " + str(self.y) + " " + str(self.z)

    @classmethod
    def from_row(self, row):
        timestamp, x, y, z = struct.unpack('>qfff', row)
        timestamp = datetime.fromtimestamp(timestamp)
        return DataRow(timestamp, x, y, z)

def print_rows_from_file(filename):
    with open(filename, "rb") as f:
        bytes = f.read(20)
        while bytes != b"":
            print(DataRow.from_row(bytes))
            bytes = f.read(20)

print_rows_from_file("assets/data/20150111143429.dat")
