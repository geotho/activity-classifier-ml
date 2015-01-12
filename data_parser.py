from pylab import *

__author__ = 'George'


def make_array_from_file(filename):
    datatype = np.dtype([('timestamp', '>i8'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
    data = np.fromfile(open(filename, "rb"), datatype)
    return data





print(make_array_from_file("assets/data/20150111143429.dat"))
