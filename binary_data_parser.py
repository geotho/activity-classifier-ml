from data_row import DataRow

__author__ = 'George'

def print_rows_from_file(filename):
    with open(filename, "rb") as f:
        bytes = f.read(20)
        while bytes != b"":
            print(DataRow.from_row(bytes))
            bytes = f.read(20)

print_rows_from_file("assets/data/20150111143429.dat")
