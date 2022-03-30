import sys
from os.path import dirname, realpath, join

sys.path.insert(1, dirname(dirname(dirname((realpath(__file__))))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.analysis.Spectrum import SpectrumData
from src.utilities.util import retrieve_data
from numpy import arccos, array, pi, sum, sqrt, dot
from numpy.linalg import norm


def angle(a, b):
    cosab = dot(a, b) / (norm(a) * norm(b))
    angle = arccos(cosab)
    b_t = b[[1, 0]] * [1, -1]
    if dot(a, b_t) < 0:
        angle = 2 * pi - angle
    return angle

def gen_orientation_key(theta, phi):
    return "{0:.1f} {1:.1f}".format(theta,phi)

def parse_orientation_key(key):
    data = key.split(" ")
    return float(data[0]), float(data[1])


def convert_coord_to_phi(Rx, Rz, Lx, Lz):
    """
    phi is the cart angle with 0 degrees being oriented west (Lz = Rz, Lx < Rx)
    """
    return angle(array([1, 0]), array([Rx - Lx, Rz - Lz])) * 180 / pi


class CartScanFiles(HFIRBG_DB):
    def __init__(self, path=None):
        super().__init__(path)

    def RetrieveAllPositionFiles(self, min_E_range=None):
        """
        retrieves position files and returns dictionary with key = (theta, phi)
        value = list of spectrum objects
        theta is the angle of the detector
        phi is the cart angle (0 degrees face west)
        """
        data = self.query_position_files()
        mydata = {}
        if not data:
            print("No files found")
            return
        for row in data:
            if row["start_time"] < 0:  # corrupted data
                continue
            if min_E_range is not None:
                max_E = row["A0"] + row["A1"] * 16384 + row["A1"] / 2.
                if min_E_range > max_E:
                    continue
            phi = convert_coord_to_phi(row["Rx"], row["Rz"], row["Lx"], row["Lz"])
            key = gen_orientation_key(row["theta"], phi)
            full_path = join(row["path"], row["filename"] + ".txt")
            if key in mydata.keys():
                mydata[key].append(retrieve_data(full_path, self))
            else:
                mydata[key] = [retrieve_data(full_path, self)]
        return mydata

def test():
    coords = [10, 0, 20, 0]  # Rx, Rz, Lx, Lz,
    print("coords: {0}, phi: {1}".format(coords, convert_coord_to_phi(*coords)))
    coords = [20, 0, 10, 0]  # Rx, Rz, Lx, Lz,
    print("coords: {0}, phi: {1}".format(coords, convert_coord_to_phi(*coords)))
    coords = [0, 10, 0, 20]  # Rx, Rz, Lx, Lz,
    print("coords: {0}, phi: {1}".format(coords, convert_coord_to_phi(*coords)))
    coords = [0, 20, 0, 10]  # Rx, Rz, Lx, Lz,
    print("coords: {0}, phi: {1}".format(coords, convert_coord_to_phi(*coords)))


if __name__ == "__main__":
    test()
