import sys
from os.path import dirname, realpath, join

sys.path.insert(1, dirname(dirname(dirname((realpath(__file__))))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import retrieve_data
from numpy import arccos, array, pi, dot
from numpy.linalg import norm
from math import sin, cos


def angle(a, b):
    cosab = dot(a, b) / (norm(a) * norm(b))
    angle = arccos(cosab)
    b_t = b[[1, 0]] * [1, -1]
    if dot(a, b_t) < 0:
        angle = 2 * pi - angle
    return angle


def gen_orientation_key(theta, phi):
    return "{0:.1f} {1:.1f}".format(theta, phi)


def parse_orientation_key(key):
    data = key.split(" ")
    return float(data[0]), float(data[1])


def convert_coord_to_phi(Rx, Rz, Lx, Lz, convert_deg=True):
    """
    phi is the cart angle with 0 degrees being oriented west (Lx = Rx, Lz < Rz)
    """
    if convert_deg:
        return angle(array([1, 0]), array([Rz - Lz, Lx - Rx])) * 180 / pi
    else:
        return angle(array([1, 0]), array([Rz - Lz, Lx - Rx]))


def convert_cart_coord_to_det_coord(Rx, Rz, Lx, Lz, angle):
    """
    center of detector face is 16 inches south and .8 inch west of right corner (Rx, Rz in room coordinates) assuming cart is oriented north
    when the angle is 90 deg. The axis of rotation is 8.8 inches west of right corner, so detector face is 8 inches from axis of rotation
    assumes angle is in degrees
    """
    phi = convert_coord_to_phi(Rx, Rz, Lx, Lz, False)
    rotation_axis = (
    Rx + sin(phi) * 8.8 + cos(phi) * 16, Rz - cos(phi) * 8.8 + sin(phi) * 16)  # coordinate of rotation axis
    face_length = sin(pi * angle / 180) * 8
    face = [rotation_axis[0] - face_length * sin(phi), rotation_axis[1] + face_length * cos(phi)]
    return face


class CartScanFiles(HFIRBG_DB):
    def __init__(self, path=None):
        super().__init__(path)

    def retrieve_position_spectra(self, min_E_range=None):
        """
        retrieves position files and returns dictionary with key = (theta, phi)
        value = list of spectrum objects
        theta is the angle of the detector
        phi is the cart angle (0 degrees face west)
        ignore track scan spectra
        """
        data = self.query_position_files()
        mydata = {}
        if not data:
            print("No files found")
            return
        for row in data:
            if row["start_time"] < 0:  # corrupted data
                continue
            if row["track"] == 1:
                continue
            if 'lead' in row["run_description"]: # remove runs where we put lead down for testing
                continue
            if min_E_range is not None:
                max_E = row["A0"] + row["A1"] * 16384 + row["A1"] / 2.
                if min_E_range > max_E:
                    continue
            phi = convert_coord_to_phi(row["Rx"], row["Rz"], row["Lx"], row["Lz"])
            key = gen_orientation_key(row["angle"], phi)
            full_path = join(row["path"], row["filename"] + ".txt")
            data_tuple = (retrieve_data(full_path, self), convert_cart_coord_to_det_coord(row["Rx"], row["Rz"], row["Lx"], row["Lz"], row["angle"]))
            if key in mydata.keys():
                mydata[key].append(data_tuple)
            else:
                mydata[key] = [data_tuple]
        return mydata

    def retrieve_track_spectra(self):
        data = self.query_position_files()
        mydata = {}
        if not data:
            print("No files found")
            return
        for row in data:
            if row["start_time"] < 0:  # corrupted data
                continue
            if row["track"] == 0:
                continue
            phi = convert_coord_to_phi(row["Rx"], row["Rz"], row["Lx"], row["Lz"])
            key = gen_orientation_key(row["angle"], phi)
            full_path = join(row["path"], row["filename"] + ".txt")
            data_tuple = (retrieve_data(full_path, self), convert_cart_coord_to_det_coord(row["Rx"], row["Rz"], row["Lx"], row["Lz"], row["angle"]))
            if key in mydata.keys():
                mydata[key].append(data_tuple)
            else:
                mydata[key] = [data_tuple]
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
    coords = [0, 20, 0, 10, 90]
    print("coords: {0}, detector coords: {1}".format(coords, convert_cart_coord_to_det_coord(*coords)))
    coords = [21.0, 94.5, 38.0, 90.5]
    print("coords: {0}, phi: {1}".format(coords, convert_coord_to_phi(*coords)))
    coords = [0, 17.5, 0, 0, 90]
    print("coords: {0}, detector coords: {1}".format(coords, convert_cart_coord_to_det_coord(*coords)))
    coords = [0, 17.5, 0, 0, 0]
    print("coords: {0}, detector coords: {1}".format(coords, convert_cart_coord_to_det_coord(*coords)))
    coords = [17.5, 0, 0, 0, 0]
    print("coords: {0}, detector coords: {1}".format(coords, convert_cart_coord_to_det_coord(*coords)))
    coords = [17.5, 0, 0, 0, 90]
    print("coords: {0}, detector coords: {1}".format(coords, convert_cart_coord_to_det_coord(*coords)))
    coords = [0, 0, 17.5, 0, 90]
    print("coords: {0}, detector coords: {1}".format(coords, convert_cart_coord_to_det_coord(*coords)))
    coords = [0, 0, 17.5, 0, 0]
    print("coords: {0}, detector coords: {1}".format(coords, convert_cart_coord_to_det_coord(*coords)))
    coords = [0, 0, 0, 17.5, 90]
    print("coords: {0}, detector coords: {1}".format(coords, convert_cart_coord_to_det_coord(*coords)))
    coords = [0, 0, 0, 17.5, 0]
    print("coords: {0}, detector coords: {1}".format(coords, convert_cart_coord_to_det_coord(*coords)))
    coords = [21.0, 94.5, 38.0, 90.5, 90]
    print("coords: {0}, detector coords: {1}".format(coords, convert_cart_coord_to_det_coord(*coords)))


if __name__ == "__main__":
    test()
