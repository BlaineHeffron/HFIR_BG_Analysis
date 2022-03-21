import numpy as np
from math import ceil, floor
from numba import jit
SMALL_MERGESORT_NUMBA = 40
from numpy import zeros, int32, float32


@jit(nopython=True)
def merge_numba(A, Aux, lo, mid, hi):
    i = lo
    j = mid + 1
    for k in range(lo, hi + 1):
        if i > mid:
            Aux[k] = A[j]
            j += 1
        elif j > hi:
            Aux[k] = A[i]
            i += 1
        elif A[j] < A[i]:
            Aux[k] = A[j]
            j += 1
        else:
            Aux[k] = A[i]
            i += 1


@jit(nopython=True)
def merge_numba_two(A, Aux, lo, mid, hi, C, CAux):
    i = lo
    j = mid + 1
    for k in range(lo, hi + 1):
        if i > mid:
            Aux[k] = A[j]
            CAux[k] = C[j]
            j += 1
        elif j > hi:
            Aux[k] = A[i]
            CAux[k] = C[i]
            i += 1
        elif A[j] < A[i]:
            Aux[k] = A[j]
            CAux[k] = C[j]
            j += 1
        else:
            Aux[k] = A[i]
            CAux[k] = C[i]
            i += 1

@jit(nopython=True)
def insertion_sort_numba(A, lo, hi, desc=True):
    for i in range(lo + 1, hi + 1):
        key = A[i]
        j = i - 1
        if desc:
            while (j >= lo) & (A[j] < key):
                A[j + 1] = A[j]
                j -= 1
        else:
            while (j >= lo) & (A[j] > key):
                A[j + 1] = A[j]
                j -= 1
        A[j + 1] = key


@jit(nopython=True)
def insertion_sort_numba_two(A, lo, hi, C, desc=True):
    for i in range(lo + 1, hi + 1):
        key = A[i]
        second = C[i]
        j = i - 1
        if desc:
            while (j >= lo) & (A[j] < key):
                A[j + 1] = A[j]
                C[j + 1] = C[j]
                j -= 1
        else:
            while (j >= lo) & (A[j] > key):
                A[j + 1] = A[j]
                C[j + 1] = C[j]
                j -= 1
        A[j + 1] = key
        C[j + 1] = second

@jit(nopython=True)
def merge_sort_numba(A, Aux, lo, hi, desc=True):
    if hi - lo > SMALL_MERGESORT_NUMBA:
        mid = lo + ((hi - lo) >> 1)
        merge_sort_numba(Aux, A, lo, mid, desc)
        merge_sort_numba(Aux, A, mid + 1, hi, desc)
        if (desc and A[mid] > A[mid + 1]) or (not desc and A[mid] < A[mid + 1]):
            merge_numba(A, Aux, lo, mid, hi)
        else:
            for i in range(lo, hi + 1):
                Aux[i] = A[i]
    else:
        insertion_sort_numba(Aux, lo, hi, desc)

@jit(nopython=True)
def merge_sort_numba_two(A, Aux, lo, hi, C, CAux, desc=True):
    if hi - lo > SMALL_MERGESORT_NUMBA:
        mid = lo + ((hi - lo) >> 1)
        merge_sort_numba_two(Aux, A, lo, mid, C, CAux, desc)
        merge_sort_numba_two(Aux, A, mid + 1, hi, C, CAux, desc)
        if (desc and A[mid] > A[mid + 1]) or (not desc and A[mid] < A[mid+1]):
            merge_numba_two(A, Aux, lo, mid, hi, C, CAux)
        else:
            for i in range(lo, hi + 1):
                Aux[i] = A[i]
                CAux[i] = C[i]
    else:
        insertion_sort_numba_two(Aux, lo, hi, CAux, desc)

@jit(nopython=True)
def merge_sort_main_numba(A, sort="d"):
    """
    @param A: 1d vector
    @param sort: "descending or "ascending"
    @return: sorted 1d vector of same length
    """
    B = np.copy(A)
    Aux = np.copy(A)
    merge_sort_numba(Aux, B, 0, len(B) - 1, sort.startswith("d"))
    return B

@jit(nopython=True)
def merge_sort_two(A, C, sort="d"):
    #sorts C based on A
    B = np.copy(A)
    Aux = np.copy(A)
    second = np.copy(C)
    second_aux = np.copy(C)
    merge_sort_numba_two(Aux, B, 0, len(B) - 1, second_aux, second, sort.startswith("d"))
    return B, second

@jit(nopython=True)
def average_median(v, centerfrac=0.33):
    assert v.shape[0] and centerfrac <= 1
    if v.shape[0] == 0:
        return 0
    v = merge_sort_main_numba(v, sort="a")
    res = centerfrac * v.shape[0]
    if 1 > res:
        ndiscard = v.shape[0] - 1
    else:
        ndiscard = v.shape[0] - int(centerfrac * v.shape[0])
    istart = int(ndiscard / 2)
    iend = v.shape[0] - istart
    dsum = 0
    for i in range(istart, iend):
        dsum += v[i]
    return dsum / (iend - istart)


@jit(nopython=True)
def sum_range(v, r0, r1):
    if r0 >= 0:
        r0 = r0
    else:
        r0 = 0
    if not r0 < v.size:
        return 0
    if r1 < v.size:
        r1 = r1
    else:
        r1 = v.size - 1
    if not (r0 <= r1):
        return 0
    sum = 0
    for i in range(r0, r1 + 1):
        sum += v[i]
    return sum


@jit(nopython=True)
def integrate_lininterp_range(v, r0, r1):
    i0 = ceil(r0)
    d0 = i0 - r0
    i1 = floor(r1)
    d1 = r1 - i1
    if i0 <= i1:
        s = sum_range(v, i0, i1)
    else:
        s = 0
    if 0 <= i0 < v.size:
        s -= (1 - d0) * (1 - d0) / 2 * v[i0]
    if 1 <= i0 <= v.size:
        s += d0 * d0 / 2 * v[i0 - 1]
    if 0 <= i1 < v.size:
        s -= (1 - d1) * (1 - d1) / 2 * v[i1]
    if -1 <= i1 < v.size - 1:
        s += d1 * d1 / 2 * v[i1 + 1]
    return s


@jit(nopython=True)
def find_peaks(v, maxloc, sep):
    local_maxima = zeros(shape=(50,), dtype=int32)
    maxima_vals = zeros(shape=(50,), dtype=float32)
    local_maxpos = 100000
    max_index = 0
    global_maxpos = 0
    for i in range(1, v.shape[0]):
        if v[i] > v[i - 1]:
            local_maxpos = i
        elif v[i] < v[i - 1] and local_maxpos != 100000:
            lmax = int((local_maxpos + i - 1) / 2)
            local_maxima[max_index] = lmax
            maxima_vals[max_index] = v[lmax]
            max_index += 1
            if v[lmax] > v[global_maxpos]:
                global_maxpos = lmax
            if max_index >= 50:
                break
            local_maxpos = 100000
    local_maxima = remove_end_zeros(local_maxima)
    maxima_vals = remove_end_zeros(maxima_vals)
    if local_maxima.shape[0] == 1 and local_maxima[0] == 0 and maxima_vals[0] == 0:
        return 0
    maxima_vals, local_maxima = merge_sort_two(maxima_vals, local_maxima)
    if local_maxima.shape[0] == 1:
        maxloc[0] = local_maxima[0]
        return global_maxpos
    global_maxpos = local_maxima[0]
    maxloc[0] = global_maxpos
    max_index = 1
    for i in range(local_maxima.shape[0] - 1):
        within_range = True
        for j in range(max_index):
            if abs(local_maxima[i + 1] - maxloc[j]) <= sep * 2:
                within_range = False
                break
        if within_range:
            maxloc[max_index] = local_maxima[i + 1]
            max_index += 1
        if max_index > 4:
            break

    """
    current_candidate = local_maxima[0]
    for i in range(1, local_maxima.shape[0]):
        if (local_maxima[i] - current_candidate) <= sep:
            if v[local_maxima[i]] > v[current_candidate]:
                current_candidate = local_maxima[i]
        else:
            maxloc[max_index] = current_candidate
            max_index += 1
            if max_index > 4:
                return global_maxpos
            current_candidate = local_maxima[i]
    """
    return global_maxpos


@jit(nopython=True)
def remove_end_zeros(v, val=0):
    if v[0] == val and val == 0:
        return v[0:1]
    elif v[0] == val:
        return None
    for i in range(1, v.shape[0]):
        if v[i] == val:
            return v[0:i]

