import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from astropy.io import fits
import gc
import matplotlib.cm as cm
from importlib import reload
from time import sleep
import os

global_fig = None
global_ax = None

# CONSTANTS, do not edit
CENTROID = 0
PEAK = 1


def showdata_nb(data):
    global global_fig, global_ax

    if global_fig is None:
        global_fig = plt.figure()
        global_ax = global_fig.add_subplot(111)  # Changed from 311 to 111
    else:
        global_ax.clear()

    global_ax.imshow(data, cmap=cm.jet, interpolation='nearest', aspect='auto')

    numrows, numcols = data.shape

    def my_format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = data[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    global_ax.format_coord = my_format_coord
    plt.show(block=False)


def showdata_nb_new(data):
    # Create a new figure and axes every time the function is called
    fig = plt.figure()
    ax = fig.add_subplot(111)  # One subplot occupying the whole figure

    ax.imshow(data, cmap=cm.jet, interpolation='nearest', aspect='auto')

    numrows, numcols = data.shape

    def my_format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = data[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = my_format_coord
    plt.show(block=False)


def show_data(data):
    showdata_nb_new(data)
    centroid_x, centroid_y = calculate_centroid(data)


def print_centroid(data):
    centroid_x, centroid_y = calculate_centroid(data)
    print("Centroid (x, y):", centroid_x, centroid_y)

def extract_subwindow(data, center_x, center_y, size):
    # Calculate the start and end indices for rows and columns
    row_start = max(center_y - size, 0)
    row_end = min(center_y + size + 1, data.shape[0])
    col_start = max(center_x - size, 0)
    col_end = min(center_x + size + 1, data.shape[1])

    # Extract the subwindow
    subwindow = data[row_start:row_end, col_start:col_end]
    return subwindow


def calculate_centroid(subwindow):
    # Get the indices of all pixels in the subwindow
    y_indices, x_indices = np.indices(subwindow.shape)

    # Calculate the weighted average of the indices
    total_weight = subwindow.sum()
    if total_weight == 0:
        return None  # Avoid division by zero

    centroid_x = (subwindow * x_indices).sum() / total_weight
    centroid_y = (subwindow * y_indices).sum() / total_weight

    return centroid_x, centroid_y


def calculate_peak(subwindow):
    peak_position = np.unravel_index(np.argmax(subwindow), subwindow.shape)
    peak_x, peak_y = peak_position

    return peak_x, peak_y



def shift_window(window, dx, dy):
    shifted = np.zeros_like(window)

    # Calculate the start and end indices for the original and shifted windows
    orig_y_start, orig_y_end = max(dy, 0), min(window.shape[0] + dy, window.shape[0])
    orig_x_start, orig_x_end = max(dx, 0), min(window.shape[1] + dx, window.shape[1])

    shift_y_start, shift_y_end = max(-dy, 0), min(window.shape[0] - dy, window.shape[0])
    shift_x_start, shift_x_end = max(-dx, 0), min(window.shape[1] - dx, window.shape[1])

    # Shift the window
    shifted[shift_y_start:shift_y_end, shift_x_start:shift_x_end] = window[orig_y_start:orig_y_end, orig_x_start:orig_x_end]

    return shifted


def writefits(data, fname):
   fdata = np.float32(data)
   hdu = fits.PrimaryHDU()
   hdu.data = fdata
   hdu.writeto(fname)


def get_filenames_range(filename_base, filename_ext, start_file_number, end_file_number):
    file_names = []
    for iii in range(start_file_number,end_file_number):
        file_number_string = "{:05d}".format(iii)
        file_names.append(filename_base + file_number_string + filename_ext)

    return file_names


def get_filenames_curated(filename_base, filename_ext, filenums):
    file_names = []
    for nnn in filenums:
        fn = "{:05d}".format(nnn)
        file_names.append(filename_base + fn + filename_ext)

    return file_names


def read_input_file(filename):
    with open(filename, 'r') as file:

        path = file.readline().strip()
        base_filename = file.readline().strip()
        file_extension = file.readline().strip()
        size = int(file.readline().strip())
        xxx = int(file.readline().strip())
        yyy = int(file.readline().strip())

        next_line = file.readline().strip()
        if next_line.startswith("range"):
            start = int(file.readline().strip())
            end = int(file.readline().strip())
            integers = [start, end]
        else:
            integers = [int(next_line)] + [int(line.strip()) for line in file]

    return path, base_filename, file_extension, size, xxx, yyy, integers


def shift_and_add(data, mean_x, mean_y, center_x, center_y):
    height, width = data[0].shape
    final_window = np.zeros((height, width), dtype=float)
    shifted_window = []
    mmm = 0
    for dd in data:
        shift_x = mean_x - center_x[mmm]
        shift_y = mean_y - center_y[mmm]

        temp_data = shift_window(dd, round(shift_x), round(shift_y))
        shifted_window.append(temp_data)
        final_window += shifted_window[mmm]
        final_window += temp_data

        mmm = mmm + 1

    return final_window



def write_file(filename, window, subwindow):
    try:
        os.remove(filename + ".fits")
    except FileNotFoundError:
        pass
    try:
        os.remove(filename + "sub.fits")
    except FileNotFoundError:
        pass

    writefits(window, filename + ".fits")
    writefits(subwindow, filename + "sub.fits")


def process_data(data, center_type):
    sub_data = []
    for dd in data:
        sub_data.append(extract_subwindow(dd, xx, yy, size))

    center_x = []
    center_y = []

    for dd in sub_data:
        if center_type == CENTROID:
            cx,cy = calculate_centroid(dd)
        else:
            cx,cy = calculate_peak(dd)
        center_x.append(cx)
        center_y.append(cy)

    mean_x = np.mean(center_x)
    mean_y = np.mean(center_y)

    final_window = shift_and_add(data, mean_x, mean_y, center_x, center_y)
    final_subwindow = shift_and_add(sub_data, mean_x, mean_y, center_x, center_y)

    show_data(final_window)
    show_data(final_subwindow)

    if center_type == CENTROID:
        write_file("centroid", final_window, final_subwindow)
    else:
        write_file("peak", final_window, final_subwindow)



# main() begins here
data_path, filename_base, filename_ext, size, xx, yy, filenums = read_input_file("input.txt")

print("data path is " + data_path)
print("base filename is " + filename_base)
print("filename extension is " + filename_ext)

if len(filenums) == 2:
    file_names = get_filenames_range(filename_base, filename_ext, filenums[0], filenums[1])
else:
    file_names = get_filenames_curated(filename_base, filename_ext, filenums)

data = []
for filename in file_names:
    data.append(fits.getdata(data_path + filename))

process_data(data, CENTROID)
process_data(data, PEAK)
