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
    print("Showing data")
    showdata_nb_new(data)
    print("shown")
    centroid_x, centroid_y = calculate_centroid(data)
    print("Centroid (x, y):", centroid_x, centroid_y)


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


def get_filenames(basefilename, filename_end, start_file_number, end_file_number):
    file_names = []
    for iii in range(start_file_number,end_file_number):
        file_number_string = "{:05d}".format(iii)
        file_names.append(basefilename + file_number_string + filename_end)


    return file_names


basefilename = "sgd.2023B999.231110.std."
filename_end = ".a.fits"

# files start at 32, go to 62, 00032 - 00061
start_file_number = 32
end_file_number = 61

file_names = get_filenames(basefilename, filename_end, start_file_number, end_file_number+1)

xx = 235
yy = 209
size = 12 

data = []
sub_data = []

for filename in file_names:
    data.append(fits.getdata("data/" + filename))

for dd in data:
    sub_data.append(extract_subwindow(dd, xx, yy, size))

center_x = []
center_y = []

for dd in sub_data:
    cx,cy = calculate_centroid(dd)
    center_x.append(cx)
    center_y.append(cy)
    print(str(cx) + ", " + str(cy))

mean_x = np.mean(center_x)
mean_y = np.mean(center_y)

print("Means: " + str(mean_x) + ", " + str(mean_y))

shifted_window = []
shifted_subwindow = []

final_window = np.zeros((512, 512), dtype=float)
final_subwindow = np.zeros((25, 25), dtype=float)

mmm = 0
for dd in data:
    shift_x = mean_x - center_x[mmm]
    shift_y = mean_y - center_y[mmm]

    temp_data = shift_window(dd, round(shift_x), round(shift_y))
    shifted_window.append(temp_data)
    final_window += shifted_window[mmm]
    final_window += temp_data

    mmm = mmm + 1

mmm = 0
for sd in sub_data:
    shift_x = mean_x - center_x[mmm]
    shift_y = mean_y - center_y[mmm]

    temp_data = shift_window(sd, round(shift_x), round(shift_y))
    shifted_subwindow.append(temp_data)
    final_subwindow += shifted_subwindow[mmm]
    final_subwindow += temp_data

    mmm = mmm + 1

show_data(final_window)
show_data(final_subwindow)

os.remove("final.fits")
os.remove("finalsub.fits")
writefits(final_window, "final.fits")
writefits(final_subwindow, "finalsub.fits")
