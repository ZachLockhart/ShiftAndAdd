# ShiftAndAdd

shift_and_add.py is a tiny utility for testing some code used in processing image data for the upcoming ASM project.

input.txt explained:

line 1: path to data, eg /scrs1/guidedog/2023B999/
line 2: base filename, eg sgd.2023B999.231110.std.
#filenames for irtf instruments all start with the same format
line 3: filename end, eg .a.fits
# filenames for irtf instruments all end with the same format, mostly, we're assuming all beam 'a' reads
line 4: subwindow is a square, this is 1/2 the length of the square side
line 5: position of object center along horizontal axis
line 6: position of object center along vertical axis
lines 7+: individually selected filenumbers (don't need the 0's at front to pad)


Example of a range of values:
/scrs1/guidedog/2023B999/
sgd.2023B999.231110.std.
.a.fits
12
241
228
range
64
92

Example of curated values:
/scrs1/guidedog/2023B999/
sgd.2023B999.231110.std.
.a.fits
12
241
228
64
66
68
70
71
72
73
80
83
85
86
87
88
90
91
92