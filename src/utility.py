import cv2
import os
import numpy as np


def compute_wavelength_rgb(wavelength_data):
    result = []
    for wld in wavelength_data:
        # derive the color from the wavelengthData array
        rgb = wavelength_to_rgb(round(wld))
        result.append(rgb)
    result.reverse()
    return result


def compute_wavelength_rgb_bg(height, width, array, graph):
    result = np.full([height, width, 3], fill_value=255, dtype=np.uint8)
    for index, wl in enumerate(array):
        r, g, b = array[index]
        result[:, index, :] = [b, g, r]
    # overlay the graticule
    result = cv2.bitwise_and(graph, result)
    return result


def wavelength_to_rgb(nm):
    # from: Chris Webb https://www.codedrome.com/exploring-the-visible-spectrum-in-python/
    # returns RGB vals for a given wavelength
    gamma = 0.8
    max_intensity = 255
    factor = 0
    rgb = {"R": float(0), "G": float(0), "B": float(0)}
    if 380 <= nm <= 439:
        rgb["R"] = -(nm - 440) / (440 - 380)
        rgb["G"] = 0.0
        rgb["B"] = 1.0
    elif 440 <= nm <= 489:
        rgb["R"] = 0.0
        rgb["G"] = (nm - 440) / (490 - 440)
        rgb["B"] = 1.0
    elif 490 <= nm <= 509:
        rgb["R"] = 0.0
        rgb["G"] = 1.0
        rgb["B"] = -(nm - 510) / (510 - 490)
    elif 510 <= nm <= 579:
        rgb["R"] = (nm - 510) / (580 - 510)
        rgb["G"] = 1.0
        rgb["B"] = 0.0
    elif 580 <= nm <= 644:
        rgb["R"] = 1.0
        rgb["G"] = -(nm - 645) / (645 - 580)
        rgb["B"] = 0.0
    elif 645 <= nm <= 780:
        rgb["R"] = 1.0
        rgb["G"] = 0.0
        rgb["B"] = 0.0
    if 380 <= nm <= 419:
        factor = 0.3 + 0.7 * (nm - 380) / (420 - 380)
    elif 420 <= nm <= 700:
        factor = 1.0
    elif 701 <= nm <= 780:
        factor = 0.3 + 0.7 * (780 - nm) / (780 - 700)
    if rgb["R"] > 0:
        rgb["R"] = int(max_intensity * ((rgb["R"] * factor) ** gamma))
    else:
        rgb["R"] = 0
    if rgb["G"] > 0:
        rgb["G"] = int(max_intensity * ((rgb["G"] * factor) ** gamma))
    else:
        rgb["G"] = 0
    if rgb["B"] > 0:
        rgb["B"] = int(max_intensity * ((rgb["B"] * factor) ** gamma))
    else:
        rgb["B"] = 0
    # display no color as gray
    if (rgb["R"] + rgb["G"] + rgb["B"]) == 0:
        rgb["R"] = 155
        rgb["G"] = 155
        rgb["B"] = 155
    return rgb["R"], rgb["G"], rgb["B"]


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    # scipy
    # From: https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    """
    Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    from math import factorial
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    first_vals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    last_vals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((first_vals, y, last_vals))
    return np.convolve(m[::-1], y, mode='valid')


def peak_indexes(y, threshold=0.3, min_dist=1, threshold_abs=False):
    # from peakutils
    # from https://bitbucket.org/lucashnegri/peakutils/raw/f48d65a9b55f61fb65f368b75a2c53cbce132a0c/peakutils/peak.py
    """
    The MIT License (MIT)

    Copyright (c) 2014-2022 Lucas Hermann Negri

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not threshold_abs:
        threshold = threshold * (np.max(y) - np.min(y)) + np.min(y)

    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros, = np.where(dy == 0)

    # check if the signal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])

    if len(zeros):
        # compute first order difference of zero indexes
        zeros_diff = np.diff(zeros)
        # check when zeros are not chained together
        zeros_diff_not_one, = np.add(np.where(zeros_diff != 1), 1)
        # make an array of the chained zero indexes
        zero_plateaus = np.split(zeros, zeros_diff_not_one)

        # fix if leftmost value in dy is zero
        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)

        # fix if rightmost value of dy is zero
        if len(zero_plateaus) and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)

        # for each chain of zero indexes
        for plateau in zero_plateaus:
            l_plat = plateau.shape[0]
            left = dy[plateau[0] - 1]
            right = dy[plateau[l_plat - 1] + 1]
            middle = l_plat // 2
            # always in sorted order
            median = plateau[middle]
            for p in plateau:
                if p < median:
                    dy[p] = left
                else:
                    dy[p] = right

    # find the peaks by using the first order difference
    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0)
        & (np.hstack([0.0, dy]) > 0.0)
        & (np.greater(y, threshold))
    )[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks


def compute_wavelengths(width, pixels, wavelengths):
    wavelength_data = []

    if len(pixels) == 3:
        print("Calculating second order polynomial...")
        coefficients = np.poly1d(np.polyfit(pixels, wavelengths, 2))
        print(coefficients)
        c1 = coefficients[2]
        c2 = coefficients[1]
        c3 = coefficients[0]
        print("Generating Wavelength Data!\n\n")
        for pixel in range(width):
            wavelength = ((c1 * pixel ** 2) + (c2 * pixel) + c3)
            wavelength = round(wavelength, 6)  # because seriously!
            wavelength_data.append(wavelength)
        print("Done! Note that calibration with only 3 wavelengths will not be accurate!")

    if len(pixels) > 3:
        print("Calculating third order polynomial...")
        coefficients = np.poly1d(np.polyfit(pixels, wavelengths, 3))
        print(coefficients)
        # note this pulls out extremely precise numbers.
        # this causes slight differences in vals then when we compute manual, but hey ho, more precision
        # that said, we waste that precision later, but tbh, we wouldn't get that kind of precision in
        # the real world anyway! 1/10 of a nm is more than adequate!
        c1 = coefficients[3]
        c2 = coefficients[2]
        c3 = coefficients[1]
        c4 = coefficients[0]

        print("Generating Wavelength Data!\n\n")
        for pixel in range(width):
            wavelength = ((c1 * pixel ** 3) + (c2 * pixel ** 2) + (c3 * pixel) + c4)
            wavelength = round(wavelength, 6)
            wavelength_data.append(wavelength)

        # final job, we need to compare all the recorded wavelengths with predicted wavelengths
        # and note the deviation!
        # do something if it is too big!
        predicted = []
        # iterate over the original pixel number array and predict results
        for i in pixels:
            px = i
            y = ((c1 * px ** 3) + (c2 * px ** 2) + (c3 * px) + c4)
            predicted.append(y)

        # calculate 2 squared of the result
        # if this is close to 1 we are all good!
        corr_matrix = np.corrcoef(wavelengths, predicted)
        corr = corr_matrix[0, 1]
        r_sq = corr ** 2

        print("R-Squared={}".format(r_sq))

    return wavelength_data


def read_calibration(width):
    # read in the calibration points
    # compute second or third order polynomial, and generate wavelength array!
    # Les Wright 28 Sept 2022
    errors = 0
    try:
        print("Loading calibration data...")
        with open(os.path.expanduser('~/caldata.txt'), 'r') as f:
            # read both the pixel numbers and wavelengths into two arrays.
            lines = f.readlines()
            calibrated_frame_width = lines[0].strip()
            calibrated_frame_width = float(calibrated_frame_width)
            ratio = width / calibrated_frame_width
            line0 = lines[1].strip()  # strip newline
            pixels = line0.split(',')  # split on ,
            pixels = [int(int(i) * ratio) for i in pixels]  # convert list of strings to ints
            line1 = lines[2].strip()
            wavelengths = line1.split(',')
            wavelengths = [float(i) for i in wavelengths]  # convert list of strings to floats
    except:
        errors = 1

    try:
        if len(pixels) != len(wavelengths):
            # The Calibration points are of unequal length!
            errors = 1
        if len(pixels) < 3:
            # The Cal data contains less than 3 pixels!
            errors = 1
        if len(wavelengths) < 3:
            # The Cal data contains less than 3 wavelengths!
            errors = 1
    except:
        errors = 1

    if errors == 1:
        print("Loading of Calibration data failed (missing caldata.txt or corrupted data!")
        print("Loading placeholder data...")
        print("You MUST perform a Calibration to use this software!\n\n")
        pixels = [0, width / 2, width]
        wavelengths = [380, 560, 750]

    wavelength_data = compute_wavelengths(width, pixels, wavelengths)

    return wavelength_data


def closest(lst, K):
    # https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


def write_calibration(click_array, frame_width):
    px_data = []

    for _, _, pixel, wavelength_displayed in sorted(click_array, key=lambda x: x[0]):
        px_data.append(pixel)

    px_data = ','.join(map(str, px_data))  # convert array to string
    with open(os.path.expanduser('~/caldata.txt'), 'w') as f:
        f.write(str(frame_width) + '\r\n')
        f.write(px_data + '\r\n')

    print("Calibration Data Written!  Please update calibration file with known wavelengths for observed pixels.")
    return True


def generate_graticule(wavelength_data):
    low = wavelength_data[0]
    high = wavelength_data[-1]

    # find the closest 10th to the start and end
    first_ten = int(round(np.ceil(low / 10) * 10))  # closest without going under
    last_ten = int(round(np.floor(high / 10) * 10))  # closest without going over

    # find positions of every whole 10nm
    tens = []
    fifties = []
    for tenth_wl in range(first_ten, last_ten + 1, 10):
        pixel, wl = min(enumerate(wavelength_data), key=lambda x: abs(tenth_wl - x[1]))
        # If the difference between the target and result is <9 show the line
        # (otherwise depending on the scale we get dozens of number either end that are close to the target)
        if abs(tenth_wl - wl) < 1:
            if tenth_wl % 50 == 0:
                fifties.append([pixel, int(round(wl))])
            else:
                tens.append(pixel)

    return [tens, fifties]
