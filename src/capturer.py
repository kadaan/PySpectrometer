import utility
import constants
import cv2
import numpy as np


class Capturer:
    def __init__(self, args):
        self.__camera = None
        if args.source == constants.COMMAND_PI_CAMERA:
            from cameras.piCamera import PiCamera
            self.__camera = PiCamera(constants.WIDTH, constants.HEIGHT)
        elif args.source == constants.COMMAND_DEVICE_CAMERA:
            from cameras.deviceCamera import DeviceCamera
            self.__camera = DeviceCamera(args, constants.WIDTH, constants.HEIGHT)
        else:
            raise ValueError("Unsupported source: {}".format(args.source))

        # settings for peak detect
        self.peak_width = 50  # minimum distance between peaks
        self.peak_threshold = 20  # Threshold

        self.savitzky_golay_filter = 7  # savgol filter polynomial
        self.gain = 0
        self.hold_peaks = False
        self.pause = False

        if not self.__camera.is_opened():
            raise ValueError("Unable to open video source", args.source)

        # Go grab the computed calibration data
        self.__wavelength_data = utility.read_calibration(constants.WIDTH)

        self.__graph_base = self.build_graph_base(self.__wavelength_data)
        self.__graph = np.copy(self.__graph_base)
        self.__image = None

        self.__raw_intensity = np.zeros(constants.WIDTH)

        self.__pts = np.zeros((constants.WIDTH + 2, 2), dtype=int)
        self.__pts[0] = [constants.WIDTH, constants.SPECTRUM_HEIGHT - 1]
        self.__pts[1] = [0, constants.SPECTRUM_HEIGHT - 1]
        self.__pts[2:, 0] = np.arange(constants.WIDTH)
        self.__mask = np.zeros(self.__graph.shape[:2], np.uint8)

        wavelength_data_rgbs = utility.compute_wavelength_rgb(self.__wavelength_data)
        self.__wavelength_data_rgbs_bg = utility.compute_wavelength_rgb_bg(constants.SPECTRUM_HEIGHT, constants.WIDTH,
                                                                           wavelength_data_rgbs, self.__graph_base)

    @staticmethod
    def build_graph_base(wavelength_data):
        # generate the graticule data
        tens, fifties = utility.generate_graticule(wavelength_data)

        # blank image for Graph, filled white
        result = np.full([constants.SPECTRUM_HEIGHT, constants.WIDTH, 3], fill_value=255, dtype=np.uint8)

        # vertical lines every whole 10nm
        for position in tens:
            cv2.line(result, (position, 15), (position, constants.SPECTRUM_HEIGHT), (200, 200, 200), 1)

        # vertical lines every whole 50nm
        for position_data in fifties:
            cv2.line(result, (position_data[0], 15), (position_data[0], constants.SPECTRUM_HEIGHT), (0, 0, 0), 1)
            cv2.putText(result, "{}nm".format(position_data[1]), (position_data[0] - constants.TEXT_OFFSET, 12),
                        constants.FONT, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # horizontal lines
        for i in range(constants.SPECTRUM_HEIGHT):
            if i >= 64 and i % 64 == 0:  # suppress the first line then draw the rest...
                cv2.line(result, (0, i), (constants.WIDTH, i), (100, 100, 100), 1)

        return result

    def get_graph(self):
        if self.__camera.is_opened():
            if self.pause:
                return True, self.__image, self.__graph

            ret, frame = self.__camera.capture()
            if ret:
                x = 0  # origin of the horiz
                # h = 80  # height of the crop
                y = int((constants.HEIGHT / 2) - (constants.VIDEO_HEIGHT / 2))  # origin of the vertical crop
                w = constants.WIDTH  # width of the crop
                self.__image = frame[y:y + constants.VIDEO_HEIGHT, x:x + w]
                bw_image = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
                rows, cols = bw_image.shape
                halfway = int(rows / 2)
                # show our line on the original image
                # now a 3px wide region
                cv2.line(self.__image, (0, halfway - 2), (constants.WIDTH, halfway - 2), (255, 255, 255), 1)
                cv2.line(self.__image, (0, halfway + 2), (constants.WIDTH, halfway + 2), (255, 255, 255), 1)
                self.__image = cv2.cvtColor(self.__image, cv2.COLOR_BGR2RGB)

                # np.copyto(messages, banner_image)
                # reset the graph to the base
                np.copyto(self.__graph, self.__graph_base)

                num_mean = 3  # average the data from this many rows of pixels

                # get the mean value for each column spanning "num_mean" rows
                current_intensities = np.uint8(
                    np.mean(bw_image[halfway - (num_mean // 2):halfway + (num_mean // 2) + 1, :], axis=0))

                if self.hold_peaks:
                    # find the maximums, doing so in-place
                    np.maximum(self.__raw_intensity, current_intensities, casting="no", out=self.__raw_intensity)
                else:
                    self.__raw_intensity = current_intensities

                np.clip(self.__raw_intensity, 0, 255, out=self.__raw_intensity)

                # intensity = None
                if not self.hold_peaks:
                    intensity = utility.savitzky_golay(self.__raw_intensity, 17, self.savitzky_golay_filter)
                    intensity = np.array(intensity, dtype=np.int32)
                else:
                    intensity = np.int32(self.__raw_intensity)

                np.clip(intensity, 0, 255, out=intensity)

                # draw intensity data
                self.__pts[2:, 1] = constants.SPECTRUM_HEIGHT - 1 - intensity
                # mask holds where we should draw the rainbow
                self.__mask.fill(0)
                cv2.drawContours(self.__mask, [self.__pts], -1, 1, -1, cv2.LINE_AA)
                mask3d = np.dstack([self.__mask] * 3)
                # draw the rainbow using the mask onto graph
                np.putmask(self.__graph, mask3d, self.__wavelength_data_rgbs_bg)
                # draw the black line
                self.__graph[self.__pts[2:, 1], self.__pts[2:, 0]] = 0

                # find peaks and label them
                thresh = int(self.peak_threshold)  # make sure the data is int.
                peak_intensities = utility.peak_indexes(intensity, threshold=thresh / max(intensity),
                                                        min_dist=self.peak_width)

                for i in peak_intensities:
                    height = intensity[i]
                    height = constants.SPECTRUM_HEIGHT - constants.FLAGPOLE_HEIGHT - height
                    wavelength = round(self.__wavelength_data[i], 1)
                    cv2.rectangle(self.__graph, ((i - constants.TEXT_OFFSET) - 2, height),
                                  ((i - constants.TEXT_OFFSET) + 60, height - 15),
                                  (0, 255, 255), -1)
                    cv2.rectangle(self.__graph, ((i - constants.TEXT_OFFSET) - 2, height),
                                  ((i - constants.TEXT_OFFSET) + 60, height - 15), (0, 0, 0), 1)
                    cv2.putText(self.__graph, "{}nm".format(wavelength), (i - constants.TEXT_OFFSET, height - 3),
                                constants.FONT, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                    # flagpoles
                    cv2.line(self.__graph, (i, height), (i, height + constants.FLAGPOLE_HEIGHT), (0, 0, 0), 1)

                return ret, self.__image, self.__graph

        return False, None, None

    def __del__(self):
        if self.__camera is not None:
            self.__camera.close()
