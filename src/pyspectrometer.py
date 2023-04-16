import tkinter
import PIL.Image
import PIL.ImageTk
import argparse
import constants
import capturer
from widgets.numericSpinbox import NumericSpinbox


# https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
class App:
    def __init__(self, args, window, window_title):
        self.__window = window
        self.__window.geometry("{}x{}".format(constants.WIDTH, constants.HEIGHT))
        self.__window.resizable(width=False, height=False)
        self.__window.title(window_title)
        self.__video_image_container = None
        self.__video_image = None
        self.__graph_image_container = None
        self.__graph_image = None

        self.__capturer = capturer.Capturer(args)

        # Create frames
        self.__top_frame = tkinter.Frame(self.__window, width=constants.WIDTH, height=constants.VIDEO_HEIGHT)
        self.__top_frame.pack()

        self.__middle_frame = tkinter.Frame(self.__window, width=constants.WIDTH, height=constants.SPECTRUM_HEIGHT)
        self.__middle_frame.pack()

        self.__bottom_frame = tkinter.Frame(self.__window, width=constants.WIDTH, height=constants.CONTROL_HEIGHT)
        self.__bottom_frame.pack()
        
        self.__control_frame = tkinter.Frame(self.__bottom_frame, width=constants.WIDTH)
        self.__control_frame.place(relx=.5, rely=.5, anchor="center")

        self.__video_canvas = tkinter.Canvas(self.__top_frame, width=constants.WIDTH, height=constants.VIDEO_HEIGHT,
                                             borderwidth=0, highlightthickness=0)
        self.__video_canvas.grid(row=0, column=0, padx=(0, 0))

        self.__spectrum_canvas = tkinter.Canvas(self.__middle_frame, width=constants.WIDTH, height=constants.SPECTRUM_HEIGHT,
                                                borderwidth=0, highlightthickness=0, cursor="tcross")
        self.__spectrum_canvas.grid(row=0, column=0, padx=0, pady=0, columnspan=8)

        self.__peak_width = NumericSpinbox(self.__control_frame, "Peak\nWidth", 0, 100, self.__capturer.peak_width,
                                           self.__update_peak_width)
        self.__peak_width.pack(padx=5, pady=0, side="left")

        self.__peak_threshold = NumericSpinbox(self.__control_frame, "Peak\nThreshold", 0, 100,
                                               self.__capturer.peak_threshold, self.__update_peak_threshold)
        self.__peak_threshold.pack(padx=5, pady=0, side="left")

        self.__savitzky_golay_filter = NumericSpinbox(self.__control_frame, "Smoothing", 0, 15,
                                                      self.__capturer.savitzky_golay_filter,
                                                      self.__update_savitzky_golay_filter)
        self.__savitzky_golay_filter.pack(padx=5, pady=0, side="left")

        self.__gain = NumericSpinbox(self.__control_frame, "Gain", 0, 50, self.__capturer.gain, self.__update_gain)
        self.__gain.pack(padx=5, pady=0, side="left")

        self.__peak_hold_button = tkinter.Button(self.__control_frame, text="Peak Hold", width=7, height=3, fg="black",
                                                 bg="yellow", activebackground='yellow',
                                                 command=self.__update_peak_hold)
        self.__peak_hold_button.pack(padx=5, pady=0, side="left")

        self.__pause_button = tkinter.Button(self.__control_frame, text="Pause", width=7, height=3, fg="black",
                                             bg="yellow", activebackground='yellow', command=self.__update_pause)
        self.__pause_button.pack(padx=5, pady=0, side="left")

        self.__update()

        self.__window.mainloop()

    def __update_peak_width(self, event):
        setattr(self.__capturer, 'peak_width', event)

    def __update_peak_threshold(self, event):
        setattr(self.__capturer, 'peak_threshold', event)

    def __update_savitzky_golay_filter(self, event):
        setattr(self.__capturer, 'savitzky_golay_filter', event)

    def __update_gain(self, event):
        setattr(self.__capturer, 'gain', event)

    def __update_peak_hold(self):
        if self.__peak_hold_button.cget("bg") == 'yellow':
            self.__peak_hold_button.configure(fg="yellow", bg="red", activebackground='red', activeforeground="yellow")
            setattr(self.__capturer, 'hold_peaks', True)
            self.__savitzky_golay_filter.disabled()
        else:
            self.__peak_hold_button.configure(fg="black", bg="yellow", activebackground='yellow',
                                              activeforeground="black")
            setattr(self.__capturer, 'hold_peaks', False)
            self.__savitzky_golay_filter.enabled()

    def __update_pause(self):
        if self.__pause_button.cget("bg") == 'yellow':
            self.__pause_button.configure(fg="yellow", bg="red", activebackground='red', activeforeground="yellow")
            setattr(self.__capturer, 'pause', True)
            self.__peak_width.disabled()
            self.__peak_threshold.disabled()
            self.__savitzky_golay_filter.disabled()
            self.__gain.disabled()
            self.__peak_hold_button.configure(state="disabled")
        else:
            self.__pause_button.configure(fg="black", bg="yellow", activebackground='yellow', activeforeground="black")
            setattr(self.__capturer, 'pause', False)
            self.__peak_width.enabled()
            self.__peak_threshold.enabled()
            self.__savitzky_golay_filter.enabled()
            self.__gain.enabled()
            self.__peak_hold_button.configure(state="active")

    def __update(self):
        ret, image, graph = self.__capturer.get_graph()
        if ret:
            self.__video_image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
            self.__graph_image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(graph))
            if self.__video_image_container is None:
                self.__video_image_container = self.__video_canvas.create_image(0, 0, image=self.__video_image,
                                                                                anchor=tkinter.NW)
            else:
                self.__video_canvas.itemconfig(self.__video_image_container, image=self.__video_image)

            if self.__graph_image_container is None:
                self.__graph_image_container = self.__spectrum_canvas.create_image(0, 0, image=self.__graph_image,
                                                                                   anchor=tkinter.NW)
            else:
                self.__spectrum_canvas.itemconfig(self.__graph_image_container, image=self.__graph_image)

        self.__window.after(constants.UPDATE_DELAY, self.__update)


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='PySpectrometer')
    subparsers = parser.add_subparsers(help='Source to capture images', dest='source')

    parser_device = subparsers.add_parser(constants.COMMAND_DEVICE_CAMERA, help='device help')
    parser_device.add_argument("--device", type=int, default=-1,
                               help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
    parser_device.add_argument("--fps", type=int, default=30, help="Frame Rate e.g. 30")

    subparsers.add_parser(constants.COMMAND_PI_CAMERA, help='Use the picamera')

    return parser


if __name__ == '__main__':
    App(arg_parser().parse_args(), tkinter.Tk(), "PySpectrometer")
