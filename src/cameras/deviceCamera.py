import cv2
from . import camera


class DeviceCamera(camera.Camera):
    def __init__(self, args: any, width: int, height: int):
        desired_width = width
        desired_height = height
        self.__args = args
        self.__cap = self.__open()
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        while self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH) < desired_width or \
                self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT) < desired_height:
            width *= 2
            height *= 2
            self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.__cap.set(cv2.CAP_PROP_FPS, args.fps)

    def __open(self):
        if self.__args.device == -1:
            return cv2.VideoCapture(0)
        return cv2.VideoCapture("/dev/video{}".format(self.__args.device), cv2.CAP_V4L)

    def is_opened(self) -> bool:
        return self.__cap.isOpened()

    def release(self):
        self.__cap.release()

    def capture(self) -> (bool, bytearray):
        if self.is_opened():
            return self.__cap.read()
        return False, None

    def set_gain(self, gain: int):
        if self.is_opened():
            self.__cap.set(cv2.CAP_PROP_GAIN, gain)
