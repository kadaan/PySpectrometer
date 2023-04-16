try:
    from libcamera import controls
except ImportError as e:
    raise e
try:
    from picamera2 import Picamera2
except ImportError as e:
    raise e
from . import camera
from pprint import pprint


class PiCamera(camera.Camera):
    def __init__(self, width: int, height: int, gain: int):
        self.__camera = Picamera2()
        # need to spend more time at: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
        # but this will do for now!
        # min and max microseconds per frame gives framerate.
        # 30fps (33333, 33333)
        # 25fps (40000, 40000)

        video_config = self.__camera.create_video_configuration(
            main={
                "format": 'RGB888',
                "size": (width, height)
            },
            controls={
                "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Fast,
                "FrameDurationLimits": (33333, 33333),
                "AnalogueGain": float(gain)
            })
        pprint(video_config["controls"])
        self.__camera.configure(video_config)
        self.__camera.start()

    def is_opened(self) -> bool:
        return self.__camera.started

    def release(self):
        self.__camera.close()

    def capture(self) -> (bool, bytearray):
        if self.is_opened():
            return True, self.__camera.capture_array()
        return False, None

    def set_gain(self, gain: int):
        if self.is_opened():
            self.__camera.set_controls({"AnalogueGain": gain})
