from abc import ABC, abstractmethod


class Camera(ABC):
    @abstractmethod
    def is_opened(self) -> bool:
        pass

    def close(self):
        if self.is_opened():
            self.release()

    @abstractmethod
    def release(self):
        pass

    @abstractmethod
    def capture(self) -> (bool, bytearray):
        pass

    @abstractmethod
    def set_gain(self, gain: int):
        pass
