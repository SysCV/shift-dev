import logging
from time import perf_counter


class LoggerSingleton:
    __instance = None

    @staticmethod
    def get_logger():
        if LoggerSingleton.__instance is None:
            LoggerSingleton()
        return LoggerSingleton.__instance.logger

    def __init__(self):
        if LoggerSingleton.__instance is not None:
            raise Exception(
                "LoggerSingleton is a singleton class, use get_logger() instead"
            )
        else:
            self.logger = logging.getLogger("shift_dev_logger")
            self.logger.setLevel(logging.DEBUG)
            log_formatter = logging.Formatter(
                "[%(asctime)s] SHIFT DevKit - %(levelname)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
            )
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(log_formatter)
            self.logger.addHandler(ch)
            self.logger.propagate = False
            
            LoggerSingleton.__instance = self


def setup_logger():
    logger = LoggerSingleton.get_logger()
    return logger


class Timer:  # pragma: no cover
    """Timer class based on perf_counter."""

    def __init__(self) -> None:
        """Creates an instance of the class."""
        self._tic = perf_counter()
        self._toc = None
        self.paused = False

    def reset(self) -> None:
        """Reset timer."""
        self._tic = perf_counter()
        self._toc = None
        self.paused = False

    def pause(self) -> None:
        """Pause function."""
        if self.paused:
            raise ValueError("Timer already paused!")
        self._toc = perf_counter()
        self.paused = True

    def resume(self) -> None:
        """Resume function."""
        if self.paused:
            raise ValueError("Timer is not paused!")
        assert self._toc is not None
        self._tic = perf_counter() - (self._toc - self._tic)
        self._toc = None
        self.paused = False

    def time(self, milliseconds: bool = False) -> float:
        """Return elapsed time."""
        if not self.paused:
            self._toc = perf_counter()
        assert self._toc is not None
        time_elapsed = self._toc - self._tic
        if milliseconds:
            return time_elapsed * 1000
        return time_elapsed
