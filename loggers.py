import  logging
import sys


class ColoredFormatter(logging.Formatter):
    """Форматтер с цветами для консоли."""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',  # Reset
    }

    def format(self, record):
        # Сохраняем оригинальный форматтер
        original_format = self._style._fmt
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])

        # Временно добавляем цвет к формату
        self._style._fmt = f"{color}{original_format}{self.COLORS['RESET']}"
        result = super().format(record)

        # Возвращаем оригинальный формат
        self._style._fmt = original_format
        return result


def setup_logging(log_file: str = "pipeline_critical.log"):
    """
    Консоль: цветная, уровень INFO+
    Файл: без цветов, только ERROR/CRITICAL
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Форматы
    console_fmt_str = "%(asctime)s [%(processName)s] %(levelname)s: %(message)s"
    file_fmt_str = (
        "%(asctime)s [%(processName)s] %(levelname)s\n"
        "  → %(message)s\n"
        "  → File: %(filename)s:%(lineno)d, Func: %(funcName)s\n"
    )

    # 1. Консоль: цветная, INFO+
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColoredFormatter(console_fmt_str, datefmt="%Y-%m-%d %H:%M:%S"))

    # 2. Файл: без цветов, только ERROR+
    fh = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter(file_fmt_str, datefmt="%Y-%m-%d %H:%M:%S"))

    if not root.handlers:
        root.addHandler(ch)
        root.addHandler(fh)