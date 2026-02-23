from __future__ import annotations

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"


def _wrap(code: str, text: object) -> str:
    return f"{code}{text}{RESET}"


def green(text: object) -> str:
    return _wrap("\x1b[32m", text)


def yellow(text: object) -> str:
    return _wrap("\x1b[33m", text)


def red(text: object) -> str:
    return _wrap("\x1b[31m", text)


def cyan(text: object) -> str:
    return _wrap("\x1b[36m", text)


def blue(text: object) -> str:
    return _wrap("\x1b[34m", text)


def magenta(text: object) -> str:
    return _wrap("\x1b[35m", text)


def white(text: object) -> str:
    return _wrap("\x1b[37m", text)


def bright_white(text: object) -> str:
    return f"\x1b[1;37m{text}{RESET}"


def bold(text: object) -> str:
    return f"{BOLD}{text}{RESET}"


def dim(text: object) -> str:
    return f"{DIM}{text}{RESET}"


def separator(char: str = "-", length: int = 64, color_func=None) -> str:
    line = char * length
    return color_func(line) if color_func else dim(line)


def section_header(title: str, emoji: str = "") -> str:
    icon = f"{emoji} " if emoji else ""
    return f"\n{separator()}\n{bold(cyan(icon + title))}\n{separator()}"
