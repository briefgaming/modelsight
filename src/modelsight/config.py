"""Load and check an experiment YAML file.

Engine and benchmark flags are not validated. `server` and `bench` are command
strings, copied through as the user wrote them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TARGETS = ("local", "modal")
DEFAULT_READY_URL = "http://localhost:30000/health"
DEFAULT_READY_TIMEOUT = "20m"
DEFAULT_TIMEOUT = "2h"
DEFAULT_OUTPUT = "results/"

_KNOWN_FIELDS = {
    "target",
    "gpu",
    "image",
    "server",
    "bench",
    "ready_url",
    "ready_timeout",
    "timeout",
    "output",
}
_DURATION = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")


class ConfigError(Exception):
    """The experiment file is missing something or has a bad value."""


@dataclass(frozen=True)
class Experiment:
    target: str
    server: str
    bench: str
    gpu: str | None
    image: str | None
    ready_url: str
    ready_timeout: str
    timeout: str
    output: str

    @property
    def ready_timeout_seconds(self) -> float:
        return parse_duration(self.ready_timeout)

    @property
    def timeout_seconds(self) -> float:
        return parse_duration(self.timeout)


def parse_duration(value: str) -> float:
    """Parse `90`, `90s`, `20m`, `2h`, or a combination like `1h30m` into seconds."""
    text = value.strip()
    if text.isdigit():
        seconds = int(text)
    else:
        match = _DURATION.fullmatch(text)
        if match is None or not any(match.groups()):
            raise ConfigError(
                f"invalid duration {value!r}; use seconds, or a value like 90s, 20m, 2h"
            )
        hours, minutes, secs = (int(part or 0) for part in match.groups())
        seconds = hours * 3600 + minutes * 60 + secs
    if seconds <= 0:
        raise ConfigError(f"duration must be greater than zero, got {value!r}")
    return float(seconds)


def load_experiment(path: Path) -> tuple[Experiment, str]:
    """Return the experiment and the exact file text, so a run can save it unchanged."""
    import yaml

    try:
        text = path.read_text()
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping of experiment fields")
    return experiment_from_mapping(data), text


def experiment_from_mapping(data: dict) -> Experiment:
    unknown = sorted(set(data) - _KNOWN_FIELDS)
    if unknown:
        raise ConfigError(f"unknown field {unknown[0]!r}")

    target = _require_string(data, "target")
    if target not in TARGETS:
        raise ConfigError(f"target must be one of: {', '.join(TARGETS)}")

    gpu = _optional_string(data, "gpu")
    image = _optional_string(data, "image")
    if target == "modal":
        if not gpu:
            raise ConfigError("gpu is required when target is modal, for example B200")
        if not image:
            raise ConfigError("image is required when target is modal")

    ready_url = _optional_string(data, "ready_url") or DEFAULT_READY_URL
    if not ready_url.startswith(("http://", "https://")):
        raise ConfigError("ready_url must start with http:// or https://")
    ready_timeout = _optional_string(data, "ready_timeout") or DEFAULT_READY_TIMEOUT
    timeout = _optional_string(data, "timeout") or DEFAULT_TIMEOUT
    parse_duration(ready_timeout)
    parse_duration(timeout)

    return Experiment(
        target=target,
        server=_require_string(data, "server"),
        bench=_require_string(data, "bench"),
        gpu=gpu,
        image=image,
        ready_url=ready_url,
        ready_timeout=ready_timeout,
        timeout=timeout,
        output=_optional_string(data, "output") or DEFAULT_OUTPUT,
    )


def _require_string(data: dict, field: str) -> str:
    value = _optional_string(data, field)
    if not value:
        raise ConfigError(f"{field} is required and must be a non-empty string")
    return value


def _optional_string(data: dict, field: str) -> str | None:
    if field not in data or data[field] is None:
        return None
    value = data[field]
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a string")
    value = value.strip()
    return value or None
