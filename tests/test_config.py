from pathlib import Path

import pytest

from modelsight.config import ConfigError, load_experiment, parse_duration

ROOT = Path(__file__).resolve().parents[1]


def test_parse_duration_units() -> None:
    assert parse_duration("90") == 90
    assert parse_duration("90s") == 90
    assert parse_duration("20m") == 1200
    assert parse_duration("2h") == 7200
    assert parse_duration("1h30m") == 5400


def test_parse_duration_rejects_junk() -> None:
    with pytest.raises(ConfigError):
        parse_duration("soon")
    with pytest.raises(ConfigError):
        parse_duration("0s")


def test_example_yaml_loads() -> None:
    experiment, text = load_experiment(ROOT / "examples" / "qwen27b-fp8.yaml")
    assert experiment.target == "modal"
    assert experiment.gpu == "B200"
    assert "sglang.launch_server" in experiment.server
    assert "--ui none" in experiment.bench
    assert "target: modal" in text


def test_local_yaml_fills_defaults(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text("target: local\nserver: sleep 1\nbench: echo ok\n")
    experiment, _ = load_experiment(path)
    assert experiment.ready_url == "http://localhost:30000/health"
    assert experiment.ready_timeout == "20m"
    assert experiment.timeout == "2h"
    assert experiment.output == "results/"
    assert experiment.gpu is None


def test_modal_requires_gpu_and_image(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text("target: modal\nserver: true\nbench: true\n")
    with pytest.raises(ConfigError, match="gpu"):
        load_experiment(path)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text("target: local\nserver: true\nbench: true\nsweep: []\n")
    with pytest.raises(ConfigError, match="unknown field"):
        load_experiment(path)


def test_bad_ready_url_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text(
        "target: local\nserver: true\nbench: true\nready_url: localhost:30000/health\n"
    )
    with pytest.raises(ConfigError, match="ready_url"):
        load_experiment(path)
