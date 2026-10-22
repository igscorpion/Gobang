"""Isolated HTTP service; never modifies a user's running game."""

import socket, subprocess, sys, time, json, urllib.request, urllib.error
from pathlib import Path
import pytest


@pytest.fixture(scope="module")
def api():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "web.backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    def request(path, body=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}" + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    for _ in range(100):
        try:
            request("/")
            break
        except OSError:
            if p.poll() is not None:
                raise RuntimeError("Test server failed to start")
            time.sleep(0.1)
    else:
        p.terminate()
        raise RuntimeError("Test server startup timed out")
    yield request
    p.terminate()
    p.wait(timeout=10)


def test_rule_switch_and_separate_moves(api):
    state = api("/game/rules", {"rules": "renju"})
    assert state["rules"] == "renju"
    state = api("/game/move", {"row": 4, "col": 4})
    assert state["current_player"] == "white"
    state = api("/game/ai_move", {})
    assert state["current_player"] == "black"
    assert sum(v != 0 for row in state["board"] for v in row) == 2
    state = api("/game/rules", {"rules": "freestyle"})
    assert all(v == 0 for row in state["board"] for v in row)


def test_invalid_rule_and_model_are_atomic(api):
    api("/game/move", {"row": 4, "col": 4})
    before = api("/game")
    with pytest.raises(urllib.error.HTTPError) as e:
        api("/game/rules", {"rules": "invalid"})
    assert e.value.code == 422
    with pytest.raises(urllib.error.HTTPError) as e:
        api("/game/agent", {"name": "invalid"})
    assert e.value.code == 400
    assert api("/game") == before


def test_out_of_range_column_does_not_wrap_to_next_row(api):
    api("/game/reset", {})
    before = api("/game")
    with pytest.raises(urllib.error.HTTPError) as e:
        api("/game/move", {"row": 0, "col": 9})
    assert e.value.code == 400 and api("/game") == before


def test_tactical_a2c_is_selectable_and_plays_after_rule_switch(api):
    if "a2c4_tactical" not in api("/game/agents")["agents"]:
        pytest.skip("A validation-selected A2C checkpoint is not installed")
    api("/game/rules", {"rules": "renju"})
    state = api("/game/agent", {"name": "a2c4_tactical"})
    assert state["agent"] == "a2c4_tactical" and state["rules"] == "renju"
    api("/game/move", {"row": 4, "col": 4})
    state = api("/game/ai_move", {})
    assert state["current_player"] == "black"
    assert sum(v != 0 for row in state["board"] for v in row) == 2
    api("/game/agent", {"name": "greedy"})
    api("/game/rules", {"rules": "freestyle"})
