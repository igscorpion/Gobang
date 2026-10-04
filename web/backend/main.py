"""FastAPI 展示层：人机对战，默认使用 RandomAgent。"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.agents.random_agent import RandomAgent
from rl.environment.gomoku import GomokuEnv, decode_action, encode_action

app = FastAPI(title="RL Gomoku", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = GomokuEnv()
agent = RandomAgent()


class MoveRequest(BaseModel):
    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)


def _snapshot(ai_move: dict | None = None) -> dict:
    winner = None
    if env.winner == env.BLACK:
        winner = "black"
    elif env.winner == env.WHITE:
        winner = "white"
    elif env.done and env.winner == env.EMPTY:
        winner = "draw"

    last_move = None
    if env.last_move is not None:
        last_move = {"row": env.last_move[0], "col": env.last_move[1]}

    return {
        "board": env.get_board(),
        "board_size": env.board_size,
        "current_player": "black" if env.current_player == env.BLACK else "white",
        "ai_move": ai_move,
        "last_move": last_move,
        "winner": winner,
        "game_over": env.done,
    }


def _ai_move() -> dict | None:
    if env.done:
        return None
    state = env.get_state()
    legal_actions = env.get_legal_actions()
    action = agent.select_action(state, legal_actions)
    env.step(action)
    row, col = decode_action(action, env.board_size)
    return {"row": row, "col": col}


@app.get("/")
def root():
    return {
        "name": "RL Gomoku API",
        "docs": "/docs",
        "game": "/game",
    }


@app.get("/game")
def get_game():
    return _snapshot()


@app.post("/game/start")
def start_game():
    env.reset()
    return _snapshot()


@app.post("/game/reset")
def reset_game():
    env.reset()
    return _snapshot()


@app.post("/game/move")
def player_move(body: MoveRequest):
    if env.done:
        raise HTTPException(status_code=400, detail="对局已经结束，请重新开始")

    if env.current_player != env.BLACK:
        raise HTTPException(status_code=400, detail="当前不是黑棋（玩家）回合")

    try:
        env.step(encode_action(body.row, body.col, env.board_size))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ai_move = _ai_move()
    return _snapshot(ai_move=ai_move)
