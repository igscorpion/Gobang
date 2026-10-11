"""FastAPI 展示层：人机对战，AI 对手通过 rl/agents/registry.py 按名称切换。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.agents.registry import available_agents, create_agent
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

# 默认 AI 可用环境变量 GOMOKU_AI 覆盖，例如 GOMOKU_AI=random
DEFAULT_AGENT = os.environ.get("GOMOKU_AI", "greedy")

env = GomokuEnv()
agent = create_agent(DEFAULT_AGENT)
current_agent = DEFAULT_AGENT


class MoveRequest(BaseModel):
    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)


class AgentRequest(BaseModel):
    name: str


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
        "agent": current_agent,
    }


def _ai_move() -> dict | None:
    if env.done:
        return None
    state = env.get_state()
    legal_actions = env.get_legal_actions()
    if agent.needs_last_move:
        # AlphaZero 需要「对手上一步」；env.last_move 为 (row, col)
        last = None
        if env.last_move is not None:
            last = encode_action(env.last_move[0], env.last_move[1], env.board_size)
        state = (state, last)
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


@app.get("/game/agents")
def list_agents():
    return {"agents": available_agents(), "current": current_agent}


@app.post("/game/agent")
def set_agent(body: AgentRequest):
    global agent, current_agent
    try:
        agent = create_agent(body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    current_agent = body.name
    env.reset()
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
    """玩家（黑棋）落子：只落这一子并立即返回，AI 由前端随后单独请求。"""
    if env.done:
        raise HTTPException(status_code=400, detail="对局已经结束，请重新开始")

    if env.current_player != env.BLACK:
        raise HTTPException(status_code=400, detail="当前不是黑棋（玩家）回合")

    try:
        env.step(encode_action(body.row, body.col, env.board_size))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _snapshot()


@app.post("/game/ai_move")
def ai_move():
    """AI（白棋）思考并落子。

    与玩家落子分开，前端可先渲染玩家的棋子，再等待 AI，避免棋盘迟迟不更新的卡顿感。
    """
    if env.done:
        return _snapshot()

    if env.current_player != env.WHITE:
        raise HTTPException(status_code=400, detail="当前不是白棋（AI）回合")

    return _snapshot(ai_move=_ai_move())
