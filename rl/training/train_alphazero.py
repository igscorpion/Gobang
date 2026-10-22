"""Compatibility entry point; training options now use explicit flags."""

from rl.training.az_games import selfplay_game, vs_opponent_game, evaluate
from rl.training.local_alphazero import parser, run

if __name__ == "__main__":
    run(parser().parse_args())
