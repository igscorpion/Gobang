"""Compatibility entry point for the consolidated Double DQN trainer."""

from rl.training.local_dqn import parser, run

if __name__ == "__main__":
    run(parser().parse_args())
