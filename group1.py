# You need to complete this function which should return action,

from game.go import Board
import random


class Agent1:
    """A class to generate a random action for a Go board."""

    def __init__(self, color):
        self.color = color

    def get_action(self, board: Board):
        """
        Returns a random legal action from the board.

        :param board: The current Go board state.
        :return: A random legal action (tuple) or None if no actions are available.
        """
        actions = board.get_legal_actions()
        if actions:
            return random.choice(actions)
        return None