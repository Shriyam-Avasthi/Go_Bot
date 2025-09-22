# You need to complete this function which should return action,

from game.go import Board
import random
from game.go import opponent_color
from game.util import PointDict
from copy import deepcopy

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
    
class BoardSnapshot:
    def __init__(self, board: Board):
        # simple scalars
        self.next = board.next
        self.winner = board.winner
        self.end_by_no_legal_actions = board.end_by_no_legal_actions
        self.counter_move = board.counter_move
        self.legal_actions = board.get_legal_actions()

        group_mapping = {group: deepcopy(group) for group in board.groups['BLACK'] + board.groups['WHITE']}

        # groups & lists – just keep references; they’re immutable for our backtracking step
        self.groups = {
            'BLACK': [group_mapping[group] for group in board.groups['BLACK']],
            'WHITE': [group_mapping[group] for group in board.groups['WHITE']]
        }
        self.endangered = [group_mapping[group] for group in board.endangered_groups]
        self.removed = [group_mapping[group] for group in board.removed_groups]

        # PointDict shallow snapshot: dict of dict of lists
        self.stonedict = {
            'BLACK': {p: list(group_mapping[group] for group in groups) for p, groups in board.stonedict.d['BLACK'].items()},
            'WHITE': {p: list(group_mapping[group] for group in groups) for p, groups in board.stonedict.d['WHITE'].items()}
        }
        self.libertydict = {
            'BLACK': {p: list(group_mapping[group] for group in groups) for p, groups in board.libertydict.d['BLACK'].items()},
            'WHITE': {p: list(group_mapping[group] for group in groups) for p, groups in board.libertydict.d['WHITE'].items()}
        }

def restore(board: Board, snap: BoardSnapshot):
    board.next = snap.next
    board.winner = snap.winner
    board.end_by_no_legal_actions = snap.end_by_no_legal_actions
    board.counter_move = snap.counter_move
    board.groups['BLACK'] = list(snap.groups['BLACK'])
    board.groups['WHITE'] = list(snap.groups['WHITE'])
    board.endangered_groups = list(snap.endangered)
    board.removed_groups = list(snap.removed)
    board.legal_actions = snap.legal_actions

    # restore PointDicts
    board.stonedict.d['BLACK'] = {p: list(groups) for p, groups in snap.stonedict['BLACK'].items()}
    board.stonedict.d['WHITE'] = {p: list(groups) for p, groups in snap.stonedict['WHITE'].items()}
    board.libertydict.d['BLACK'] = {p: list(groups) for p, groups in snap.libertydict['BLACK'].items()}
    board.libertydict.d['WHITE'] = {p: list(groups) for p, groups in snap.libertydict['WHITE'].items()}

class Agent1v1:
    """A class to generate a random action for a Go board."""
    def __init__(self, color):
        self.color = color
        self.MAX_DEPTH = 3

        self.W_LIBERTIES = 10
        self.W_ATARI = 40
        self.W_WIN = 1e5

    def evaluate(self, board: Board) -> int:
        if(board.winner is not None):
            if(board.winner == self.color): return self.W_WIN
            else: return -self.W_WIN
        score = 0
        score += self.W_LIBERTIES * (len(board.libertydict.d[self.color]) - len(board.libertydict.d[opponent_color(self.color)]))
        return score

    def minimax(self, board: Board, depth: int, maximizing_player: bool) -> int:  
        pos = 0   
        if((depth == 0) or (board.winner is not None)):
            return self.evaluate(board),1

        if(len(board.get_legal_actions()) == 0): 
            return 0,1

        if(maximizing_player):
            max_score = -1e9
            for action in board.get_legal_actions():
                # snap = BoardSnapshot(board)
                # board.put_stone(action, check_legal=False)
                successor = board.copy()
                successor.put_stone(action, check_legal=False)
                score, new_pos = self.minimax(successor, depth-1, False)
                # score, new_pos = self.minimax(board, depth-1, False)
                max_score = max(max_score, score)
                pos += new_pos
                # restore(board, snap)
            return max_score, pos
        
        else:
            min_score = 1e9
            for action in board.get_legal_actions():
                # snap = BoardSnapshot(board)
                # board.put_stone(action, check_legal=False)
                successor = board.copy()
                successor.put_stone(action, check_legal=False)
                score, new_pos = self.minimax(successor, depth-1, True)
                # score, new_pos = self.minimax(board, depth-1, True)
                min_score = min(min_score, score)
                pos += new_pos
                # restore(board, snap)
            return min_score, pos

    def get_best_action(self, board: Board, depth: int):
        pos = 0
        best_score = -1e9
        best_action = None
        # self.print_point_dict(board.libertydict)
        # print(f"Legal Actions: {board.get_legal_actions()}")
        for action in board.get_legal_actions():
            snap = BoardSnapshot(board)
            
            # print("STORED_______________________")
            # for key,val in snap.libertydict['BLACK'].items(): 
            #     print(key , val)
        
            # for key,val in snap.libertydict['WHITE'].items(): 
            #     print(key , val)
            # print()

            board.put_stone(action, check_legal=False)
            # successor = board.copy()
            # successor.put_stone(action, check_legal=False)
            # score, new_pos = self.minimax(successor, depth, maximizing_player=False)
            # self.print_point_dict(board.libertydict)
            score, new_pos = self.minimax(board, depth, maximizing_player=False)
            pos += new_pos
            if(score > best_score):
                best_score = score
                best_action = action
            
            restore(board, snap)
            # print("RESTORED: ")
            # self.print_point_dict(board.libertydict)

        print(f"Possibilities considered: {pos}")
        return best_action
            
    def print_point_dict(self, point_dict:PointDict):
        for key,val in point_dict.d['BLACK'].items(): 
            print(key , val)
        
        for key,val in point_dict.d['WHITE'].items(): 
            print(key , val)
        print()

    def get_action(self, board: Board):
        """
        Returns a random legal action from the board.

        :param board: The current Go board state.
        :return: A random legal action (tuple) or None if no actions are available.
        """

        actions = board.get_legal_actions()
        if actions:
            best_action = self.get_best_action(board, self.MAX_DEPTH)
            # print("BEST ACTION: ", best_action, board.next)
            return best_action
        return None