# You need to complete this function which should return action,

from game.go import Board
import random
from game.go import opponent_color
from game.util import PointDict
from copy import deepcopy
import time

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
        actions = board.legal_actions
        if actions:
            return random.choice(actions)
        return None

class GroupShadow:
    """Store only the mutable fields of a Group (cheap)."""
    __slots__ = ("group", "points", "liberties")
    def __init__(self, group):
        self.group = group
        # copies of the mutable fields (primitive containers)
        self.points = list(group.points)
        # liberties is a set in your Group; store a shallow copy
        self.liberties = set(group.liberties)


class BoardSnapshot:
    """
    Snapshot that keeps references to the ORIGINAL Group objects
    but stores copies of their mutable internals so the snapshot
    is delinked from future mutations.
    """
    def __init__(self, board: Board, shadow_pool: dict):
        # cheap scalars
        self.next = board.next
        self.winner = board.winner
        self.end_by_no_legal_actions = board.end_by_no_legal_actions
        self.counter_move = board.counter_move
        self.legal_actions = board.legal_actions[:] if board.legal_actions is not None else []

        # keep references to the original Group objects (no deepcopy)
        self.groups_black = board.groups['BLACK'][:]
        self.groups_white = board.groups['WHITE'][:]
        self.endangered = board.endangered_groups[:]
        self.removed = board.removed_groups[:]

        self.group_shadows = {}
        # per-group saved internals
        for g in (self.groups_black + self.groups_white):
            if g in shadow_pool:
                shadow = shadow_pool[g]
                # update the shadow with current values instead of creating new
                shadow.points[:] = list(g.points)
                shadow.liberties.clear()
                shadow.liberties.update(list(g.liberties))
            else:
                shadow = GroupShadow(g)
                if shadow_pool is not None:
                    shadow_pool[g] = shadow
            self.group_shadows[g] = shadow
        # snapshot point-dicts: mapping point -> list(of original Group refs)
        self.stonedict = {'BLACK': {}, 'WHITE': {}}
        self.libertydict = {'BLACK': {}, 'WHITE': {}}

        for color in ('BLACK', 'WHITE'):
            for p, groups in board.stonedict.get_items(color):
                if groups:
                    # store a shallow copy of the list of group refs (original objects)
                    self.stonedict[color][p] = groups[:]
            for p, groups in board.libertydict.get_items(color):
                if groups:
                    self.libertydict[color][p] = groups[:]


def restore(board: Board, snap: BoardSnapshot):
    """Restore board to snapshot state by mutating existing objects in-place."""
    board.next = snap.next
    board.winner = snap.winner
    board.end_by_no_legal_actions = snap.end_by_no_legal_actions
    board.counter_move = snap.counter_move
    board.legal_actions = snap.legal_actions[:]

    board.groups['BLACK'][:] = snap.groups_black
    board.groups['WHITE'][:] = snap.groups_white
    board.endangered_groups[:] = snap.endangered
    board.removed_groups[:] = snap.removed

    for group, shadow in snap.group_shadows.items():
        group.points[:] = shadow.points
        try:
            group.liberties.clear()
            group.liberties.update(shadow.liberties)
        except AttributeError:
            group.liberties = set(shadow.liberties)

    for color in ('BLACK', 'WHITE'):
        board.stonedict.d[color].clear()
        for p, groups in snap.stonedict[color].items():
            board.stonedict.d[color][p] = groups[:]

        board.libertydict.d[color].clear()
        for p, groups in snap.libertydict[color].items():
            board.libertydict.d[color][p] = groups[:]

class Agent1v1:
    """A class to generate a random action for a Go board."""
    def __init__(self, color, verbose = False):
        self.color = color
        self.opponent_color = opponent_color(self.color)
        self.MAX_DEPTH = 3

        self.W_LIBERTIES = 10
        self.W_ATARI = 40
        self.W_WIN = 1e5
        self.verbose = verbose

        self._snap_stack = [BoardSnapshot.__new__(BoardSnapshot)
                            for _ in range(self.MAX_DEPTH + 2)]
        
        self._shadow_pools = [{} for _ in range(self.MAX_DEPTH + 2)]

    def evaluate(self, board: Board) -> int:
        if(board.winner is not None):
            if(board.winner == self.color): return self.W_WIN
            else: return -self.W_WIN
        score = 0
        score += self.W_LIBERTIES * (len(board.libertydict.d[self.color]) - len(board.libertydict.d[self.opponent_color]))
        return score

    def minimax(self, board: Board, depth: int, maximizing_player: bool) -> int:  
        pos = 0   
        if((depth == 0) or (board.winner is not None)):
            return self.evaluate(board),1

        if(len(board.legal_actions) == 0): 
            return 0,1

        snap = self._snap_stack[depth]
        shadow_pool = self._shadow_pools[depth]
        snap.__init__(board, shadow_pool)

        if(maximizing_player):
            max_score = -1e9
            for action in board.legal_actions:
                board.put_stone(action, check_legal=False)
                # successor = board.copy()
                # successor.put_stone(action, check_legal=False)
                # score, new_pos = self.minimax(successor, depth-1, False)
                score, new_pos = self.minimax(board, depth-1, False)
                max_score = max(max_score, score)
                pos += new_pos
                restore(board, snap)
            return max_score, pos
        
        else:
            min_score = 1e9
            for action in board.legal_actions:
                board.put_stone(action, check_legal=False)
                # successor = board.copy()
                # successor.put_stone(action, check_legal=False)
                # score, new_pos = self.minimax(successor, depth-1, True)
                score, new_pos = self.minimax(board, depth-1, True)
                min_score = min(min_score, score)
                pos += new_pos
                restore(board, snap)
            return min_score, pos

    def get_best_action(self, board: Board, depth: int):
        pos = 0
        best_score = -1e9
        best_action = None
        # print(f"Legal Actions: {board.legal_actions}")
        # self.print_point_dict(board.libertydict)
        snap = self._snap_stack[-1]
        snap = self._snap_stack[-1]
        snap.__init__(board, self._shadow_pools[-1]) 

        for action in board.legal_actions:
            
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
        if(self.verbose):
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
        start_time = time.time()
        actions = board.legal_actions
        if actions:
            best_action = self.get_best_action(board, self.MAX_DEPTH)
            # print("BEST ACTION: ", best_action, board.next)
            if(self.verbose):
                print(f"Decision Time: {time.time() - start_time}")
            return best_action
        return None