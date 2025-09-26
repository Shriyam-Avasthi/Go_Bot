# You need to complete this function which should return action,

from game.go import Board
import random
from game.go import opponent_color, BOARD_SIZE
from game.util import PointDict
from copy import deepcopy
import time
import numpy as np

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
        actions = board.legal_actions
        if actions:
            return random.choice(actions)
        return None
    
#########################################################################################################################
def make_zobrist(board_size: int, seed: int = 0) -> list[list[list[int]]]:
    """
    Returns a table [x][y][color] -> random 64-bit int.
    color: 0 = BLACK, 1 = WHITE
    """
    rnd = random.Random(seed)
    return [[[rnd.getrandbits(64) for _ in range(2)]
             for _ in range(board_size)]
             for _ in range(board_size)]

def board_hash(board, zobrist) -> int:
    h = 0
    for p in board.stonedict.d['BLACK']:
        x, y = p  
        h ^= zobrist[x][y][0]
    for p in board.stonedict.d['WHITE']:
        x, y = p
        h ^= zobrist[x][y][1]
    return h

class LightweightBoardHandler():
    def __init__(self, board: Board):
        self.EMPTY = 0
        self.BLACK = 1
        self.WHITE = -1
        self.parent = {}
        self.group_data = {}
        self.winner = None
        self.draw = False
        self.next = self.BLACK if board.next == 'BLACK' else self.WHITE

        self.DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))

        self.possible_actions = {self.BLACK: set() ,self.WHITE: set()}
        self.endangered_groups = {self.BLACK: set(),self.WHITE: set()}
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), np.int8)
        self.undo_stack: list[tuple] = []

        self.zobrist = make_zobrist(board_size=BOARD_SIZE)
        self.hash = 0
        
        for point, groups in board.stonedict.d["BLACK"].items():
            if groups: 
                self.put_stone(point, self.BLACK)
                self.hash ^= self.zobrist[point[0]][point[1]][1]
        for point,groups in board.stonedict.d["WHITE"].items():
            if groups: 
                self.put_stone(point, self.WHITE)
                self.hash ^= self.zobrist[point[0]][point[1]][0]

    def find_root(self, point: tuple[int,int]):
        if(self.parent[point] == point): return point
        self.parent[point] = self.find_root(self.parent[point])        
        return self.parent[point]
    
    def union(self, point1: tuple[int,int], point2: tuple[int,int]):
        root1 = self.find_root(point1)
        root2 = self.find_root(point2)
        if root1 != root2:
            if self.group_data[root1]['stones'] < self.group_data[root2]['stones']:
                root1, root2 = root2, root1

            self.parent[root2] = root1

            self.endangered_groups[self.board[root1]].discard(root1)
            self.endangered_groups[self.board[root2]].discard(root2)

            self.group_data[root1]['stones'] += self.group_data[root2]['stones']
            self.group_data[root1]['liberties'].update(self.group_data[root2]['liberties'])
            
            del self.group_data[root2]
    
    def get_opponent(self, color: int):
        return 1 if color==-1 else -1

    def put_stone(self, point: tuple[int,int], color: int):
        self.board[point] = color
        self.parent[point] = point
        self.group_data[point] = {'liberties' : set(), 'stones': 1}
        
        liberties = []
        for dir in self.DIRS:
            x = point[0] + dir[0]
            y = point[1] + dir[1]

            if((1 <= x < BOARD_SIZE) and (1 <= y < BOARD_SIZE)):
                if(self.board[(x,y)] == color):
                    self.union((x,y), point)
                    self.group_data[self.find_root(point)]['liberties'].discard(point)
                elif(self.board[(x,y)] == self.get_opponent(color)):
                    opp_root = self.find_root((x,y))
                    opp_group = self.group_data[opp_root]
                    opp_group['liberties'].discard(point)
                    remaining_liberties = len(opp_group['liberties'])
                    if remaining_liberties == 1: 
                        self.endangered_groups[self.get_opponent(color)].add(opp_root)
                    elif remaining_liberties == 0: 
                        self.winner = color
                else:
                    liberties.append((x,y))
        final_root = self.find_root(point)
        final_group = self.group_data[final_root]
        final_group['liberties'].update(liberties)

        self.possible_actions[self.BLACK].discard(point)
        self.possible_actions[self.WHITE].discard(point)

        self.possible_actions[self.get_opponent(color)].update(liberties)
        remaining_liberties = len(final_group['liberties'])
        if remaining_liberties == 1: 
            self.endangered_groups[color].update([final_root])
        elif remaining_liberties == 0: 
            self.winner = self.get_opponent(color)

        if(self.possible_actions[self.WHITE] == self.possible_actions[self.BLACK] == set()): self.draw = True

    def get_legal_actions(self):
        opponent = self.get_opponent(self.next)
        if self.endangered_groups[opponent]:
            capture_moves = set()
            for opp_root in self.endangered_groups[opponent]:
                capture_moves.update(self.group_data[opp_root]['liberties'])
            return list(capture_moves)
        
        if self.endangered_groups[self.next]:
            actions = set()
            for root in self.endangered_groups[self.next]:
                actions.update(self.group_data[root]['liberties'])
            return list(actions)
        
        moves = []
        for m in self.possible_actions[self.next]:
            if not self._is_suicide(m, self.next):
                moves.append(m)
        return moves
    
    def _is_suicide(self, point: tuple[int, int], color: int) -> bool:
        for dx, dy in self.DIRS:
            x, y = point[0] + dx, point[1] + dy
            if( (1 <= x < BOARD_SIZE) and (1 <= y < BOARD_SIZE)):
                if self.board[(x, y)] == self.EMPTY:
                    return False

                if self.board[(x, y)] == color:
                    root = self.find_root((x, y))
                    liberties = self.group_data[root]['liberties']

                    if len(liberties) > 1 :
                        return False 
        return True
    
    def _snapshot(self, action: tuple[int,int]):
        self.undo_stack.append((
            self.parent.copy(),
            {k: {'stones': v['stones'], 'liberties': v['liberties'].copy()} for k,v in self.group_data.items()},
            {k: v.copy() for k,v in self.endangered_groups.items()},
            {k: v.copy() for k, v in self.possible_actions.items()},
            action,
            self.hash,
            self.winner,
            self.draw,
            self.next
        ))

    def undo(self):
        if not self.undo_stack:
            raise RuntimeError("Nothing to undo")
        (self.parent,
         self.group_data,
         self.endangered_groups,
         self.possible_actions,
         action,
         self.hash, self.winner, self.draw, self.next) = self.undo_stack.pop()

        self.board[action] = self.EMPTY

    def perform_move(self, action):
        self._snapshot(action)
        self.put_stone(action, self.next)
        self.next = self.get_opponent(self.next)
        color_idx = 0 if self.next == -1 else 1
        self.hash ^= self.zobrist[action[0]][action[1]][color_idx]


class Agent1v4:
    """A class to generate a random action for a Go board."""
    def __init__(self, color, max_depth = None, verbose = False):
        self.color = -1 if color == 'WHITE' else 1
        self.opponent_color = -self.color
        self.MAX_DEPTH = 5 if not max_depth else max_depth

        self.W_LIBERTIES = 10
        self.W_ATARI = 40
        self.W_WIN = 1e5
        self.verbose = verbose

        self.light_board = None
        
        self.transposition = {}   

    def evaluate(self, board: LightweightBoardHandler) -> int:
        if(board.draw): return 0
        if(board.winner is not None):
            if(board.winner == self.color): return self.W_WIN
            else: return -self.W_WIN
        score = 0
        score += self.W_LIBERTIES * (len(board.possible_actions[self.opponent_color]) - len(board.possible_actions[self.color]))
        return score

    def minimax(self, board: LightweightBoardHandler, depth: int, alpha: int, beta: int, maximizing_player: bool) -> int:  
        # color_idx = 0 if board.next == 'BLACK' else 1
        key = (board.hash, board.next)
        
        if(key in self.transposition):
            return self.transposition[key]
        
        legal_actions = board.get_legal_actions()
        if(len(legal_actions) == 0): 
            return 0,1
        
        pos = 0   
        if((depth == 0) or (board.winner is not None)):
            score = self.evaluate(board)
            self.transposition[key] = (score,1)
            return score,1
        if(maximizing_player):
            max_score = -1e9
            for action in legal_actions:
                board.perform_move(action)
                # print(board.parent)
                # successor = board.copy()
                # successor.put_stone(action, check_legal=False)
                # score, new_pos = self.minimax(successor, depth-1, False)
                score, new_pos = self.minimax(board, depth-1, alpha, beta, False)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                pos += new_pos
                if(beta <= alpha): 
                    board.undo()
                    break
                board.undo()
                # print("RESTORED: ")
                # print(self.light_board.board)
                
            self.transposition[key] = (max_score, pos)
            return max_score, pos
        
        else:
            min_score = 1e9
            for action in legal_actions:
                board.perform_move(action)
                # print(board.parent)
                # successor = board.copy()
                # successor.put_stone(action, check_legal=False)
                # score, new_pos = self.minimax(successor, depth-1, True)
                score, new_pos = self.minimax(board, depth-1, alpha, beta, True)
                min_score = min(min_score, score)
                beta = min(beta, score)
                pos += new_pos
                if(beta <= alpha): 
                    board.undo()
                    break
                board.undo()
                # print("RESTORED: ")
                # print(self.light_board.board)
            self.transposition[key] = (min_score, pos)
            return min_score, pos

    def get_best_action(self, board: Board, depth: int):
        pos = 0
        best_score = -1e9
        best_action = None

        self.transposition.clear()
        # print(f"Legal Actions: {board.legal_actions}")
        # self.print_point_dict(board.libertydict)
        self.light_board = LightweightBoardHandler(board)
        legal_actions = self.light_board.get_legal_actions()
        for action in legal_actions:   
            # print("STORED_______________________")
            # for key,val in snap.libertydict['BLACK'].items(): 
            #     print(key , val)
        
            # for key,val in snap.libertydict['WHITE'].items(): 
            #     print(key , val)
            # print()

            self.light_board.perform_move(action)
            # print(self.light_board.parent)
            # successor = board.copy()
            # successor.put_stone(action, check_legal=False)
            # score, new_pos = self.minimax(successor, depth, maximizing_player=False)
            # self.print_point_dict(board.libertydict)
            score, new_pos = self.minimax(self.light_board, depth, alpha=-1e9, beta=1e9, maximizing_player=False)
            # if(self.verbose):
                # self.light_board.perform_move(self.light_board.get_legal_actions()[0])
                # print(self.light_board.possible_actions, score)         
                # self.light_board.undo()
            pos += new_pos
            if(score > best_score):
                best_score = score
                best_action = action
            
            self.light_board.undo()
            # print("RESTORED: ")
            # print(self.light_board.board)
        # print("---------------------------------------------------------------------------")
        return best_action, pos

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
            best_action, pos = self.get_best_action(board, self.MAX_DEPTH)
            # print("BEST ACTION: ", best_action, board.next)
            if(self.verbose):
                print(f"Possibilities considered: {pos}")
                print(f"Decision Time: {time.time() - start_time}. ({pos/(time.time() - start_time)} possibilities/sec)")
                # print(self.light_board.board)
                # print(self.light_board.get_legal_actions())

            if(time.time() - start_time > 10): 
                print(f"Took too much time: {time.time() - start_time}")
                print(f"Possibilities considered: {pos}")
            return best_action
        return None
    
class Agent1v3:
    """A class to generate a random action for a Go board."""
    def __init__(self, color, verbose = False):
        self.color = -1 if color == 'WHITE' else 1
        self.opponent_color = -self.color
        self.MAX_DEPTH = 4

        self.W_LIBERTIES = 10
        self.W_ATARI = 40
        self.W_WIN = 1e5
        self.verbose = verbose

        self.light_board = None
        
        self.transposition = {}   

    def evaluate(self, board: LightweightBoardHandler) -> int:
        if(board.draw): return 0
        if(board.winner is not None):
            if(board.winner == self.color): return self.W_WIN
            else: return -self.W_WIN
        score = 0
        score += self.W_LIBERTIES * (len(board.possible_actions[self.opponent_color]) - len(board.possible_actions[self.color]))
        return score

    def minimax(self, board: LightweightBoardHandler, depth: int, maximizing_player: bool) -> int:  
        # color_idx = 0 if board.next == 'BLACK' else 1
        key = (board.hash, board.next)
        
        if(key in self.transposition):
            return self.transposition[key]
        
        if(len(board.get_legal_actions()) == 0): 
            return 0,1
        
        pos = 0   
        if((depth == 0) or (board.winner is not None)):
            score = self.evaluate(board)
            self.transposition[key] = (score,1)
            return score,1
        if(maximizing_player):
            max_score = -1e9
            for action in board.get_legal_actions():
                board.perform_move(action)
                # print(board.parent)
                # successor = board.copy()
                # successor.put_stone(action, check_legal=False)
                # score, new_pos = self.minimax(successor, depth-1, False)
                score, new_pos = self.minimax(board, depth-1, False)
                max_score = max(max_score, score)
                pos += new_pos
                board.undo()
                # print("RESTORED: ")
                # print(self.light_board.board)
                
            self.transposition[key] = (max_score, pos)
            return max_score, pos
        
        else:
            min_score = 1e9
            for action in board.get_legal_actions():
                board.perform_move(action)
                # print(board.parent)
                # successor = board.copy()
                # successor.put_stone(action, check_legal=False)
                # score, new_pos = self.minimax(successor, depth-1, True)
                score, new_pos = self.minimax(board, depth-1, True)
                min_score = min(min_score, score)
                pos += new_pos
                board.undo()
                # print("RESTORED: ")
                # print(self.light_board.board)
            self.transposition[key] = (min_score, pos)
            return min_score, pos

    def get_best_action(self, board: Board, depth: int):
        pos = 0
        best_score = -1e9
        best_action = None

        self.transposition.clear()
        # print(f"Legal Actions: {board.legal_actions}")
        # self.print_point_dict(board.libertydict)
        self.light_board = LightweightBoardHandler(board)

        for action in self.light_board.get_legal_actions():            
            # print("STORED_______________________")
            # for key,val in snap.libertydict['BLACK'].items(): 
            #     print(key , val)
        
            # for key,val in snap.libertydict['WHITE'].items(): 
            #     print(key , val)
            # print()

            self.light_board.perform_move(action)
            # print(self.light_board.parent)
            # successor = board.copy()
            # successor.put_stone(action, check_legal=False)
            # score, new_pos = self.minimax(successor, depth, maximizing_player=False)
            # self.print_point_dict(board.libertydict)
            score, new_pos = self.minimax(self.light_board, depth, maximizing_player=False)
            pos += new_pos
            if(score > best_score):
                best_score = score
                best_action = action
            
            self.light_board.undo()
            # print("RESTORED: ")
            # print(self.light_board.board)

        return best_action, pos

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
            best_action, pos = self.get_best_action(board, self.MAX_DEPTH)
            # print("BEST ACTION: ", best_action, board.next)
            if(self.verbose):
                print(f"Possibilities considered: {pos}")
                print(f"Decision Time: {time.time() - start_time}. ({pos/(time.time() - start_time)} possibilities/sec)")
                # print(self.light_board.board)
                # print(self.light_board.possible_actions)

            if(time.time() - start_time > 10): 
                print(f"Took too much time: {time.time() - start_time}")
                print(f"Possibilities considered: {pos}")
            return best_action
        return None
###########################################################################################################

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

class Agent1v2:
    """A class to generate a random action for a Go board."""
    def __init__(self, color, verbose = False):
        self.color = color
        self.opponent_color = opponent_color(self.color)
        self.MAX_DEPTH = 4

        self.W_LIBERTIES = 10
        self.W_ATARI = 40
        self.W_WIN = 1e5
        self.verbose = verbose

        self._snap_stack = [BoardSnapshot.__new__(BoardSnapshot)
                            for _ in range(self.MAX_DEPTH + 2)]
        
        self._shadow_pools = [{} for _ in range(self.MAX_DEPTH + 2)]

        self.zobrist = make_zobrist(board_size=20)
        self.transposition = {}   

    def evaluate(self, board: Board) -> int:
        if(board.winner is not None):
            if(board.winner == self.color): return self.W_WIN
            else: return -self.W_WIN
        score = 0
        score += self.W_LIBERTIES * (len(board.libertydict.d[self.color]) - len(board.libertydict.d[self.opponent_color]))
        return score

    def minimax(self, board: Board, depth: int, maximizing_player: bool, hash: int) -> int:  
        color_idx = 0 if board.next == 'BLACK' else 1
        key = (hash, color_idx)
        
        if(key in self.transposition):
            return self.transposition[key]
        
        pos = 0   
        if((depth == 0) or (board.winner is not None)):
            score = self.evaluate(board)
            self.transposition[key] = (score,1)
            return score,1

        if(len(board.legal_actions) == 0): 
            return 0,1
        
        snap = self._snap_stack[depth]
        shadow_pool = self._shadow_pools[depth]
        snap.__init__(board, shadow_pool)

        if(maximizing_player):
            max_score = -1e9
            for action in board.legal_actions:
                x, y = action
                new_h = hash ^ self.zobrist[x][y][color_idx]
                board.put_stone(action, check_legal=False)
                # successor = board.copy()
                # successor.put_stone(action, check_legal=False)
                # score, new_pos = self.minimax(successor, depth-1, False)
                score, new_pos = self.minimax(board, depth-1, False, new_h)
                max_score = max(max_score, score)
                pos += new_pos
                restore(board, snap)
            self.transposition[key] = (max_score, pos)
            return max_score, pos
        
        else:
            min_score = 1e9
            for action in board.legal_actions:
                x, y = action
                new_h = hash ^ self.zobrist[x][y][color_idx]
                board.put_stone(action, check_legal=False)
                # successor = board.copy()
                # successor.put_stone(action, check_legal=False)
                # score, new_pos = self.minimax(successor, depth-1, True)
                score, new_pos = self.minimax(board, depth-1, True, new_h)
                min_score = min(min_score, score)
                pos += new_pos
                restore(board, snap)
            self.transposition[key] = (min_score, pos)
            return min_score, pos

    def get_best_action(self, board: Board, depth: int):
        pos = 0
        best_score = -1e9
        best_action = None

        self.transposition.clear()
        # print(f"Legal Actions: {board.legal_actions}")
        # self.print_point_dict(board.libertydict)
        snap = self._snap_stack[-1]
        snap = self._snap_stack[-1]
        snap.__init__(board, self._shadow_pools[-1]) 
        initial_hash = board_hash(board, self.zobrist)

        for action in board.legal_actions:
            x, y = action
            color_idx = 0 if board.next == 'BLACK' else 1
            new_h = initial_hash ^ self.zobrist[x][y][color_idx]
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
            score, new_pos = self.minimax(board, depth, maximizing_player=False, hash=new_h)
            pos += new_pos
            if(score > best_score):
                best_score = score
                best_action = action
            
            restore(board, snap)
            # print("RESTORED: ")
            # self.print_point_dict(board.libertydict)
        return best_action, pos

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
            best_action, pos = self.get_best_action(board, self.MAX_DEPTH)
            # print("BEST ACTION: ", best_action, board.next)
            if(self.verbose):
                print(f"Decision Time: {time.time() - start_time}")
                print(f"Possibilities considered: {pos}")

            if(time.time() - start_time > 10): 
                print(f"Took too much time: {time.time() - start_time}")
                print(f"Possibilities considered: {pos}")
                print(board.legal_actions)
            return best_action
        return None
###########################################################################################################################

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