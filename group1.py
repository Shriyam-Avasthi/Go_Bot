# You need to complete this function which should return action,

from game.go import Board
import random
from game.go import opponent_color, BOARD_SIZE
from game.util import PointDict
from copy import deepcopy
import time
import numpy as np
import math

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
    def __init__(self, board: Board, save_states = True):
        self.EMPTY = 0
        self.BLACK = 1
        self.WHITE = -1
        self.parent = {}
        self.group_data = {}
        self.winner = None
        self.draw = False
        self.next = self.BLACK if board.next == 'BLACK' else self.WHITE
        self.stone_count = 0
        self.save_states = save_states
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
                self.hash ^= self.zobrist[point[0]][point[1]][0]
        for point,groups in board.stonedict.d["WHITE"].items():
            if groups: 
                self.put_stone(point, self.WHITE)
                self.hash ^= self.zobrist[point[0]][point[1]][1]

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
        return -color

    def put_stone(self, point: tuple[int,int], color: int):
        self.board[point] = color
        self.parent[point] = point
        self.stone_count += 1
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
                for liberty in self.group_data[root]['liberties']:
                    if not self._is_suicide(liberty, self.next):
                        actions.add(liberty)
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
        self.stone_count -= 1

    def perform_move(self, action):
        current_player = self.next
        if(self.save_states):
            self._snapshot(action)
        self.put_stone(action, self.next)
        self.next = self.get_opponent(self.next)
        
        color_idx = 0 if current_player == self.BLACK else 1
        self.hash ^= self.zobrist[action[0]][action[1]][color_idx]


class MCTSNode:
    def __init__(self, move=None, parent=None, to_play=None):
        self.move = move     
        self.parent = parent
        self.children = {}     
        self.visits = 0
        self.wins = 0.0            
        self.untried_moves = None   
        self.to_play = to_play    

    def q(self):
        return self.wins

    def n(self):
        return self.visits
    
class MCTS:
    def __init__(self, color: int, time_limit: float = None, iterations: int = 1000, c_puct: float = 1.4):
        self.color = color
        self.opponent_color = -color
        self.iterations = iterations
        self.time_limit = time_limit
        self.c = c_puct
        self.rng = random.Random()

        # Reduced weights for more conservative evaluation
        self.W_LIBERTIES = 5
        self.W_MOBILITY = 10
        self.W_ATARI = 100
        self.W_WIN = 1e5

    def evaluate_v2(self, board: LightweightBoardHandler) -> int:
        if board.draw:
            return 0
        if board.winner is not None:
            return self.W_WIN if board.winner == self.color else -self.W_WIN

        legal_actions = board.get_legal_actions()
        if not legal_actions:
            return self.W_WIN if board.next == self.opponent_color else -self.W_WIN

        own_atari = len(board.endangered_groups[self.color])
        opp_atari = len(board.endangered_groups[self.opponent_color])

        score = self.W_ATARI * (opp_atari - own_atari)

        if (board.next == self.color) and (opp_atari > 0):
            return self.W_WIN * 0.95
        if (board.next == self.opponent_color) and (own_atari > 0):
            return -self.W_WIN * 0.95
        
        own_libs = len(board.possible_actions[self.color]) 
        opp_libs = len(board.possible_actions[self.opponent_color])
                
        score += self.W_LIBERTIES * (own_libs - opp_libs)

        if board.next == self.color:
            score += self.W_MOBILITY * len(legal_actions)
        else:
            score -= self.W_MOBILITY * len(legal_actions)

        return score
    
    def eval_to_reward(self, eval_score: float) -> float:
        if eval_score >= 1e4:
            return 1.0
        if eval_score <= -1e4:
            return 0
        
        scale = 500.0  # Controls steepness
        normalized = math.tanh(eval_score / scale)
        return 0.5 * (normalized + 1.0)

    def _get_move_scores(self, board: LightweightBoardHandler):
        move_scores = {}
        opponent = board.get_opponent(board.next)

        # Priority 1: Capture moves
        if board.endangered_groups[opponent]:
            for opp_root in board.endangered_groups[opponent]:
                for liberty in board.group_data[opp_root]['liberties']:
                    move_scores[liberty] = 1000

        # Priority 2: Saving moves
        if board.endangered_groups[board.next]:
            for root in board.endangered_groups[board.next]:
                for liberty in board.group_data[root]['liberties']:
                    if not board._is_suicide(liberty, board.next):
                        if liberty not in move_scores:
                            move_scores[liberty] = 900

        # Priority 3: Threatening moves (reduce opponent to 2 liberties)
        for root, data in board.group_data.items():
            if board.board[root] == opponent and len(data['liberties']) == 2:
                for liberty in data['liberties']:
                    if not board._is_suicide(liberty, board.next):
                        if liberty not in move_scores:
                            move_scores[liberty] = 500
        
        return move_scores

    def _uct_select(self, node: MCTSNode):
        best_move, best_child, best_val = None, None, -1e9
        
        for move, child in node.children.items():
            if child.n() == 0:
                u = float('inf')
            else:
                child_winrate = child.q() / child.n()
                exploit = 1.0 - child_winrate
                explore = self.c * math.sqrt(math.log(node.n() + 1) / child.n())
                u = exploit + explore
            
            if u > best_val:
                best_val = u
                best_child = child
                best_move = move
        
        return best_move, best_child

    def _rollout_policy(self, board: 'LightweightBoardHandler'):
        """
        Improved rollout policy: prioritize capture/save moves, 
        then sample randomly with weighted probabilities.
        """
        moves = board.get_legal_actions()
        if not moves:
            return None
            
        move_scores = self._get_move_scores(board)
        
        # First check for high-priority moves
        for move in moves:
            if move_scores.get(move, 0) >= 900:  # Capture or save
                return move
        
        # Otherwise, use weighted random selection
        weights = [move_scores.get(m, 1) for m in moves]
        total = sum(weights)
        if total == 0:
            return self.rng.choice(moves)
        
        # Weighted random choice
        r = self.rng.random() * total
        cumsum = 0
        for move, weight in zip(moves, weights):
            cumsum += weight
            if r <= cumsum:
                return move
        
        return moves[-1]  # Fallback

    def _simulate(self, board: LightweightBoardHandler, rollout_limit: int = 50):

        initial_player = board.next
        steps = 0
        
        while (board.winner is None) and (not board.draw) and (steps < rollout_limit):
            moves = board.get_legal_actions()
            if not moves:
                break
            
            move = self._rollout_policy(board)
            if move is None:
                break
                
            board.perform_move(move)
            steps += 1

        if board.draw:
            reward = 0.5
        elif board.winner is not None:
            reward = 1.0 if board.winner == initial_player else 0.0
        else:
            eval_score = self.evaluate_v2(board)
            if initial_player == self.color:
                reward = self.eval_to_reward(eval_score)
            else:
                reward = 1.0 - self.eval_to_reward(eval_score)
        
        return reward, steps

    def _expand(self, node: MCTSNode, board: LightweightBoardHandler):

        if node.untried_moves is None:
            moves = board.get_legal_actions()
            move_scores = self._get_move_scores(board)
            moves.sort(key=lambda m: move_scores.get(m, 0), reverse=True)
            node.untried_moves = moves
        
        if not node.untried_moves:
            return None
        
        move = node.untried_moves.pop(0)
        board.perform_move(move)
        
        child = MCTSNode(move=move, parent=node, to_play=board.next)
        node.children[move] = child
        return child

    def _backpropagate(self, node: MCTSNode, reward: float):
        current_reward = reward
        
        while node is not None:
            node.visits += 1
            node.wins += current_reward
            
            current_reward = 1.0 - current_reward
            node = node.parent

    def choose(self, board: 'Board'):
        
        light = LightweightBoardHandler(board, save_states=True)
        root = MCTSNode(move=None, parent=None, to_play=light.next)
        root.untried_moves = None
        root.visits = 1
        root.wins = 0.5

        start_time = time.time()
        iterations = 0
        initial_stack_size = len(light.undo_stack)
        
        while True:
            if (self.time_limit is not None) and ((time.time() - start_time) >= self.time_limit):
                break
            if (self.time_limit is None) and (iterations >= self.iterations):
                break
            iterations += 1

            node = root

            # Selection
            while (node.untried_moves is not None and len(node.untried_moves) == 0) and node.children:
                move, child = self._uct_select(node)
                if child is None:
                    break
                light.perform_move(move)
                node = child

            # Expansion
            if light.winner is None and not light.draw:
                child = self._expand(node, light)
                if child is not None:
                    node = child

            # Simulation
            reward, steps = self._simulate(light)

            # Backprop
            self._backpropagate(node, reward)

            while len(light.undo_stack) > initial_stack_size:
                light.undo()

        print(f"MCTS Iterations: {iterations}")
        if root.children:
            sorted_children = sorted(root.children.items(), 
                                   key=lambda x: x[1].visits, 
                                   reverse=True)
            print("Top moves:")
            for mv, ch in sorted_children[:5]:
                winrate = ch.wins / max(1, ch.visits)
                print(f"  {mv}: visits={ch.visits}, winrate={winrate:.3f}")

        if not root.children:
            return None
        
        best = max(root.children.values(), key=lambda n: n.visits)
        return best.move
    

class Agent1v5:

    def __init__(self, color, time_limit: float = 1.0, iterations: int = None, verbose: bool = False):

        self.color = -1 if color == 'WHITE' else 1
        self.opponent_color = -self.color
        self.verbose = verbose

        if iterations is not None:
            self.mcts = MCTS(color=self.color, iterations=iterations, time_limit=None)
        else:
            self.mcts = MCTS(color=self.color, time_limit=time_limit)

    def get_action(self, board: Board):
        
        start_time = time.time()
        move = self.mcts.choose(board)
        if self.verbose:
            # if move:
            #     print(f"[Agent1v5] Selected move {move} for {('BLACK' if self.color==1 else 'WHITE')}")
            # else:
            #     print(f"[Agent1v5] No legal moves for {('BLACK' if self.color==1 else 'WHITE')}")
            print(f"[Agent1v5] Decision time: {time.time() - start_time:.3f}s")
        return move

#################################################################################################################################

class Agent1v6:
    def __init__(self, color, max_time = None, verbose = False):
        self.color = -1 if color == 'WHITE' else 1
        self.opponent_color = -self.color
        self.MAX_TIME = 2 if not max_time else max_time

        self.W_LIBERTIES = 10
        self.W_MOBILITY = 30
        self.W_ATARI = 80
        self.W_WIN = 1e5
        self.verbose = verbose

        self.light_board = None
        
        self.transposition = {}   
        self.best_moves_history = {}

        self.EXACT_FLAG = 0
        self.LOWERBOUND_FLAG = 1
        self.UPPERBOUND_FLAG = 2  

        self.TIME_CHECK_NODES = 128
        self.start_time = 0
        self.node_count = 0

        self.time_is_up = False  

    def time_exceeded(self):
        self.node_count += 1
        if self.node_count % self.TIME_CHECK_NODES == 0:
            self.node_count = 0
            self.elapsed = time.time() - self.start_time
            self.time_is_up = (self.elapsed >= self.MAX_TIME)
        return self.time_is_up

    def _get_move_scores(self, board: LightweightBoardHandler):
        move_scores = {}
        opponent = board.get_opponent(board.next)

        # Priority 1: Capture moves are the best
        if board.endangered_groups[opponent]:
            for opp_root in board.endangered_groups[opponent]:
                for liberty in board.group_data[opp_root]['liberties']:
                    move_scores[liberty] = 100

        # Priority 2: Saving moves are the next best
        if board.endangered_groups[board.next]:
            for root in board.endangered_groups[board.next]:
                for liberty in board.group_data[root]['liberties']:
                    if not board._is_suicide(liberty, board.next):
                        if liberty not in move_scores:
                            move_scores[liberty] = 90

        # Priority 3: Threatening moves
        for root, data in board.group_data.items():
            if board.board[root] == opponent and len(data['liberties']) == 2:
                for liberty in data['liberties']:
                    if not board._is_suicide(liberty, board.next):
                        if liberty not in move_scores:
                            move_scores[liberty] = 50
        
        return move_scores
    
    def _count_potential_eyes_numpy(self, board: LightweightBoardHandler):
        padded_board = np.full((BOARD_SIZE + 2, BOARD_SIZE + 2), 99)
        padded_board[1:-1, 1:-1] = board.board

        center      = padded_board[1:-1, 1:-1]
        north      = padded_board[0:-2, 1:-1]
        south       = padded_board[2:  , 1:-1]
        west        = padded_board[1:-1, 0:-2]
        east        = padded_board[1:-1, 2:  ]

        is_empty          = (center == board.EMPTY)
        north_is_own      = (north == self.color)
        south_is_own      = (south == self.color)
        west_is_own       = (west  == self.color)
        east_is_own       = (east  == self.color)
        
        own_eyes_mask = is_empty & north_is_own & south_is_own & west_is_own & east_is_own
        own_eyes_count = np.sum(own_eyes_mask)

        north_is_opp      = (north == self.opponent_color)
        south_is_opp      = (south == self.opponent_color)
        west_is_opp       = (west  == self.opponent_color)
        east_is_opp       = (east  == self.opponent_color)

        opp_eyes_mask = is_empty & north_is_opp & south_is_opp & west_is_opp & east_is_opp
        opp_eyes_count = np.sum(opp_eyes_mask)
        
        return own_eyes_count, opp_eyes_count
    
    def _order_moves(self, board: LightweightBoardHandler, legal_actions: list, depth: int):
        move_scores = self._get_move_scores(board)
        
        tt_move = None
        key = (board.hash, board.next) 
        if key in self.transposition:
            entry = self.transposition[key]
            tt_move = entry[4]

        for action in legal_actions:
            score = move_scores.get(action, 0)
            
            if action == tt_move:
                score += 2000

            if board.hash in self.best_moves_history and self.best_moves_history[board.hash] == action:
                score += 1000 
            
            move_scores[action] = score
        
        legal_actions.sort(key=lambda m: move_scores.get(m, 0), reverse=True)
        return legal_actions

    def evaluate_v2(self, board: LightweightBoardHandler) -> int:
        if board.draw:
            return 0
        if board.winner is not None:
            return self.W_WIN if board.winner == self.color else -self.W_WIN

        legal_actions = board.get_legal_actions()
        if not legal_actions:
            return self.W_WIN if board.next == self.opponent_color else -self.W_WIN

        own_atari = len(board.endangered_groups[self.color])
        opp_atari = len(board.endangered_groups[self.opponent_color])

        score = self.W_ATARI * (opp_atari - own_atari)


        if (board.next == self.color) and (opp_atari > 0):
            return self.W_WIN * 0.9
        if (board.next == self.opponent_color) and (own_atari > 0):
            return -self.W_WIN * 0.9
        
        own_libs = len(board.possible_actions[self.color]) 
        opp_libs = len(board.possible_actions[self.opponent_color])

        # own_eyes, opp_eyes = self._count_potential_eyes_numpy(board)
                
        # W_EYE = 200
        # score += W_EYE * (own_eyes - opp_eyes)
        score += self.W_LIBERTIES * (own_libs - opp_libs)

        if board.next == self.color:
            score += self.W_MOBILITY * len(legal_actions)
        else:
            score -= self.W_MOBILITY * len(legal_actions)

        return score
    
    def quiescence_search(self, board: LightweightBoardHandler, depth: int, alpha: int, beta: int, maximizing_player: bool):
        if self.time_exceeded():
            return self.evaluate_v2(board), 0
        
        key = (board.hash, board.next)
        q_depth_offset = 1000

        if key in self.transposition:
            score, stored_depth, flag, _, best_move = self.transposition[key]
            if stored_depth >= (q_depth_offset + depth):
                if flag == self.EXACT_FLAG:
                    return score, 0
                elif flag == self.LOWERBOUND_FLAG:
                    alpha = max(alpha, score)
                elif flag == self.UPPERBOUND_FLAG:
                    beta = min(beta, score)
                
                if alpha >= beta:
                    return score, 0

        initial_eval = self.evaluate_v2(board)
        if depth == 0:
            return initial_eval, 1

        if maximizing_player:
            alpha = max(alpha, initial_eval)
        else:
            beta = min(beta, initial_eval)
        if alpha >= beta:
            return initial_eval, 1
            
        violent_moves = [action for action in self._order_moves(board, board.get_legal_actions(), 0) 
                        if self._get_move_scores(board).get(action, 0) > 0][:5]
        
        if not violent_moves:
            return initial_eval, 1
        
        pos = 0
        original_alpha = alpha
        original_beta = beta
        best_action_for_node = violent_moves[0]

        if maximizing_player:
            max_score = initial_eval
            for action in violent_moves:
                board.perform_move(action)
                score, new_pos = self.quiescence_search(board, depth - 1, alpha, beta, False)
                board.undo()

                if self.time_is_up:
                    return max_score, 0
                
                pos += new_pos
                if score > max_score:
                    max_score = score
                    best_action_for_node = action
                
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            
            flag = self.EXACT_FLAG
            if max_score <= original_alpha:
                flag = self.UPPERBOUND_FLAG
            elif max_score >= beta:
                flag = self.LOWERBOUND_FLAG
            
            self.transposition[key] = (max_score, q_depth_offset + depth, flag, 0, best_action_for_node)
            return max_score, pos
        
        else:
            min_score = initial_eval
            for action in violent_moves:
                board.perform_move(action)
                score, new_pos = self.quiescence_search(board, depth - 1, alpha, beta, True)
                board.undo()
                
                if self.time_is_up:
                    return min_score, 0
                
                pos += new_pos
                if score < min_score:
                    min_score = score
                    best_action_for_node = action
                
                beta = min(beta, score)
                if alpha >= beta:
                    break
            
            flag = self.EXACT_FLAG
            if min_score >= original_beta:
                flag = self.LOWERBOUND_FLAG
            elif min_score <= alpha:
                flag = self.UPPERBOUND_FLAG
            
            self.transposition[key] = (min_score, q_depth_offset + depth, flag, 0, best_action_for_node)
            return min_score, pos

    def minimax(self, board: LightweightBoardHandler, depth: int, alpha: int, beta: int, maximizing_player: bool) -> tuple:
        original_alpha = alpha
        original_beta = beta
        key = (board.hash, board.next)

        if key in self.transposition:
            score, stored_depth, flag, _, best_move = self.transposition[key]

            if stored_depth >= depth and stored_depth < 1000:  # Not from quiescence
                if flag == self.EXACT_FLAG:
                    return score, 0
                elif flag == self.LOWERBOUND_FLAG:
                    alpha = max(alpha, score)
                elif flag == self.UPPERBOUND_FLAG:
                    beta = min(beta, score)
                
                if alpha >= beta:
                    return score, 0
        
        if board.winner is not None or board.draw:
            return self.evaluate_v2(board), 1
        
        if self.time_exceeded():
            return self.evaluate_v2(board), 0
        
        if (depth == 0):
            time_remaining = self.MAX_TIME - self.elapsed
            if time_remaining < 0.5:
                return self.evaluate_v2(board), 1
            q_depth = min(3, int(time_remaining * 2))
            return self.quiescence_search(board, q_depth, alpha, beta, maximizing_player)
        

        pos = 0
        
        legal_actions = self._order_moves(board, board.get_legal_actions(), depth)
        
        if not legal_actions:
            return self.evaluate_v2(board), 1

        best_action_for_node = legal_actions[0]

        if maximizing_player:
            max_score = -1e9
            for action in legal_actions:
                board.perform_move(action)
                score, new_pos = self.minimax(board, depth - 1, alpha, beta, False)
                board.undo()

                if self.time_is_up:
                    return max_score, 0
                
                pos += new_pos
                if score > max_score:
                    max_score = score
                    best_action_for_node = action
                
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            
            flag = self.EXACT_FLAG
            if max_score <= original_alpha:
                flag = self.UPPERBOUND_FLAG
            elif max_score >= beta:
                flag = self.LOWERBOUND_FLAG
            
            self.transposition[key] = (max_score, depth, flag, 0, best_action_for_node)
            return max_score, pos
        
        else:
            min_score = 1e9
            for action in legal_actions:
                board.perform_move(action)
                score, new_pos = self.minimax(board, depth - 1, alpha, beta, True)
                board.undo()

                if self.time_is_up:
                    return min_score, 0
                
                pos += new_pos
                if score < min_score:
                    min_score = score
                    best_action_for_node = action
                
                beta = min(beta, score)
                if alpha >= beta:
                    break
            
            flag = self.EXACT_FLAG
            if min_score >= original_beta:
                flag = self.LOWERBOUND_FLAG
            elif min_score <= alpha:
                flag = self.UPPERBOUND_FLAG
            
            self.transposition[key] = (min_score, depth, flag, 0, best_action_for_node)
            return min_score, pos

    def iterative_deepening(self, board: LightweightBoardHandler, max_time: int):
        best_action = None
        best_score = -1e9
        total_pos = 0
        current_depth = 1
        self.start_time = time.time()
        self.time_is_up = 0
        self.node_count = 0
        self.elapsed = 0
        
        self.transposition.clear()
        
        while self.elapsed < max_time:
            if board.winner is not None or board.draw:
                break

            self.current_iteration_depth = current_depth
            
            if self.verbose:
                print(f"[{'BLACK' if self.color == 1 else 'WHITE'}] Searching depth {current_depth}...")
            
            depth_best_action = None
            depth_best_score = -1e9
            pos = 0
            
            legal_actions = board.get_legal_actions()
            
            if current_depth > 1:
                legal_actions = self._order_moves(board, legal_actions, current_depth)
            else:
                move_scores = self._get_move_scores(board)
                legal_actions.sort(key=lambda m: move_scores.get(m, 0), reverse=True)
            
            search_complete = True
            for action in legal_actions:
                self.elapsed = time.time() - self.start_time
                if self.elapsed >= max_time:
                    if self.verbose:
                        print("Time nearly up before trying action, breaking out")
                    search_complete = False
                    break
                board.perform_move(action)
                score, new_pos = self.minimax(board, current_depth - 1, -1e9, 1e9, False)
                board.undo()
                
                pos += new_pos
                if score > depth_best_score:
                    depth_best_score = score
                    depth_best_action = action
                
                if score == self.W_WIN:
                    break
            
            if (search_complete) or (best_action is None):
                total_pos += pos
                best_action = depth_best_action
                best_score = depth_best_score
                
                self.best_moves_history[board.hash] = best_action
                
                if self.verbose:
                    print(f"[{'BLACK' if self.color == 1 else 'WHITE'}] Depth {current_depth}: Best move {best_action} with score {best_score} ({pos} positions)")
                
                if best_score >= self.W_WIN * 0.9:
                    if self.verbose:
                        print(f"[{'BLACK' if self.color == 1 else 'WHITE'}] Winning move found!")
                    break

                if best_score <= -self.W_WIN * 0.9:
                    if self.verbose:
                        print(f"[{'BLACK' if self.color == 1 else 'WHITE'}] Winning not possible :(")
                    break

                current_depth += 1
                self.elapsed = time.time() - self.start_time
        
        return best_action, total_pos

    # def get_best_action(self, board: Board, max_time: int):
    #     pos = 0
    #     best_score = -1e9
    #     best_action = None

    #     self.transposition.clear()
    #     # print(f"Legal Actions: {board.legal_actions}")
    #     # self.print_point_dict(board.libertydict)
    #     self.light_board = LightweightBoardHandler(board)
    #     legal_actions = self.light_board.get_legal_actions()
    #     for action in legal_actions:   
    #         # print("STORED_______________________")
    #         # for key,val in snap.libertydict['BLACK'].items(): 
    #         #     print(key , val)
        
    #         # for key,val in snap.libertydict['WHITE'].items(): 
    #         #     print(key , val)
    #         # print()

    #         self.light_board.perform_move(action)
    #         # print(self.light_board.parent)
    #         # successor = board.copy()
    #         # successor.put_stone(action, check_legal=False)
    #         # score, new_pos = self.minimax(successor, depth, maximizing_player=False)
    #         # self.print_point_dict(board.libertydict)
    #         score, new_pos = self.minimax(self.light_board, depth, alpha=-1e9, beta=1e9, maximizing_player=False)
    #         if(self.verbose):
    #             # self.light_board.perform_move(self.light_board.get_legal_actions()[0])
    #             print(action, score)         
    #             # self.light_board.undo()
    #         pos += new_pos
    #         if(score > best_score):
    #             best_score = score
    #             best_action = action
            
    #         if(score == self.W_WIN):
    #             break
    #         self.light_board.undo()
    #         # print("RESTORED: ")
    #         # print(self.light_board.board)
    #     # print("---------------------------------------------------------------------------")
    #     return best_action, pos

    # def print_point_dict(self, point_dict:PointDict):
    #     for key,val in point_dict.d['BLACK'].items(): 
    #         print(key , val)
        
    #     for key,val in point_dict.d['WHITE'].items(): 
    #         print(key , val)
    #     print()

    def get_action(self, board: Board):
        """Returns the best action using iterative deepening."""
        start_time = time.time()
        actions = board.legal_actions
        
        if not actions:
            return None
        
        self.best_moves_history.clear()
        
        self.light_board = LightweightBoardHandler(board)
        
        best_action, pos = self.iterative_deepening(self.light_board, self.MAX_TIME)
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"[{'BLACK' if self.color == 1 else 'WHITE'}] Total possibilities considered: {pos}")
            print(f"[{'BLACK' if self.color == 1 else 'WHITE'}] Best Move: {best_action}")
            print(f"[{'BLACK' if self.color == 1 else 'WHITE'}] Decision Time: {elapsed:.2f}s ({pos/elapsed:.0f} pos/sec)")
        
        if elapsed > 0.8 * self.MAX_TIME:
            print(f"[{'BLACK' if self.color == 1 else 'WHITE'}] !!! Took too much time: {elapsed:.2f}s ({pos/elapsed:.0f} pos/sec)")
            print(f"[{'BLACK' if self.color == 1 else 'WHITE'}] Possibilities considered: {pos}")
        
        return best_action

class Agent1v4:
    """A class to generate a random action for a Go board."""
    def __init__(self, color, max_depth = None, verbose = False):
        self.color = -1 if color == 'WHITE' else 1
        self.opponent_color = -self.color
        self.MAX_DEPTH = 5 if not max_depth else max_depth

        self.W_LIBERTIES = 10
        self.W_MOBILITY = 20
        self.W_ATARI = 50
        self.W_WIN = 1e5
        self.verbose = verbose

        self.light_board = None
        
        self.transposition = {}   

    def evaluate(self, board: LightweightBoardHandler, legal_action_count: int) -> int:
        if(board.draw): 
            return 0
        if(board.winner is not None):
            if(board.winner == self.color): return self.W_WIN
            else: return -self.W_WIN
        
        if(legal_action_count == 0): return 0
        
        own_atari = len(board.endangered_groups[self.color])
        opp_atari = len(board.endangered_groups[self.opponent_color])

        if((board.next == self.color) and (opp_atari > 0)):
            return self.W_WIN
        if((board.next == self.opponent_color) and (own_atari > 0)):
            return -self.W_WIN

        score = self.W_ATARI * (opp_atari - own_atari)
        score += self.W_MOBILITY * legal_action_count
        own_libs = len(board.possible_actions[self.color]) 
        opp_libs  = len(board.possible_actions[self.opponent_color])
        score += self.W_LIBERTIES * ((own_libs - opp_libs) / max(1, board.stone_count))  

        return score

    def minimax(self, board: LightweightBoardHandler, depth: int, alpha: int, beta: int, maximizing_player: bool) -> int:  
        # color_idx = 0 if board.next == 'BLACK' else 1
        key = (board.hash, board.next)
        
        if(key in self.transposition):
            return self.transposition[key]
        
        legal_actions = board.get_legal_actions()
        if((depth == 0) or (board.winner is not None)):
            score = self.evaluate(board, len(legal_actions))
            self.transposition[key] = (score,1)
            return score,1
        
        pos = 0   
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
            score, new_pos = self.minimax(self.light_board, depth, alpha=-1e9, beta=1e9, maximizing_player=False)
            if(self.verbose):
                print(action, score)         
            pos += new_pos
            if(score > best_score):
                best_score = score
                best_action = action
            
            if(score == self.W_WIN):
                break
            self.light_board.undo()
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
                print(f"!!! Took too much time: {time.time() - start_time}. ({pos/(time.time() - start_time)} possibilities/sec)")
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