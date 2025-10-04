from game.go import Board
import random
from game.go import BOARD_SIZE
import time
import numpy as np

import random

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

class Agent1:
    def __init__(self, color: str, max_time: int = 0.5, verbose: bool = False):
        self.color = -1 if color.upper() == 'WHITE' else 1
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