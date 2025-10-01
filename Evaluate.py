#!/usr/bin/env python
from game.go import Board, opponent_color
from game.ui import UI
import pygame
import time
from group1 import Agent1v6, Agent1v5, Agent1v4, Agent1v3, Agent1v2, Agent1v1
from group2 import Agent2
from os.path import join
from argparse import ArgumentParser

from tqdm import tqdm

class Match:
    def __init__(self, agent_black=None, agent_white=None, gui=True, dir_save=None):
        self.agent_black = agent_black
        self.agent_white = agent_white

        self.board = Board(next_color='BLACK')

        gui = gui if agent_black and agent_white else True
        self.ui = UI() if gui else None
        self.dir_save = dir_save

        # Metadata

        self.time_elapsed = None

    @property
    def winner(self):
        return self.board.winner

    @property
    def next(self):
        return self.board.next

    @property
    def counter_move(self):
        return self.board.counter_move

    def start(self):
        if self.ui:
            self._start_with_ui()
        else:
            self._start_without_ui()

    def _start_with_ui(self):
        """Start the game with GUI."""
        self.ui.initialize()
        self.time_elapsed = time.time()

        # First move is fixed on the center of board
        first_move = (10, 10)
        self.board.put_stone(first_move, check_legal=False)
        self.ui.draw(first_move, opponent_color(self.board.next))

        # Take turns to play move
        while self.board.winner is None:
            # time.sleep(1)
            if self.board.next == 'BLACK':
                point = self.perform_one_move(self.agent_black)
            else:
                point = self.perform_one_move(self.agent_white)
            # print("!!!", self.board.next, self.board.legal_actions)
            # Check if action is legal
            if point not in self.board.legal_actions:
                continue
            # print("$$$", self.board.next)

            # Apply action
            prev_legal_actions = self.board.legal_actions.copy()
            self.board.put_stone(point, check_legal=True)
            # Remove previous legal actions on board
            for action in prev_legal_actions:
                self.ui.remove(action)
            # Draw new point
            self.ui.draw(point, opponent_color(self.board.next))
            # Update new legal actions and any removed groups
            if self.board.winner:
                for group in self.board.removed_groups:
                    for point in group.points:
                        self.ui.remove(point)
                if self.board.end_by_no_legal_actions:
                    print('Game ends early (no legal action is available for %s)' % self.board.next)
            else:
                for action in self.board.legal_actions:
                    self.ui.draw(action, 'BLUE', 8)

        self.time_elapsed = time.time() - self.time_elapsed
        if self.dir_save:
            path_file = join(self.dir_save, 'go_' + str(time.time()) + '.jpg')
            self.ui.save_image(path_file)
            print('Board image saved in file ' + path_file)

    def _start_without_ui(self):
        """Start the game without GUI. Only possible when no human is playing."""
        # First move is fixed on the center of board
        self.time_elapsed = time.time()
        first_move = (10, 10)
        self.board.put_stone(first_move, check_legal=False)

        # Take turns to play move
        while self.board.winner is None:
            if self.board.next == 'BLACK':
                point = self.perform_one_move(self.agent_black)
            else:
                point = self.perform_one_move(self.agent_white)

            # Apply action
            self.board.put_stone(point, check_legal=True)

        if self.board.end_by_no_legal_actions:
            print('Game ends early (no legal action is available for %s)' % self.board.next)

        self.time_elapsed = time.time() - self.time_elapsed

    def perform_one_move(self, agent):
        if agent:
            return self._move_by_agent(agent)
        else:
            return self._move_by_human()

    def _move_by_agent(self, agent):
        if self.ui:
            pygame.time.wait(100)
            pygame.event.get()
        return agent.get_action(self.board)

    def _move_by_human(self):
        while True:
            pygame.time.wait(100)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and self.ui.outline.collidepoint(event.pos):
                        x = int(round(((event.pos[0] - 5) / 40.0), 0))
                        y = int(round(((event.pos[1] - 5) / 40.0), 0))
                        point = (x, y)
                        stone = self.board.exist_stone(point)
                        if not stone:
                            return point

def main():

    SAMPLE_SIZE = 1
    black_wins = 0
    white_wins = 0
    draws = 0
    for i in tqdm(range(SAMPLE_SIZE), desc="Progress: ", unit="matches"):
        #agent_white = Agent1v5(color="WHITE", time_limit=10.0, verbose=True) #Agent1v4('WHITE', max_depth=5, verbose=True)
        #agent_white = Agent1v4('WHITE', max_depth=7, verbose=True)
        agent_white = Agent1v6('WHITE', max_time=15, verbose=False)
        #agent_white = Agent2('WHITE')
        #agent_black = Agent1v5(color="BLACK", time_limit=10.0, verbose=False) #Agent1v4('BLACK', max_depth=5)
        #agent_black = Agent1v4('BLACK', max_depth=7, verbose=True)
        #agent_black = Agent1v6('BLACK', max_time=15, verbose=False)
        agent_black = None #Agent2('BLACK')
        
        dir_save = 'game/images'
        match = Match(agent_black=agent_black, agent_white=agent_white, gui=True, dir_save=dir_save)

        match.start()

        if(match.board.end_by_no_legal_actions): draws += 1
        elif(match.winner == 'WHITE'): white_wins += 1
        else: black_wins += 1
        # print(match.winner + ' wins!')
        # print('Match ends in ' + str(match.time_elapsed) + ' seconds')
        # print('Match ends in ' + str(match.counter_move) + ' moves')

    print(f"White win rate: {(white_wins/SAMPLE_SIZE) * 100}%")
    print(f"Black win rate: {(black_wins/SAMPLE_SIZE) * 100}%")
    print(f"Draw rate: {(draws/SAMPLE_SIZE) * 100}%")


if __name__ == '__main__':
    main()
