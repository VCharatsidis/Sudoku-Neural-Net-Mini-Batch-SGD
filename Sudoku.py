# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 09:35:03 2018

@author: vcharatsidis
"""
import random
import copy
import numpy as np

class SolvedSudoku:

    def __init__(self, num_boards):
        self.boards = []
        self.create_boards(num_boards)
        self.size = 9

    def free_boxes(self):
        free_boxes = []
        for row in range(self.size):
            for col in range(self.size):
                free_boxes.append(row * self.size + col)

        return free_boxes

    def create_boards(self, num_boards):
        for i in range(num_boards):
            self.boards.append(self.construct_puzzle_solution())

    def board_reduction(self, reduction):
        solution = (9, 9)
        solution = np.zeros(solution).astype(int)
        board = random.choice(self.boards)
        reduced_board = copy.deepcopy(board)
        free_boxes = self.free_boxes()
        indexes_to_erase = random.sample(range(0, len(free_boxes)), reduction)

        for index in indexes_to_erase:
            to_erase = free_boxes[index]
            col = to_erase % self.size
            row = to_erase // self.size

            solution[row][col] = reduced_board[row][col]
            reduced_board[row][col] = 0

        np.reshape(solution, 81)
        rows, cols = np.nonzero(solution)

        return reduced_board, solution[rows, cols]

    def board_to_row(self, board):
        row_board = []

        for row in range(self.size):
            for col in range(self.size):
                row_board.append(board[row][col])

        row_board_column = np.array(row_board).reshape(1, 81)
        return row_board_column

    def construct_puzzle_solution(self):
        # Loop until we're able to fill all 81 cells with numbers, while
        # satisfying the constraints above.
        while True:
            try:
                puzzle = [[0] * 9 for i in range(9)]  # start with blank puzzle
                rows = [set(range(1, 10)) for i in range(9)]  # set of available
                columns = [set(range(1, 10)) for i in range(9)]  # numbers for each
                squares = [set(range(1, 10)) for i in range(9)]  # row, column and square
                for i in range(9):
                    for j in range(9):
                        # pick a number for cell (i,j) from the set of remaining available numbers
                        choices = rows[i].intersection(columns[j]).intersection(squares[(i // 3) * 3 + j // 3])
                        choice = random.choice(list(choices))

                        puzzle[i][j] = choice

                        rows[i].discard(choice)
                        columns[j].discard(choice)
                        squares[(i // 3) * 3 + j // 3].discard(choice)

                # success! every cell is filled.
                return puzzle

            except IndexError:
                # if there is an IndexError, we have worked ourselves in a corner (we just start over)
                pass

