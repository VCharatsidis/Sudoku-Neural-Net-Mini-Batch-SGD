# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 09:35:03 2018

@author: vcharatsidis
"""
import random
import copy
import numpy as np

class SolvedSudoku:

    def __init__(self, given_board, fixed_boxes):
        self.board = given_board
        self.fixed = fixed_boxes
        self.size = len(given_board)
        self.solution = (9, 9)
        self.solution = np.zeros(self.solution).astype(int)

    def free_boxes(self):
        free_boxes = []

        for row in range(self.size):
            for col in range(self.size):
                free_boxes.append(row * self.size + col)

        return free_boxes

    def board_reduction(self, reduction):
        self.solution = (9, 9)
        self.solution = np.zeros(self.solution).astype(int)
        reduced_board = copy.deepcopy(self.board)
        free_boxes = self.free_boxes()
        indexes_to_erase = random.sample(range(0, len(free_boxes)), reduction)

        for index in indexes_to_erase:
            to_erase = free_boxes[index]
            col = to_erase % self.size
            row = (to_erase) // self.size

            self.solution[row][col] = reduced_board[row][col]
            reduced_board[row][col] = 0

        return reduced_board

    def board_to_row(self, board):
        row_board = []

        for row in range(self.size):
            for col in range(self.size):
                row_board.append(board[row][col])

        row_board_column = np.array(row_board).reshape(1, 81)
        return row_board_column

