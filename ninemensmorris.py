"""
Project: ninemensmorris.py
Author: Jonathan Walsh

The ninemensmorris.py project implements an environment
where human players and artificial intelligence agents
can play the game Nine Men's Morris.

Nine Men's Morris is a strategy board game for two players
dating at least to the Roman Empire. The rules of the game
can be found at:
https://en.wikipedia.org/wiki/Nine_Men%27s_Morris

Games can be played involving 0, 1, or 2 human players
and the corresponding number of AI agents controlling
the remaining players.

The AI agents make game decisions by evaluating and
exploring a game tree with the Iterative Deepening
Depth Limited Alpha-Beta Pruned Minimax algorithm.
They can also be set to select moves randomly by
altering their initializations in the game loops.

Note: The large number of possible moves in the end
game and the relatively slow running time of Python3
make search depths of 3 or greater take a long time.
The game is also solved to show that when 2 players
have their final 3 pieces remaining, both players
can indefinitely move the pieces to avoid a loss.
The AI's will sometimes continuously play the game
even though a human player would consider the game
a draw because neither strategy can win or lose.
"""

import copy
import random


"""The space class maintains the location of a space on the board,
its contents, and the spaces that it is connected to"""


class space:
    def __init__(self, space):
        # a tuple coordinate pair
        self.location = space
        # a list of the spaces that are connected
        self.adjacents = []
        # O for open, W for white piece, B for black piece
        self.content = 'O'

        # assign adjacent spaces for valid moves
        if space == ('a', 1):
            self.adjacents.append(('a', 4))
            self.adjacents.append(('d', 1))
        elif space == ('a', 4):
            self.adjacents.append(('a', 7))
            self.adjacents.append(('a', 1))
            self.adjacents.append(('b', 4))
        elif space == ('a', 7):
            self.adjacents.append(('a', 4))
            self.adjacents.append(('d', 7))
        elif space == ('b', 2):
            self.adjacents.append(('b', 4))
            self.adjacents.append(('d', 2))
        elif space == ('b', 4):
            self.adjacents.append(('b', 6))
            self.adjacents.append(('c', 4))
            self.adjacents.append(('b', 2))
            self.adjacents.append(('a', 4))
        elif space == ('b', 6):
            self.adjacents.append(('b', 4))
            self.adjacents.append(('d', 6))
        elif space == ('c', 3):
            self.adjacents.append(('d', 3))
            self.adjacents.append(('c', 4))
        elif space == ('c', 4):
            self.adjacents.append(('c', 5))
            self.adjacents.append(('c', 3))
            self.adjacents.append(('b', 4))
        elif space == ('c', 5):
            self.adjacents.append(('c', 4))
            self.adjacents.append(('d', 5))
        elif space == ('d', 1):
            self.adjacents.append(('a', 1))
            self.adjacents.append(('d', 2))
            self.adjacents.append(('g', 1))
        elif space == ('d', 2):
            self.adjacents.append(('d', 3))
            self.adjacents.append(('f', 2))
            self.adjacents.append(('b', 2))
            self.adjacents.append(('d', 1))
        elif space == ('d', 3):
            self.adjacents.append(('e', 3))
            self.adjacents.append(('d', 2))
            self.adjacents.append(('c', 3))
        elif space == ('d', 5):
            self.adjacents.append(('d', 6))
            self.adjacents.append(('e', 5))
            self.adjacents.append(('c', 5))
        elif space == ('d', 6):
            self.adjacents.append(('d', 7))
            self.adjacents.append(('f', 6))
            self.adjacents.append(('d', 5))
            self.adjacents.append(('b', 6))
        elif space == ('d', 7):
            self.adjacents.append(('a', 7))
            self.adjacents.append(('g', 7))
            self.adjacents.append(('d', 6))
        elif space == ('e', 3):
            self.adjacents.append(('d', 3))
            self.adjacents.append(('e', 4))
        elif space == ('e', 4):
            self.adjacents.append(('e', 5))
            self.adjacents.append(('f', 4))
            self.adjacents.append(('e', 3))
        elif space == ('e', 5):
            self.adjacents.append(('d', 5))
            self.adjacents.append(('e', 4))
        elif space == ('f', 2):
            self.adjacents.append(('d', 2))
            self.adjacents.append(('f', 4))
        elif space == ('f', 4):
            self.adjacents.append(('f', 6))
            self.adjacents.append(('g', 4))
            self.adjacents.append(('f', 2))
            self.adjacents.append(('e', 4))
        elif space == ('f', 6):
            self.adjacents.append(('d', 6))
            self.adjacents.append(('f', 4))
        elif space == ('g', 1):
            self.adjacents.append(('d', 1))
            self.adjacents.append(('g', 4))
        elif space == ('g', 4):
            self.adjacents.append(('g', 7))
            self.adjacents.append(('g', 1))
            self.adjacents.append(('f', 4))
        elif space == ('g', 7):
            self.adjacents.append(('d', 7))
            self.adjacents.append(('g', 4))


"""
The board class maintains the spaces that make up the board
as well as the pieces of both players
"""


class board:
    # defines the number of previous actions
    # that are tracked to determine a draw loop
    MAX_PREVIOUS_ACTIONS = 16

    def __init__(self):
        # lists containing the coordinates of the pieces
        # on the board
        self.white_pieces = []
        self.black_pieces = []

        # tracks whether the board's previous actions
        # indicate that the first half and last half
        # of them have been repeated indicating a loop
        # of repetition. This is treated as a draw.
        self.repetition_draw = False
        # a list of the moves registered to the board
        self.previous_actions = []
        # the turn of the game
        self.turn = 1
        # a measure of the value of a state
        self.utility = 0
        # Labels and initializes the spaces on the board
        # specified by the rules of the game
        self.spaces = {
            ('a', 1): space(('a', 1)),
            ('a', 4): space(('a', 4)),
            ('a', 7): space(('a', 7)),
            ('b', 2): space(('b', 2)),
            ('b', 4): space(('b', 4)),
            ('b', 6): space(('b', 6)),
            ('c', 3): space(('c', 3)),
            ('c', 4): space(('c', 4)),
            ('c', 5): space(('c', 5)),
            ('d', 1): space(('d', 1)),
            ('d', 2): space(('d', 2)),
            ('d', 3): space(('d', 3)),
            ('d', 5): space(('d', 5)),
            ('d', 6): space(('d', 6)),
            ('d', 7): space(('d', 7)),
            ('e', 3): space(('e', 3)),
            ('e', 4): space(('e', 4)),
            ('e', 5): space(('e', 5)),
            ('f', 2): space(('f', 2)),
            ('f', 4): space(('f', 4)),
            ('f', 6): space(('f', 6)),
            ('g', 1): space(('g', 1)),
            ('g', 4): space(('g', 4)),
            ('g', 7): space(('g', 7))
        }

    """
    Prints the board's current state to the standard output
    """
    def display(self):
        print('Turn: %s' % (self.turn))
        if self.turn % 2 == 0:
            turn_color = 'Black'
        else:
            turn_color = 'White'

        print('White Pieces: ', self.white_pieces)
        print('Black Pieces: ', self.black_pieces)
        print('%s\'s Move' % (turn_color))
        print('7   %s - - - - - %s - - - - - %s' % (
            self.spaces[('a', 7)].content,
            self.spaces[('d', 7)].content,
            self.spaces[('g', 7)].content))
        print('    |           |           |')
        print('6   |   %s - - - %s - - - %s   |' % (
            self.spaces[('b', 6)].content,
            self.spaces[('d', 6)].content,
            self.spaces[('f', 6)].content))
        print('    |   |       |       |   |')
        print('5   |   |   %s - %s - %s   |   |' % (
            self.spaces[('c', 5)].content,
            self.spaces[('d', 5)].content,
            self.spaces[('e', 5)].content))
        print('    |   |   |       |   |   |')
        print('4   %s - %s - %s       %s - %s - %s' % (
            self.spaces[('a', 4)].content,
            self.spaces[('b', 4)].content,
            self.spaces[('c', 4)].content,
            self.spaces[('e', 4)].content,
            self.spaces[('f', 4)].content,
            self.spaces[('g', 4)].content))
        print('    |   |   |       |   |   |')
        print('3   |   |   %s - %s - %s   |   |' % (
            self.spaces[('c', 3)].content,
            self.spaces[('d', 3)].content,
            self.spaces[('e', 3)].content))
        print('    |   |       |       |   |')
        print('2   |   %s - - - %s - - - %s   |' % (
            self.spaces[('b', 2)].content,
            self.spaces[('d', 2)].content,
            self.spaces[('f', 2)].content))
        print('    |           |           |')
        print('1   %s - - - - - %s - - - - - %s' % (
            self.spaces[('a', 1)].content,
            self.spaces[('d', 1)].content,
            self.spaces[('g', 1)].content))
        print('')
        print('    a   b   c   d   e   f   g')
        print('')

        if self.previous_actions:
            print("Previous Action:")
            self.previous_actions[-1].display()
            print("\n")

    """
    Adds a piece of the given color on the board at the coordinates of space
    """
    def place(self, color, space):
        self.spaces[space].content = color
        if color == 'W':
            # tmp assures that the function is a deep copy
            tmp = self.white_pieces[:]
            tmp.append(space)
            self.white_pieces = tmp
        else:
            # tmp assures that the function is a deep copy
            tmp = self.black_pieces[:]
            tmp.append(space)
            self.black_pieces = tmp

        self.turn += 1

        return self.is_mill(color, space)

    """
    Moves the piece located at space1 to the position space2
    """
    def move(self, color, space1, space2):
        self.spaces[space2].content = color
        self.spaces[space1].content = 'O'
        if color == 'W':
            # tmp assures that the function is a deep copy
            tmp = self.white_pieces[:]
            tmp.remove(space1)
            tmp.append(space2)
            self.white_pieces = tmp
        else:
            # tmp assures that the function is a deep copy
            tmp = self.black_pieces[:]
            tmp.remove(space1)
            tmp.append(space2)
            self.black_pieces = tmp

        self.turn += 1

        return self.is_mill(color, space2)

    """
    Removes the piece located at the coordinates of space
    """
    def remove(self, space):
        if self.spaces[space].content == 'W':
            # tmp assures that the function is a deep copy
            tmp = self.white_pieces[:]
            tmp.remove(space)
            self.white_pieces = tmp
        else:
            # tmp assures that the function is a deep copy
            tmp = self.black_pieces[:]
            tmp.remove(space)
            self.black_pieces = tmp
        self.spaces[space].content = 'O'

    def add_action(self, action):
        if len(self.previous_actions) < self.MAX_PREVIOUS_ACTIONS:
            # tmp assures that the function is a deep copy
            tmp = self.previous_actions[:]
            tmp.append(action)
            self.previous_actions = tmp
        else:
            # tmp assures that the function is a deep copy
            tmp = self.previous_actions[:]
            tmp.append(action)
            tmp.pop(0)
            self.previous_actions = tmp

    """
    Returns true if the space provided forms 3 in row with other
    pieces of the same color forming a mill. Returns false otherwise.
    """
    def is_mill(self, color, space):
        # locate pieces of the same color in adjacent spaces to space
        friendly_adj = [s for s in self.spaces[space].adjacents
                        if self.spaces[s].content == color]
        if len(friendly_adj) > 1:
            # check middle mill (ie one friendly piece on opposite sides)
            for s in friendly_adj:
                s = self.spaces[s]
                for comp in friendly_adj:
                    comp = self.spaces[comp]
                    if s is not comp:
                        if s.location[0] == comp.location[0] or s.location[1] == comp.location[1]:
                            return True
        # check long mill
        # (ie there is another friendly piece 2 spaces away in the line)
        for s in friendly_adj:
            extra_adj = [x for x in self.spaces[s].adjacents
                         if self.spaces[x].content == color and x != space]
            for x in extra_adj:
                if x[0] == space[0] or x[1] == space[1]:
                    return True
        return False

    """
    Returns the results of the is_mill function based on the board
    that results from the provided move. Allows foresight for mill
    locating without committing to a full result call.
    """
    def will_mill(self, color, move):
        test_board = self.result(move, False, None)
        if test_board.is_mill(color, move.new_coord):
            return True
        else:
            return False

    """
    Generates all possible moves for the current turn player. If
    test_existence is true, the function exits as soon as a valid
    move is found to avoid full calculation when not needed.
    """
    def generate_moves(self, test_existence=False):
        actions = []
        # determine the current controller color
        if self.turn % 2 == 0:
            current_color = 'B'
        else:
            current_color = 'W'

        if current_color == 'W':
            my_pieces = self.white_pieces
            opp_pieces = self.black_pieces
        else:
            my_pieces = self.black_pieces
            opp_pieces = self.white_pieces

        # generate piece placing moves in the opening of the game
        if self.turn < 11:
            open_spaces = [s for s in self.spaces
                           if self.spaces[s].content == 'O']
            for space in open_spaces:
                # determine if placing a piece causes a mill and split the
                # action into the possible mill removal targets if it mills
                if self.will_mill(current_color,
                                  action(current_color, None, space, None)):
                    for target in opp_pieces:
                        new_action = action(current_color, None, space, target)
                        actions.append(new_action)
                        if test_existence:
                            return actions
                else:
                    new_action = action(current_color, None, space, None)
                    actions.append(new_action)
                    if test_existence:
                        return actions

        # generate single space movements in the middle game
        elif len(my_pieces) > 3:
            for piece in my_pieces:
                open_spaces = [s for s in self.spaces[piece].adjacents
                               if self.spaces[s].content == 'O']
                for space in open_spaces:
                    # determine if the piece movement causes a mill and split
                    # the action into the possible mill removal targets if it
                    # mills
                    if self.will_mill(
                            current_color,
                            action(current_color, piece, space, None)):
                        for target in opp_pieces:
                            new_action = action(
                                current_color,
                                piece, space,
                                target)
                            actions.append(new_action)
                            if test_existence:
                                return actions
                    else:
                        new_action = action(current_color, piece, space, None)
                        actions.append(new_action)
                        if test_existence:
                            return actions

        # generate fly moves (can move anywhere on the board) if the player
        # is in the end game with their last pieces remaining
        elif len(my_pieces) == 3:
            for piece in my_pieces:
                open_spaces = [s for s in self.spaces
                               if self.spaces[s].content == 'O']
                for space in open_spaces:
                    # determine if the piece movement causes a mill and split
                    # the action into the possible mill removal targets
                    # if it mills
                    if self.will_mill(
                            current_color,
                            action(current_color, piece, space, None)):
                        for target in opp_pieces:
                            new_action = action(
                                current_color,
                                piece,
                                space,
                                target)
                            actions.append(new_action)
                            if test_existence:
                                return actions
                    else:
                        new_action = action(current_color, piece, space, None)
                        actions.append(new_action)
                        if test_existence:
                            return actions

        # return the list of actions that are found
        return actions

    """
    Returns the resulting board state of the provided action being taken.
    calc_util determines whether the new state will calculate its utility
    """
    def result(self, action, calc_util, max_color):
        # create an independent state to return
        new_board = copy.deepcopy(self)
        # if there is an old coordinate then it is a move not a placement
        if action.old_coord:
            new_board.move(action.color, action.old_coord, action.new_coord)
        # if there is no old coordinate the move is a piece placement
        else:
            new_board.place(action.color, action.new_coord)
        # checks if a piece is to be removed because of a mill
        if action.mill_removal_target:
            new_board.remove(action.mill_removal_target)
        # record the new action taken
        new_board.add_action(action)
        # check if the action causes a draw by looping
        new_board.repetition_draw = new_board.is_repetitive_draw()
        # calculate and store the board state utility if needed
        if calc_util:
            self.calculate_utility(max_color)

        return new_board

    """
    returns the utility value of the board state for the max_color
    """
    def calculate_utility(self, max_color):
        NO_MOVES_VALUE = 10000
        TWO_PIECES_REMAINING = 20000
        DRAW = 1000
        PIECE_VALUE = 200
        MILL_FORMED = 300
        tmp = 0

        # determine which piece's are to move next
        if self.turn % 2 == 0:
            current_color = 'B'
        else:
            current_color = 'W'

        # label the pieces based on the max player identity
        if max_color == 'W':
            my_pieces = self.white_pieces
            opp_pieces = self.black_pieces
        else:
            my_pieces = self.black_pieces
            opp_pieces = self.white_pieces

        # check if there are moves possible
        actions = self.generate_moves(True)

        # if the current player has no moves it is a loss
        if len(actions) == 0:
            if current_color == max_color:
                tmp -= NO_MOVES_VALUE
            else:
                tmp += NO_MOVES_VALUE

        # if a player has only 2 pieces, they cannot form
        # a mill and therefore lose
        if len(my_pieces) == 2 and self.turn > 10:
            tmp -= TWO_PIECES_REMAINING
        elif len(opp_pieces) == 2 and self.turn > 10:
            tmp += TWO_PIECES_REMAINING

        # check if the previous move was a mill and reward
        # the player that formed it
        if len(self.previous_actions) > 1:
            if self.previous_actions[-1].mill_removal_target:
                if self.previous_actions[-1].color == max_color:
                    tmp += MILL_FORMED
                else:
                    tmp -= MILL_FORMED

        # count the difference in pieces for the players
        material_difference = len(my_pieces) - len(opp_pieces)

        # weight the difference in pieces by their value
        tmp += material_difference * PIECE_VALUE

        # a draw is reported at a utility of 0 because it ends
        # the game but neither player wins
        if self.repetition_draw:
            tmp = 0

        # save the resulting sum of the utility factors and return it
        self.utility = tmp

        return self.utility

    """
    Returns true if a state is an ended game and false otherwise
    """
    def is_terminal(self):
        if len(self.white_pieces) == 2 or len(self.black_pieces) == 2:
            return True
        actions = self.generate_moves(True)
        if len(actions) > 0:
            return False
        else:
            return True
    """
    Returns true if the first half and last half of previous_actions
    are the same. This indicates that the two players are repeating
    the same loop of actions. This is counted as a draw for the game.
    Returns false otherwise.
    """
    def is_repetitive_draw(self):
        if len(self.previous_actions) < self.MAX_PREVIOUS_ACTIONS:
            return False
        half = int(len(self.previous_actions)/2)
        front = self.previous_actions[:half]
        back = self.previous_actions[half:]

        intersection = [x for x, y in zip(front, back) if x == y]

        if len(intersection) == len(self.previous_actions)/2:
            return True
        return False


"""
The AI class maintains its color and decision making functions to
select actions on its turn
"""


class AI:
    def __init__(self, color, depth, random=False):
        # piece color
        self.color = color
        # A limit on how many turns ahead the AI looks
        self.MAX_DEPTH = depth
        # a boolean determining whether the AI chooses moves randomly or not
        self.random = random

    """
    Iterative depth limited alpha beta pruned minimax algorithm:
    Returns a move that optimizes its utility at the MAX_DEPTH
    """
    def ID_AB_minimax(self, board):
        iterative_depth = 1

        # deepen the search depth of AB_minimax until the max depth is
        # explored
        while iterative_depth <= self.MAX_DEPTH:
            print('Iterative Depth: ', iterative_depth)
            result = action(self.AB_minimax(board, iterative_depth))
            iterative_depth += 1

        # return the best action found to the best route to the max depth
        return result

    """
    Returns an optimal move at the provided depth
    """
    def AB_minimax(self, board, depth):
        alpha = -100000
        beta = 100000

        possible_moves = board.generate_moves()

        max_action = possible_moves[0]

        if depth == 1:
            calc_util = True
        else:
            calc_util = False

        for move in possible_moves:
            current_utility = self.AB_min_value(
                board.result(move, calc_util, self.color),
                depth - 1,
                alpha,
                beta)
            # if a better action is found update the alpha bound to allow
            # pruning and store the best action
            if current_utility > alpha:
                alpha = current_utility
                max_action = action(move)
        return max_action

    """
    Returns the utility value of the best action from the perspective
    of the max player (the calling AI)
    """
    def AB_max_value(self, board, depth, alpha, beta):
        calc_util = False
        # prepare the result function to find the utility of the final layer
        if depth == 1:
            calc_util = True
        # terminate search if the state is an end state of the game
        if board.is_terminal():
            return board.calculate_utility(self.color)
        # return the utility value at the max depth to end recursion
        if depth == 0:
            return board.utility

        possible_moves = board.generate_moves()

        for move in possible_moves:
            current_utility = self.AB_min_value(
                board.result(move, calc_util, self.color),
                depth - 1,
                alpha,
                beta)
            # update alpha bound if a better utility is found
            if current_utility > alpha:
                alpha = current_utility
            # if the beta and alpha bounds cross over,
            # prune the rest of the search
            if beta <= alpha:
                return alpha
        return alpha

    """
    Returns the utility value of the best action from the perspective
    of the min player (the opponent of the calling AI)
    """
    def AB_min_value(self, board, depth, alpha, beta):
        calc_util = False
        # prepare the result function to find the utility of the final layer
        if depth == 1:
            calc_util = True
        # terminate search if the state is an end state of the game
        if board.is_terminal():
            return board.calculate_utility(self.color)
        # return the utility value at the max depth to end recursion
        if depth == 0:
            return board.utility

        possible_moves = board.generate_moves()

        for move in possible_moves:
            current_utility = self.AB_max_value(
                board.result(move, calc_util, self.color),
                depth - 1,
                alpha,
                beta)
            # update beta bound if a lower utility is found
            if current_utility < beta:
                beta = current_utility
            # if the beta and alpha bounds cross over,
            # prune the rest of the search
            if beta <= alpha:
                return beta
        return beta

    """
    If the AI acts randomly, a move is selected from the possible actions of
    the current board. Otherwise, the move is selected by ID_AB_minimax.
    The board is then updated with the choosen move and returns the move.
    """
    def take_turn(self, board):
        board.display()
        # random move selection
        if self.random:
            actions = board.generate_moves()
            move = actions[random.randint(0, len(actions) - 1)]
        # ID_AB_minimax move selection
        else:
            move = self.ID_AB_minimax(board)
        # update board state
        if move.old_coord:
            mill = board.move(self.color, move.old_coord, move.new_coord)
            if mill:
                board.remove(move.mill_removal_target)
        else:
            mill = board.place(self.color, move.new_coord)
            if mill:
                board.remove(move.mill_removal_target)
        return move


"""
The player class allows human control of a player in the game.
"""


class player:

    def __init__(self, color):
        # the color of the player's pieces
        self.color = color

    """
    Prompts the user to enter a space to place a new piece via
    the standard input
    """
    def place(self, board):
        while True:
            board.display()
            print('Phase 1: Placing Pieces\n\nEnter Placement Column (a-g):')
            col = input()
            print('Enter Placement Row (1-7):')
            row = int(input())
            target = (col, row)
            if target in board.spaces:
                if board.spaces[target].content == 'O':
                    mill = board.place(self.color, target)
                    if mill:
                        removal = self.remove(board)
                        return action(self.color, None, target, removal)
                    else:
                        return action(self.color, None, target, None)
            print('Invalid position to place piece. Select an open space.\n\n')

    """
    Prompts the user to enter a piece location and a new target space
    to move it to. The move must be to an adjacent space.
    """
    def move(self, board):
        while True:
            board.display()
            print('Phase 2: Standard Movement\n\nEnter' +
                  ' Target Piece Column (a-g):')
            col = input()
            print('Enter Target Piece Row (1-7):')
            row = int(input())
            start = (col, row)
            if start in board.spaces:
                if board.spaces[start].content == self.color:
                    possible = [s for s in board.spaces[start].adjacents
                                if board.spaces[s].content == 'O']
                    if possible:
                        print('Enter Movement Target Column (a-g):')
                        col = input()
                        print('Enter Movement Target Row (1-7):')
                        row = int(input())
                        target = (col, row)
                        if target in board.spaces:
                            if target in possible:
                                mill = board.move(self.color, start, target)
                                if mill:
                                    removal = self.remove(board)
                                    return action(
                                        self.color,
                                        start,
                                        target,
                                        removal)
                                else:
                                    return action(
                                        self.color,
                                        start,
                                        target,
                                        None)
                                break
                            else:
                                print('Invalid location. ' +
                                      ' Space is not open or not adjacent.')
                        else:
                            print('Invalid board coordinates.')
                    else:
                        print('Invalid location. No adjacent spaces open.')
                else:
                    print('Invalid location. ' +
                          'No controlled piece at coordinates.')
            else:
                print('Invalid location. No controlled piece at coordinates.')

    """
    Prompts the user to select a piece and an open space to move it to.
    This move type can be used when a player has only 3 pieces remaining.
    """
    def fly_move(self, board):
        while True:
            board.display()
            print('Phase 2: Flying Movement\n\nEnter ' +
                  'Target Piece Column (a-g):')
            col = input()
            print('Enter Target Piece Row (1-7):')
            row = int(input())
            start = (col, row)
            if start in board.spaces:
                if board.spaces[start].content == self.color:
                    possible = [s for s in board.spaces
                                if board.spaces[s].content == 'O']
                    if possible:
                        print('Enter Movement Target Column (a-g):')
                        col = input()
                        print('Enter Movement Target Row (1-7):')
                        row = int(input())
                        target = (col, row)
                        if target in board.spaces:
                            if target in possible:
                                mill = board.move(self.color, start, target)
                                if mill:
                                    removal = self.remove(board)
                                    return action(
                                        self.color,
                                        start,
                                        target,
                                        removal)
                                else:
                                    return action(
                                        self.color,
                                        start,
                                        target,
                                        None)
                                break
                            else:
                                print('Invalid location. ' +
                                      'Space is not open or not adjacent.')
                        else:
                            print('Invalid board coordinates.')
                    else:
                        print('Invalid location. No adjacent spaces open.')
                else:
                    print('Invalid location. ' +
                          'No controlled piece at coordinates.')
            else:
                print('Invalid location. No controlled piece at coordinates.')

    """
    Prompts the user to select an opponent's piece to remove after a mill.
    """
    def remove(self, board):
        while True:
            board.display()
            print('Mill Formed\n\nEnter Target Opponent ' +
                  'Piece Column to Remove (a-g):')
            col = input()
            print('Enter Target Opponent Piece Row to Remove (1-7):')
            row = int(input())
            target = (col, row)
            if target in board.spaces:
                if board.spaces[target].content != self.color and board.spaces[target].content != 'O':
                    board.remove(target)
                    return target
                else:
                    print('Invalid target. Not an enemy piece.')
            else:
                print('Invalid board coordinates.')


"""
The action class maintains information about the movement of a piece
"""


class action:
    """
    Initializes the action with the provided characteristics
    """
    # candidate argument sets:
    # (color, old, new, mill_removal_target)
    # or (action)
    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            self.color = args[0]
            self.old_coord = args[1]
            self.new_coord = args[2]
            self.mill_removal_target = args[3]
        else:
            self.color = args[0].color
            self.old_coord = args[0].old_coord
            self.new_coord = args[0].new_coord
            self.mill_removal_target = args[0].mill_removal_target

    """
    Prints the action to the standard output formatted
    based on what optional events occured
    """
    def display(self):
        if self.old_coord is None:
            print('Action: Place %r piece at %r.' %
                  (self.color, self.new_coord))
            if self.mill_removal_target:
                print('Mill removing %s' % (self.mill_removal_target, ))
        else:
            print('Action: Move %r piece at %r to %r.' %
                  (self.color, self.old_coord, self.new_coord))
            if self.mill_removal_target:
                print('Mill removing %s' % (self.mill_removal_target, ))


"""
Executes a game with 2 human controlled players
"""


def two_player_game():

    b = board()
    p1 = player('W')
    p2 = player('B')

    # Phase 1: Placing Pieces
    for i in range(1, 6):
        print('\nPlayer 1 Turn\n')
        move = p1.place(b)
        b.add_action(move)
        print('\nPlayer 2 Turn\n')
        move = p2.place(b)
        b.add_action(move)

    # Phase 2: Moving Pieces
    # (Phase 3 after one player has 3 pieces): Flying Pieces
    while not b.is_terminal():
        print('\nPlayer 1 Turn\n')
        if len(b.white_pieces) > 3:
            move = p1.move(b)
            b.add_action(move)
        else:
            move = p1.fly_move(b)
            b.add_action(move)

        if b.is_terminal():
            break

        print('\nPlayer 2 Turn\n')
        if len(b.black_pieces) > 3:
            move = p2.move(b)
            b.add_action(move)
        else:
            move = p2.fly_move(b)
            b.add_action(move)

    b.display()
    if len(b.white_pieces) == 2:
        print('Player 2 Wins')
    if len(b.black_pieces) == 2:
        print('Player 1 Wins')
    if b.is_repetitive_draw():
        print('Draw because of repeated set of moves')


"""
Executes a game with one human player and one AI player
"""


def one_player_game():
    b = board()
    p1 = player('W')
    ai = AI('B', 3)

    # Phase 1: Placing Pieces
    for i in range(1, 6):
        print('\nPlayer 1 Turn\n')
        move = p1.place(b)
        b.add_action(move)
        print('\nAI Turn\n')
        move = ai.take_turn(b)
        b.add_action(move)

    # Phase 2: Moving Pieces
    # (Phase 3 after one player has 3 pieces): Flying Pieces
    while not b.is_terminal():
        print('\nPlayer 1 Turn\n')
        if len(b.white_pieces) > 3:
            move = p1.move(b)
            b.add_action(move)
        else:
            move = p1.fly_move(b)
            b.add_action(move)

        if b.is_terminal():
            break

        print('\nAI Turn\n')
        move = ai.take_turn(b)
        b.add_action(move)

    b.display()
    if len(b.white_pieces) == 2:
        print('AI Wins')
    if len(b.black_pieces) == 2:
        print('Player 1 Wins')
    if b.is_repetitive_draw():
        print('Draw because of repeated set of moves')


"""
Executes a game with 2 AI controlled players
"""


def no_player_game():
    b = board()
    ai_1 = AI('W', 3)
    ai_2 = AI('B', 3)

    # Phase 1: Placing Pieces
    for i in range(1, 6):
        print('\nAI 1 Turn\n')
        move = ai_1.take_turn(b)
        b.add_action(move)
        print('\nAI 2 Turn\n')
        move = ai_2.take_turn(b)
        b.add_action(move)

    # Phase 2: Moving Pieces
    # (Phase 3 after one player has 3 pieces): Flying Pieces
    while not b.is_terminal():
        print('\nAI 1 Turn\n')
        move = ai_1.take_turn(b)
        b.add_action(move)

        if b.is_terminal():
            break

        print('\nAI 2 Turn\n')
        move = ai_2.take_turn(b)
        b.add_action(move)

    b.display()
    if len(b.white_pieces) == 2:
        print('AI 2 Wins')
    if len(b.black_pieces) == 2:
        print('AI 1 Wins')
    if b.is_repetitive_draw():
        print('Draw because of repeated set of moves')


"""
Prompts the user for what type of players should control the game
and executes
"""
if __name__ == '__main__':
    print('1) Two Player Game')
    print('2) One Player Game')
    print('3) No  Player Game')
    print('Select Game Mode:')
    while True:
        game_mode = int(input())

        if game_mode == 1:
            two_player_game()
            break
        elif game_mode == 2:
            one_player_game()
            break
        elif game_mode == 3:
            no_player_game()
            break
        else:
            print('Invalid game mode. Select options between 1 and 3.')
