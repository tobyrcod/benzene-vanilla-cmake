#----------------------------------------------------------------------------
# Plays a game of Hex.
#----------------------------------------------------------------------------

import fcntl
import getopt
import os
import socket
import string
import sys
import time
import random

from program import Program
from game import Game

#----------------------------------------------------------------------------

# Plays a game.
class GamePlayer:
    class Error(Exception):
        pass

    # Constructor.
    #  black and white are programs.
    #  size is boardsize.
    def __init__(self, black, white, size, blunder_rate):
        self._black = self._origblack = black
        self._white = self._origwhite = white
        self._size = size
        self._blunder_rate = blunder_rate
        self._game = Game()
        self._verbose = False

        for program in [self._black, self._white]:
            program.sendCommand(f"boardsize {size} {size}")

        self._blackName = self._black.getName()
        self._whiteName = self._white.getName()
        self._errorMessage = None

    def getErrorMessage(self):
        return self._errorMessage

    #--------------------------------------------------------------------------

    # Set who plays next
    def adjustToMove(self, last):
        if last == 'swap-pieces':   # flip who is playing what color!
            self._black, self._white = self._white, self._black
        else:
            self._blackToMove = not self._blackToMove

    # Ask black to dump board if verbose so that anybody watching can see what is going on.
    def showBoard(self):
        if self._verbose:
            self._sendCommand(self._black, "showboard")

    # Sends the opening moves to each program
    def playOpening(self, opening):
        if opening == '':
            return
        moves = opening.split(' ')
        for move in moves:
            if self._blackToMove:
                self._sendCommand(self._black, f"play b {move}")
                self._sendCommand(self._white, f"play b {move}")
            else:
                self._sendCommand(self._black, f"play w {move}")
                self._sendCommand(self._white, f"play w {move}")
            self._game.addMove(move)
            self.adjustToMove(move)
            self.showBoard()

    # Plays the game after the opening until a player resigns
    def continueGame(self):

        # TODO: SOMEWHERE IN HERE WE HAVE AN EXCEPTION THAT IS FALLING BACK INTO TOURNAMENT AND ENDING THE GAME
        #  current fix is just allowing the game to easily be replayed in the tournament managing code

        resigned = False
        elapsedBlack = 0.0
        elapsedWhite = 0.0

        while not resigned:
            start = time.time()

            # Make the move with the active players program

            if self._blackToMove:
                # BLACK MOVE

                # Are we making a blunder this move?
                if random.random() < self._blunder_rate:
                    # Yes? Pick a random legal move
                    legal_moves = self._sendCommand(self._black, "all_legal_moves").strip().split(" ")
                    if len(legal_moves) == 1:
                        # Our only move is to resign
                        move = legal_moves[0]
                    else:
                        # Remove resigning as a possible blunder
                        legal_moves = legal_moves[1:]
                        move = random.choice(legal_moves)
                    # And make black play it
                    self._sendCommand(self._black, f"play b {move}")
                else:
                    # No? Generate and play a move for black
                    move = self._sendCommand(self._black, "genmove b")
                elapsedBlack += (time.time() - start)
            else:
                # WHITE MOVE

                # Are we making a blunder this move?
                if random.random() < self._blunder_rate:
                    # Yes? Pick a random legal move
                    legal_moves = self._sendCommand(self._white, "all_legal_moves").strip().split(" ")
                    if len(legal_moves) == 1:
                        # Our only move is to resign
                        move = legal_moves[0]
                    else:
                        # Remove resigning as a possible blunder
                        legal_moves = legal_moves[1:]
                        move = random.choice(legal_moves)
                    # And make white play it
                    self._sendCommand(self._white, f"play w {move}")
                else:
                    # No? Generate and play a move for white
                    move = self._sendCommand(self._white, "genmove w")
                elapsedWhite += (time.time() - start)

            move = move.strip().lower()

            self._game.addMove(move)

            # Inform the other player's program of the move that has been made

            if self._blackToMove:
                self._sendCommand(self._white, f"play b {move}")
            else:
                self._sendCommand(self._black, f"play w {move}")

            if "resign" in move:
                resigned = True

            self.adjustToMove(move)
            self.showBoard()

        self._game.setElapsed("black", elapsedBlack)
        self._game.setElapsed("white", elapsedWhite)

    # Plays the opening then the remainder of the game
    def play(self, opening, verbose):
        self._verbose = verbose
        self._blackToMove = True
        self.playOpening(opening)
        self.continueGame()
        return self._game

    #--------------------------------------------------------------------------

    def save(self, fileName, name, resultBlack, resultWhite):
        t = time.localtime()
        sgfDate = time.strftime("%Y-%m-%d", t)
        longDate = time.strftime("%Y-%m-%d %X %Z", t)
        hostname = socket.gethostbyaddr(socket.gethostname())[0]
        result = self._mergeResults(resultBlack, resultWhite)

        with open(fileName, "w") as f:
            f.write("(\n;" \
                    f"GM[11]SZ[{self._size}]PB[{self._sgfText(self._blackName)}]PW[{self._sgfText(self._whiteName)}]\n" \
                    f"RE[{result}]DT[{sgfDate}]GN[{name}]US[twogtp.py]\n" \
                    "GC[Generated by twogtp.py.\n"
                    f"Black Cmd: {self._sgfText(self._origblack.getCommand())}\n" \
                    f"White Cmd: {self._sgfText(self._origwhite.getCommand())}\n" \
                    f"Host: {self._sgfText(hostname)}\n" \
                    f"Time: {self._sgfText(longDate)}\n" \
                    f"Result according to B: {resultBlack}\n" \
                    f"Result according to W: {resultWhite}]\n")

            blackToMove = True
            for move in self._game.moveList():
                if blackToMove:
                    f.write(";B[")
                else:
                    f.write(";W[")
                if move == "swap-pieces":
                    f.write("swap-pieces]\n")
                elif move == "resign":
                    f.write("resign]\n")
                    break
                else:
                    f.write(move + "]")
                    blackToMove = not blackToMove
                f.write("\n")

            f.write(")\n")

    def _mergeResults(self, result1, result2):
        if result1 == result2:
            return result1
        if result1.startswith("B+") and result2.startswith("B+"):
            return "B+"
        if result1.startswith("W+") and result2.startswith("W+"):
            return "W+"
        return "?"

    # Sends command down the channel.
    # Raises GamePlayer.Error if program dies, etc.
    def _sendCommand(self, program, command):
        try:
            out = program.sendCommand(command)
            return out
        except Program.CommandDenied:
            reason = program.getDenyReason()
            self._errorMessage = f"{program.getColor()}: {reason}"
            raise GamePlayer.Error
        except Program.Died:
            self._errorMessage = f"{program.getColor()}: program died"
            raise GamePlayer.Error

    def _sgfText(self, s):
        s = s.replace("\\", "\\\\")
        s = s.replace("]", "\\]")
        return s
