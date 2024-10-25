#----------------------------------------------------------------------------
# Plays the tournament.
#----------------------------------------------------------------------------

import os
import time
import random

from program import Program
from game import Game
from gameplayer import GamePlayer

#----------------------------------------------------------------------------

# Stores summary information about each game played.
class ResultsFile:
    def __init__(self, name, header):
        self._name = name
        self._wasExisting = self._checkExisting()
        if self._wasExisting:
            self._lastIndex = self._findLastIndex()
            self._printTimeStamp()
        else:
            self._lastIndex = -1
            self._printHeader(header)

    def addResult(self, currentRound, opening, blackName, whiteName,
                  resultBlack, resultWhite, gameLen,
                  elapsedBlack, elapsedWhite,
                  error, errorMessage):

        self._lastIndex += 1
        with open(self._name, "a") as f:
            f.write(f"{self._lastIndex:04d}\t{int(currentRound)}\t{opening}\t{blackName}\t{whiteName}\t{resultBlack}\t{resultWhite}\t{gameLen}\t{elapsedBlack:.1f}\t{elapsedWhite:.1f}\t{error}\t{errorMessage}\n")

    def wasExisting(self):
        return self._wasExisting

    def clear(self):
        with open(self._name, "w"):
            pass
        self._printHeader()
        self._gameIndex = -1
        self._wasExisting = False

    def getLastIndex(self):
        return self._lastIndex

    def _checkExisting(self):
        try:
            with open(self._name, "r"):
                return True
        except IOError:
            return False

    def _findLastIndex(self):
        last = -1
        with open(self._name, "r") as f:
            for line in f:
                if line[0] != "#":
                    array = line.split("\t")
                    last = int(array[0])
        return last

    def _printHeader(self, infolines):
        with open(self._name, "w") as f:
            f.write("# Game results file generated by twogtp.py.\n#\n")
            for l in infolines:
                f.write(f"# {l}\n")
            f.write("#\n"
                    "# GAME\tROUND\tOPENING\tBLACK\tWHITE\tRES_B\tRES_W\tLENGTH\tTIME_B\tTIME_W\tERR\tERR_MSG\n#\n")
        self._printTimeStamp()

    def _printTimeStamp(self):
        with open(self._name, "a") as f:
            timeStamp = time.strftime("%Y-%m-%d %X %Z", time.localtime())
            f.write("# Date: " + timeStamp + "\n")

#----------------------------------------------------------------------------

# Base tournament class.
# Contains useful functions for all types of tournaments.
class Tournament:
    def __init__(self,
                 p1name, p1cmd, p2name, p2cmd, size, rounds, outdir,
                 openings, verbose):

        self._p1name = p1name
        self._p1cmd = p1cmd
        self._p2name = p2name
        self._p2cmd = p2cmd
        self._size = size
        self._rounds = rounds
        self._outdir = outdir
        self._verbose = verbose

        info = [
            f"p1name: {p1name}",
            f"p1cmd: {p1cmd}",
            f"p2name: {p2name}",
            f"p2cmd: {p2cmd}",
            f"Boardsize: {size}",
            f"Rounds: {rounds}",
            f"Openings: {openings}",
            f"Directory: {outdir}",
            "Start Date: " + time.strftime("%Y-%m-%d %X %Z", time.localtime())
        ]

        if verbose:
            for line in info:
                print(line)

        self._resultsFile = ResultsFile(os.path.join(outdir, "results"), info)
        self.loadOpenings(openings)

    def loadOpenings(self, openings):
        assert(False)

    def playTournament(self):
        assert(False)

    def handleResult(self, swapped, result):
        ret = result
        if swapped:
            if result.startswith('B+'):
                ret = 'W+' + result[2:]
            elif result.startswith('W+'):
                ret = 'B+' + result[2:]
        return ret

    def playGame(self, gameIndex, currentRound,
                 blackName, blackCmd, whiteName, whiteCmd,
                 opening, verbose):
        if verbose:
            print("\n===========================================================")
            print(f"Game {gameIndex}")
            print("===========================================================\n")

        bcmd = f"nice {blackCmd} --seed %SRAND --logfile-name {self._outdir}/{blackName}-{gameIndex}.log"
        wcmd = f"nice {whiteCmd} --seed %SRAND --logfile-name {self._outdir}/{whiteName}-{gameIndex}.log"
        bLogName = os.path.join(self._outdir, f"{blackName}-{gameIndex}-stderr.log")
        wLogName = os.path.join(self._outdir, f"{whiteName}-{gameIndex}-stderr.log")
        black = Program("B", bcmd, bLogName, verbose)
        white = Program("W", wcmd, wLogName, verbose)

        resultBlack = "?"
        resultWhite = "?"
        error = 0
        errorMessage = ""
        game = Game()  # just a temporary
        gamePlayer = GamePlayer(black, white, self._size)
        try:
            # Play an entire game from the opening move given, or fail on the way
            # COME BACK HERE
            game = gamePlayer.play(opening, verbose)
            swapped = game.playedSwap()
            resultBlack = self.handleResult(swapped, black.getResult())
            resultWhite = self.handleResult(swapped, white.getResult())
        except GamePlayer.Error:
            error = 1
            errorMessage = gamePlayer.getErrorMessage()
        except Program.Died:
            error = 1
            errorMessage = "program died"

        name = f"{self._outdir}/{gameIndex:04d}"

        # save the result
        # recall it has been flipped if a swap move was played
        result = "?"
        if resultBlack == resultWhite:
            result = resultBlack
        game.setResult(result)

        # save it to the results file
        self._resultsFile.addResult(currentRound, opening,
                                    blackName, whiteName,
                                    resultBlack, resultWhite,
                                    # -1 so we don't count "resign" as a move
                                    game.getLength()-1,
                                    game.getElapsed("black"),
                                    game.getElapsed("white"),
                                    error, errorMessage)

        # save game
        gamePlayer.save(name + ".sgf", name, resultBlack, resultWhite)
        if error:
            print(f"Error: Game {gameIndex}")
        for program in [black, white]:
            try:
                program.sendCommand("quit")
            except Program.Died:
                pass

        return game

#----------------------------------------------------------------------------

# Plays a standard iterative tournament:
#   For each round, each program takes each opening as black.
class IterativeTournament(Tournament):
    def loadOpenings(self, openings):
        if openings:
            self._openings = []
            with open(openings, 'r') as f:
                lines = f.readlines()
            for line in lines:
                self._openings.append(line.strip())

    def playTournament(self):
        gamesPerRound = 2 * len(self._openings)
        first = self._resultsFile.getLastIndex() + 1
        for i in range(first, self._rounds * gamesPerRound):
            currentRound = i // gamesPerRound
            gameInRound = i % gamesPerRound
            openingIndex = gameInRound // 2

            opening = self._openings[openingIndex]

            if (i % 2) == 0:
                self.playGame(i, currentRound,
                              self._p1name, self._p1cmd,
                              self._p2name, self._p2cmd,
                              opening, self._verbose)
            else:
                self.playGame(i, currentRound,
                              self._p2name, self._p2cmd,
                              self._p1name, self._p1cmd,
                              opening, self._verbose)

#----------------------------------------------------------------------------

# Plays a random tournament:
#   For each round, pick a random opening (weighted), then:
#     if round is even, program one takes black, otherwise
#     program two takes black.
class RandomTournament(Tournament):
    def loadOpenings(self, openings):
        if openings:
            self._openings = []
            with open(openings, 'r') as f:
                lines = f.readlines()
            sum_weights = 0
            for line in lines:
                stripped = line.strip()
                array = stripped.split(' ')
                sum_weights += float(array[0])
                moves = stripped[len(array[0]):].strip()
                self._openings.append([sum_weights, moves])
            self._maxWeight = sum_weights
            print(self._openings)

    def pickOpening(self):
        randomWeight = random.random() * self._maxWeight
        for i in range(len(self._openings)):
            if randomWeight < self._openings[i][0]:
                return self._openings[i][1]
        assert(False)

    def playTournament(self):
        first = self._resultsFile.getLastIndex() + 1
        for currentRound in range(first, self._rounds):
            opening = self.pickOpening()
            if (currentRound % 2) == 0:
                self.playGame(currentRound, currentRound,
                              self._p1name, self._p1cmd,
                              self._p2name, self._p2cmd,
                              opening, self._verbose)
            else:
                self.playGame(currentRound, currentRound,
                              self._p2name, self._p2cmd,
                              self._p1name, self._p1cmd,
                              opening, self._verbose)

#----------------------------------------------------------------------------
