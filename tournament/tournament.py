#----------------------------------------------------------------------------
# Plays the tournament.
#----------------------------------------------------------------------------

import os
import time
import random

from program import Program
from game import Game
from gameplayer import GamePlayer
from resultsfile import ResultsFile

#----------------------------------------------------------------------------

# Base tournament class.
# Contains useful functions for all types of tournaments.
class Tournament:
    def __init__(self,
                 p1name, p1cmd, p2name, p2cmd, size, rounds, outdir,
                 openings, verbose, log):

        self._p1name = p1name
        self._p1cmd = p1cmd
        self._p2name = p2name
        self._p2cmd = p2cmd
        self._size = size
        self._rounds = rounds
        self._outdir = outdir
        self._verbose = verbose
        self._log = log

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

        self._resultsFile: ResultsFile = ResultsFile(os.path.join(outdir, "results"), info)
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
                 opening, verbose, log, replace):
        if verbose:
            print("\n===========================================================")
            print(f"Game {gameIndex}")
            print("===========================================================\n")


        bcmd = f"nice {blackCmd} --seed %SRAND"
        wcmd = f"nice {whiteCmd} --seed %SRAND"
        bLogName = ""
        wLogName = ""
        if log:
            bcmd += f" --logfile-name {self._outdir}/{blackName}-{gameIndex}.log"
            wcmd += f" --logfile-name {self._outdir}/{whiteName}-{gameIndex}.log"
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
                                    # -1 so we don't count "resign" as a move,
                                    # As every game ends with one player 'resigning' when they can see the opponent has won
                                    game.getLength()-1,
                                    game.getElapsed("black"),
                                    game.getElapsed("white"),
                                    error, errorMessage, gameIndex if replace else -1)

        # save game
        gamePlayer.save(name + ".sgf", name, resultBlack, resultWhite)
        if error:
            print(f"Error: Game {gameIndex}, Message {errorMessage}")
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
            self._gamesPerRound = len(self._openings)

    def playTournament(self):
        first = self._resultsFile.getLastIndex() + 1
        maxGames = self._rounds * self._gamesPerRound
        if first < maxGames:
            # This tournament hasn't finished
            if not self._resultsFile.wasExisting():
                # This tournament is new
                # Run the new tournament
                print('Starting New Tournament')
                for i in range(first, maxGames):
                    self.playGameFromIndex(i, False)
                return

            # This tournament already exists
            failedGames = self._resultsFile.getFailedResults()
            if not failedGames:
                # The tournament is valid so far, so just continue it
                print('Continuing Tournament')
                for i in range(first, maxGames):
                    self.playGameFromIndex(i, False)
            else:
                # The tournament has some failed games
                # So before continuing the tournament, replay the broken games
                failedIndices = [int(game['GAME']) for game in failedGames]
                print(f'Tournament has some failed games so far: {failedIndices}')
                print("Fixing before continuing tournament")
                self._resultsFile.printMessage("Found Failed Games",
                                               f"attempting to replay failed games {failedIndices}")
                for i in failedIndices:
                    self.playGameFromIndex(i, True)

        else:
            print('Tournament is finished')
            # We have finished this tournament
            # So we want to check it for failed games, and replay them
            if not self._resultsFile.wasExisting():
                raise Exception("We think we have finished a tournament that we don't have any results for")

            failedGames = self._resultsFile.getFailedResults()
            if not failedGames:
                print('Tournament is complete with no errors')
                self._resultsFile.printMessage("Message", "tournament is complete")
                return

            failedIndices = [int(game['GAME']) for game in failedGames]
            print(f'Tournament has some failed games: {failedIndices}')
            self._resultsFile.printMessage("Found Failed Games", f"attempting to replay failed games {failedIndices}")
            for i in failedIndices:
                self.playGameFromIndex(i, True)

    def playGameFromIndex(self, i, replace):
        currentRound = i // self._gamesPerRound
        gameInRound = i % self._gamesPerRound
        opening = self._openings[gameInRound]

        pretty_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
        print(f"round: {currentRound}, game: {gameInRound}, time, {pretty_time}, opening: {opening}")

        if (i % 2) == 0:
            self.playGame(i, currentRound,
                      self._p1name, self._p1cmd,
                      self._p2name, self._p2cmd,
                      opening, self._verbose,
                      self._log, replace)
        else:
            self.playGame(i, currentRound,
                      self._p2name, self._p2cmd,
                      self._p1name, self._p1cmd,
                      opening, self._verbose,
                      self._log, replace)

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
                              opening, self._verbose, self._log, False)
            else:
                self.playGame(currentRound, currentRound,
                              self._p2name, self._p2cmd,
                              self._p1name, self._p1cmd,
                              opening, self._verbose, self._log, False)

#----------------------------------------------------------------------------
