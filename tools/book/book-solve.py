#!/usr/bin/python3
#----------------------------------------------------------------------------
# Solves given set of positions using dfpn.
#
# Such a set of positions can be generated using the HTP command
#   'book-dump-non-terminal [min # of stones] [output file]'
# to dump all leaf states with at least the given number of stones.
# The output file can then be used as the input file for this script.
#
# Results are dumped to [thread-name].solved and [thread-name].unsolved
# files for processing by other scripts.
#----------------------------------------------------------------------------

import getopt
import sys
from threading import Thread, Lock
from program import Program

#----------------------------------------------------------------------------

class Positions:
    def __init__(self, positionsFile):
        with open(positionsFile, "r") as f:
            self._lines = f.readlines()
        self._lock = Lock()

    def getPosition(self):
        with self._lock:
            ret = ""
            if len(self._lines) > 0:
                ret = self._lines.pop(0).strip()
            return ret

#----------------------------------------------------------------------------

class DfpnSolver(Thread):
    class Error(Exception):
        pass

    def __init__(self, name, command, positions, verbose):
        super().__init__()
        self._name = name
        self._positions = positions
        self._verbose = verbose
        command = command + " --logfile-name=" + name + ".log"
        self._program = Program(command, verbose)

    def sendCommand(self, command):
        try:
            return self._program.sendCommand(command)
        except Program.CommandDenied:
            reason = self._program.getDenyReason()
            self._errorMessage = f"{self._name}: {reason}"
            raise DfpnSolver.Error
        except Program.Died:
            self._errorMessage = f"{self._name}: program died"
            raise DfpnSolver.Error

    def playVariation(self, variation):
        self.sendCommand("clear_board")
        moves = variation.split()
        color = "B"
        for move in moves:
            cmd = f"play {color} {move.strip()}"
            self.sendCommand(cmd)
            color = "W" if color == "B" else "B"

    def solvePosition(self, variation):
        if self._verbose:
            print("#####################################")
            print(f"{self._name}: {variation}")
            print("#####################################")
        else:
            print(f"{self._name}: {variation}")
        self.playVariation(variation)
        return self.sendCommand("dfpn-solve-state")

    def addResult(self, variation, winner):
        if winner == "empty":
            with open(f"{self._name}.unsolved", "a") as f:
                print(variation, file=f)
        else:
            with open(f"{self._name}.solved", "a") as f:
                print(f"{variation} {winner}", file=f)

    def run(self):
        while True:
            variation = self._positions.getPosition()
            if variation == "":
                return
            else:
                winner = self.solvePosition(variation).strip()
                self.addResult(variation, winner)
                print(f"{self._name}: {winner}")

#----------------------------------------------------------------------------

def printUsage():
    sys.stderr.write(
        "Usage: book-solve.py [options]\n"
        "Options:\n"
        "  --help      |-h: print help\n"
        "  --positions |-p: openings to use (required)\n"
        "  --program   |-c: program to run (required)\n"
        "  --threads   |-n: number of threads (default is 1)\n"
        "  --quiet     |-q: be quiet\n")

#----------------------------------------------------------------------------

def main():
    verbose = True
    program = ""
    positionFile = ""
    numThreads = 1

    try:
        options = "hp:c:n:q"
        longOptions = ["help", "positions=", "program=", "threads=", "quiet"]
        opts, args = getopt.getopt(sys.argv[1:], options, longOptions)
    except getopt.GetoptError:
        printUsage()
        sys.exit(1)

    for o, v in opts:
        if o in ("-h", "--help"):
            printUsage()
            sys.exit()
        elif o in ("-p", "--positions"):
            positionFile = v
        elif o in ("-c", "--program"):
            program = v
        elif o in ("-n", "--threads"):
            numThreads = int(v)
        elif o in ("-q", "--quiet"):
            verbose = False

    if not positionFile or not program:
        printUsage()
        sys.exit(1)

    positions = Positions(positionFile)

    solverlist = []
    for i in range(numThreads):
        solver = DfpnSolver(f"thread{i}", program, positions, verbose)
        solverlist.append(solver)
        solver.start()
    for solver in solverlist:
        solver.join()
    print("All threads finished.")

if __name__ == "__main__":
    main()
