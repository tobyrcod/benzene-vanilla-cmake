#----------------------------------------------------------------------------
# Opens connection to GTP program. 
#
# FIXME: very similar to tournament/program.py. Make them share code?
#
#----------------------------------------------------------------------------

import os
import sys
import subprocess

#----------------------------------------------------------------------------

class Program:
    class CommandDenied(Exception):
        pass

    class Died(Exception):
        pass

    def __init__(self, command, verbose):
        self._command = command
        self._verbose = verbose
        if self._verbose:
            print("Creating program:", command)

        # Using `subprocess.Popen` instead of `os.popen3`
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True  # Set text mode for automatic encoding/decoding
        )
        self._isDead = False

    def getCommand(self):
        return self._command

    def getDenyReason(self):
        return self._denyReason

    def getName(self):
        name = "?"
        try:
            name = self.sendCommand("name").strip()
            version = self.sendCommand("version").strip()
            name += " " + version
        except Program.CommandDenied:
            pass
        return name

    def isDead(self):
        return self._isDead

    def sendCommand(self, cmd):
        if self._isDead:
            raise Program.Died("Program is dead and cannot process commands.")

        try:
            if self._verbose:
                print("< " + cmd)
            self._process.stdin.write(cmd + "\n")
            self._process.stdin.flush()
            return self._getAnswer()
        except (IOError, BrokenPipeError):
            self._programDied()

    def _getAnswer(self):
        answer = ""
        numberLines = 0
        while True:
            line = self._process.stdout.readline()
            if line == "":
                self._programDied()
            if self._verbose:
                sys.stdout.write("> " + line)
            numberLines += 1
            if line == "\n":
                break
            answer += line

        # Process and validate answer format
        if answer[0] != '=':
            self._denyReason = answer[2:].strip()
            raise Program.CommandDenied
        return answer[1:].strip() if numberLines == 1 else answer[2:].strip()

    def _programDied(self):
        self._isDead = True
        raise Program.Died("The GTP program has terminated unexpectedly.")
