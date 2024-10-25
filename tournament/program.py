#----------------------------------------------------------------------------
# Connects to a Hex program.
#----------------------------------------------------------------------------

import os
import string
import sys
import subprocess
from random import randrange
from select import select

#----------------------------------------------------------------------------

class Program:
    class CommandDenied(Exception):
        pass

    class Died(Exception):
        pass

    def __init__(self, color, command, logName, verbose):
        command = command.replace("%SRAND", str(randrange(0, 1000000)))
        self._command = command
        self._color = color
        self._verbose = verbose

        if self._verbose:
            print("Creating program:", command)

        p = subprocess.Popen(command, shell=True,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        (self._stdin,
         self._stdout,
         self._stderr) = (p.stdin, p.stdout, p.stderr)

        self._isDead = False
        self._log = open(logName, "w")
        self._log.write("# " + self._command + "\n")

    def getColor(self):
        return self._color

    def getCommand(self):
        return self._command

    def getDenyReason(self):
        return getattr(self, '_denyReason', None)

    def getName(self):
        name = "?"
        try:
            name = self.sendCommand("name").strip()
            version = self.sendCommand("version").strip()
            name += " " + version
        except Program.CommandDenied:
            pass
        return name

    def getResult(self):
        try:
            l = self.sendCommand("final_score")
            return l.strip()
        except Program.CommandDenied:
            return "?"

    def getTimeRemaining(self):
        try:
            l = self.sendCommand("time_left")
            return l.strip()
        except Program.CommandDenied:
            return "?"

    def isDead(self):
        return self._isDead

    def sendCommand(self, cmd):
        try:
            print(f"sending command {cmd}")
            self._log.write(">" + cmd + "\n")
            if self._verbose:
                print(self._color + "< " + cmd)
            self._stdin.write((cmd + "\n").encode())
            print(f"std written command {cmd}")
            self._stdin.flush()
            print(f"std flushed command {cmd}")
            answer = self._getAnswer()
            print(f"command answer: {answer}")
            return answer
        except IOError:
            print("command answer: died")
            self._programDied()

    def _getAnswer(self):
        print("getting answer")
        self._logStdErr()
        answer = ""
        done = False
        numberLines = 0
        while not done:

            # AFTER A GENMOVE W COMMAND,
            # FOR SOME REASON THE LINE IS ""
            # WHEN IT SHOULD BE E.G. '= B3'

            line = self._stdout.readline().decode()
            print(f"line empty {line == ""}, line new {line == "\n"}")
            if line == "":
                self._programDied()
            self._log.write("<" + line)
            if self._verbose:
                sys.stdout.write(self._color + "> " + line)
            numberLines += 1
            done = (line == "\n")
            if not done:
                answer += line
        if answer[0] != '=':
            self._denyReason = answer[2:].strip()
            raise Program.CommandDenied
        if numberLines == 1:
            return answer[1:].strip()
        return answer[2:]

    def _logStdErr(self):
        list = select([self._stderr], [], [], 0)[0]
        for s in list:
            self._log.write(os.read(s.fileno(), 8192).decode())
        self._log.flush()

    def _programDied(self):
        self._isDead = True
        self._logStdErr()
        raise Program.Died
