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
        self._log = open(logName, "w") if logName else None
        self.writeLog("# " + self._command + "\n")

    def writeLog(self, message):
        if self._log:
            self._log.write(message)

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
            self.writeLog(">" + cmd + "\n")
            if self._verbose:
                print(self._color + "< " + cmd)
            self._stdin.write((cmd + "\n").encode())
            self._stdin.flush()
            answer = self._getAnswer()
            return answer
        except IOError:
            self._programDied()

    def _getAnswer(self):
        self._logStdErr()
        answer = ""
        done = False
        numberLines = 0
        while not done:
            line = self._stdout.readline().decode()
            if line == "":
                self._programDied()
            self.writeLog("<" + line)
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
        if not self._log:
            return

        list = select([self._stderr], [], [], 0)[0]
        for s in list:
            self._log.write(os.read(s.fileno(), 8192).decode())
        self._log.flush()

    def _programDied(self):
        self._isDead = True
        self._logStdErr()
        raise Program.Died
