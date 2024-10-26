#!/usr/bin/python3
#----------------------------------------------------------------------------
# Splits a file with M lines into N files with M/N lines.
#----------------------------------------------------------------------------

import sys
import getopt

#----------------------------------------------------------------------------

def printUsage():
    sys.stderr.write(
        "Usage: book-solve.py [options]\n"
        "Options:\n"
        "  --help      |-h: print help\n"
        "  --file      |-f: file to split (required)\n"
        "  --parts     |-p: number of parts to create (default is 2)\n"
        "  --name      |-n: base name of output files (required)\n")

def main():
    file = ""
    name = ""
    parts = 2

    try:
        options = "hf:p:n:"
        longOptions = ["help", "file=", "parts=", "name="]
        opts, args = getopt.getopt(sys.argv[1:], options, longOptions)
    except getopt.GetoptError:
        printUsage()
        sys.exit(1)

    for o, v in opts:
        if o in ("-h", "--help"):
            printUsage()
            sys.exit()
        elif o in ("-f", "--file"):
            file = v
        elif o in ("-p", "--parts"):
            parts = int(v)
        elif o in ("-n", "--name"):
            name = v

    if not file or not name:
        printUsage()
        sys.exit(1)

    with open(file, "r") as f:
        lines = f.readlines()

    numlines = len(lines)
    print("Found " + str(numlines) + " lines.")

    counts = []
    total = 0
    for i in range(parts - 1):
        counts.append(numlines // parts)
        total += counts[i]
    counts.append(numlines - total)
    print("Breakdown: " + str(counts))

    for c, count in enumerate(counts):
        with open(f"{name}{c}", "w") as f:
            for _ in range(count):
                f.write(lines.pop(0).strip() + "\n")

if __name__ == "__main__":
    main()
