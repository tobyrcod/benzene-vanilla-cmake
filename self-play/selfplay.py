# Script to construct a selfplay dataset with MoHex2.0
import select
import sys
import subprocess
import random
import time

BOARDSIZE = 6

MOHEX_PATH = "../build/src/mohex/mohex"
MOHEX_CONFIG = "mohex-selfplay.htp"


def main():
    mohex = open_mohex()

    # Interactive loop
    while True:
        # Get input from user
        user_input = input("Enter input for C++ program: ")

        # Check for exit condition
        if user_input.lower() == 'exit':
            break

        # Send input to C++ program
        answer = send_mohex_command(mohex, user_input)
        print("C++ program answer:", answer)

    close_mohex(mohex)

def open_mohex():
    mohex = None
    try:
        # Construct Command to Start MoHex
        command = (
            f"{MOHEX_PATH} "
            f"--config {MOHEX_CONFIG} "
            f"--seed {str(random.randrange(0, 1000000))} "
        )

        # Start MoHex C++ Program
        mohex = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, # Sends back and forth text instead of bytes
            shell=True # Used in original, try without as it could be unnecessary
        )

        return mohex

    except Exception as e:
        print(f"An error occurred: {e}")
        if mohex:
            mohex.terminate()


def send_mohex_command(mohex, command):
    mohex.stdin.write(command + "\n")
    mohex.stdin.flush()

    return await_mohex_answer(mohex)


def await_mohex_answer(mohex):
    mohex.stdout.flush()
    answer = []
    while True:
        line = mohex.stdout.readline().strip()
        answer.append(line)
        if line and line[0] in ['=', '?']:
            break
    return answer


def close_mohex(mohex):
    # Clean up
    mohex.stdin.close()
    mohex.terminate()
    mohex.wait(timeout=1)


if __name__ == "__main__":
    main()

