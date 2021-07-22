import sys

from . import run
from . import analyze

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please enter a command: run, eval, analyze.")
        sys.exit(1)
    command = sys.argv[1]
    if command in ["run", "eval"]:
        run.main()
    elif command == "analyze":
        analyze.main()
    else:
        print("Please enter a command: run, eval, analyze.")
        sys.exit(1)
