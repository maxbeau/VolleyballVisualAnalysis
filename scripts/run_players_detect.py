import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from players.detect import main

if __name__ == "__main__":
    main()

