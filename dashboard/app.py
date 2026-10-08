import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from app import main

if __name__ == "__main__":
    main()
