"""EnsemFormer top-level entrypoint.

Delegates to scripts/main_train.py for full training logic.

Usage:
    python Main.py [--config config/default.yaml] [--learning_rate 5e-4] ...
"""

from scripts.main_train import main

if __name__ == "__main__":
    main()
