"""WebShop training entry point, sharing the ALFWorld trainer implementation."""
from __future__ import annotations

import os

from recipe.graph_guide.main_alfworld_graph_guide import main


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
