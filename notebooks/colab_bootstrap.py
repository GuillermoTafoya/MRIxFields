"""Minimal local/Colab smoke bootstrap.

This file intentionally does not download data or install packages. It assumes
the repository has already been made available in the runtime.
"""

from clbfield.training import run_smoke_train


if __name__ == "__main__":
    result = run_smoke_train()
    print(result.to_dict())

