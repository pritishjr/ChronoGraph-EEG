"""Subject- and seizure-aware dataset splitting."""

from .loso import write_folds

__all__ = ["write_folds"]


def main() -> None:
    from pathlib import Path

    write_folds(Path("data/processed"), Path("data/manifests/folds.json"))


if __name__ == "__main__":
    main()
