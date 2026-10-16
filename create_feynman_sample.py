import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = Path("Feynman_without_units/I.9.18")
DEFAULT_OUTPUT = Path("Feynman_without_units/I.9.18_sample_10000")
SAMPLE_SIZE = 10_000
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the deterministic 10,000-row Feynman dataset sample."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source dataset (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Sample output file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(f"Source dataset not found: {args.source}")
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Source and output paths must be different.")

    with args.source.open("rb") as source_file:
        row_count = sum(1 for _ in source_file)

    if row_count < SAMPLE_SIZE:
        raise ValueError(
            f"Source contains {row_count:,} rows; at least "
            f"{SAMPLE_SIZE:,} are required."
        )

    sample_indices = pd.DataFrame(index=range(row_count)).sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    ).index.tolist()
    output_positions = {
        source_index: output_index
        for output_index, source_index in enumerate(sample_indices)
    }
    sampled_lines = [None] * SAMPLE_SIZE

    with args.source.open("rb") as source_file:
        for source_index, line in enumerate(source_file):
            output_index = output_positions.get(source_index)
            if output_index is not None:
                sampled_lines[output_index] = line.rstrip() + b"\n"

    if any(line is None for line in sampled_lines):
        raise RuntimeError("Could not locate every sampled row in the source file.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output_file:
        output_file.writelines(sampled_lines)

    print(f"Read {row_count:,} rows from {args.source}")
    print(
        f"Saved {len(sampled_lines):,} rows to {args.output} "
        f"using random_state={RANDOM_STATE}"
    )


if __name__ == "__main__":
    main()
