"""Convert KataGomo HTML analysis to symmetry-augmented training samples."""
import argparse
import pickle

from opening_data import (augment_sample, html_files, make_sample, parse_opening,
                          prepare_outputs, read_opening)

HTML_DIR = 'raw_openings'
OUTPUT_FILE = 'data/html_opening_data.pkl'


def parse_html_content(content):
    """Compatibility interface returning the first alternative in row-major form."""
    position = parse_opening(content)
    if position is None:
        return None
    row, col = position.candidates[0]
    # This interface historically returned an environment-format board.
    import numpy as np
    board = np.array(position.board, dtype=np.int8).reshape(19, 19)
    board[board == 2] = -1
    return board, position.player, row*19+col


def process_html_files(*, limit=None, min_value=None, overwrite=False):
    prepare_outputs([OUTPUT_FILE], overwrite)
    dataset, seen = [], set()
    scanned = 0
    for path in html_files([HTML_DIR]):
        if limit is not None and scanned >= limit:
            break
        scanned += 1
        position = read_opening(path, min_value)
        if position is None or position.board in seen:
            continue
        seen.add(position.board)
        dataset.extend(augment_sample(make_sample(position)))
    if not dataset:
        raise ValueError('No usable opening positions found')
    with open(OUTPUT_FILE, 'wb') as file:
        pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Exported {len(seen)} positions, {len(dataset)} augmented samples.')
    return {'scanned': scanned, 'positions': len(seen), 'samples': len(dataset)}


def main():
    global HTML_DIR, OUTPUT_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html-dir', default=HTML_DIR)
    parser.add_argument('--output', default=OUTPUT_FILE)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--min-value', type=float)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.min_value is not None and not -1 <= args.min_value <= 1:
        parser.error('--min-value must be between -1 and 1')
    HTML_DIR, OUTPUT_FILE = args.html_dir, args.output
    try:
        process_html_files(limit=args.limit, min_value=args.min_value, overwrite=args.overwrite)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
