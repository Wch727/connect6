"""Build an opening-candidate table and teacher-labelled supervised samples."""
import argparse
import pickle

from opening_data import html_files, make_sample, prepare_outputs, read_opening

HTML_DIRS = ['raw_openings']
OUTPUT_BOOK_FILE = 'connect6_book.pkl'
OUTPUT_TRAIN_FILE = 'data/sl_data_opening.pkl'


def parse_html_file_final(filepath):
    """Compatibility interface: board, alternative coordinates, player, status."""
    position = read_opening(filepath)
    if position is None:
        return None, None, None, 1
    return list(position.board), list(position.candidates), position.player, 0


def build_book(html_dirs=None, output_book=None, output_data=None, *, limit=None,
               min_value=None, overwrite=False):
    roots = HTML_DIRS if html_dirs is None else html_dirs
    output_book = output_book or OUTPUT_BOOK_FILE
    output_data = output_data or OUTPUT_TRAIN_FILE
    prepare_outputs([output_book, output_data], overwrite)
    book, samples, seen = {}, [], set()
    scanned = 0
    for path in html_files(roots):
        if limit is not None and scanned >= limit:
            break
        scanned += 1
        position = read_opening(path, min_value)
        if position is None or position.board in seen:
            continue
        seen.add(position.board)
        book[position.board] = list(position.candidates)
        samples.append(make_sample(position))
    if not samples:
        raise ValueError('No usable opening positions found')
    with open(output_book, 'wb') as file:
        pickle.dump(book, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_data, 'wb') as file:
        pickle.dump(samples, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Scanned {scanned} HTML files; exported {len(samples)} unique positions.')
    return {'scanned': scanned, 'positions': len(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html-dirs', nargs='+', default=HTML_DIRS)
    parser.add_argument('--output-book', default=OUTPUT_BOOK_FILE)
    parser.add_argument('--output-data', default=OUTPUT_TRAIN_FILE)
    parser.add_argument('--limit', type=int, help='Maximum HTML files scanned')
    parser.add_argument('--min-value', type=float, help='Minimum teacher value from player-to-act perspective')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.min_value is not None and not -1 <= args.min_value <= 1:
        parser.error('--min-value must be between -1 and 1')
    try:
        build_book(args.html_dirs, args.output_book, args.output_data, limit=args.limit,
                   min_value=args.min_value, overwrite=args.overwrite)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
