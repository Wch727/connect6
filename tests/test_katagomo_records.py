import json
import unittest
from game.records import parse_record, export_sgf, record_env


class RecordTests(unittest.TestCase):
    def test_katagomo_mainline_and_escaped_comment(self):
        sgf = r'(;SZ[19]C[notes \] ( text];B[jj];W[ij];W[ji](;B[hh])(;B[kk]))'
        self.assertEqual(parse_record(sgf), [180,179,161,140])

    def test_export_roundtrip_and_json_row_col(self):
        moves=[180,179,161,140]
        self.assertEqual(parse_record(export_sgf(moves)),moves)
        data={'board_size':19,'moves':[{'row':a//19,'col':a%19} for a in moves]}
        self.assertEqual(parse_record(json.dumps(data)),moves)

    def test_invalid_game_is_not_imported(self):
        for value in ['(;SZ[15];B[aa])','(;SZ[19];W[aa])','(;B[aa];W[aa])',
                      '(;B[] private text)', '(;B[tt])', '(;AB[aa];B[bb])',
                      json.dumps({'moves':[{'row':True,'col':0}]}),
                      '(;B[aa])(;B[bb])']:
            with self.subTest(value=value),self.assertRaises(ValueError):parse_record(value)

    def test_turn_order_and_terminal_validation(self):
        # Black opens with one, White must place two before Black moves again.
        with self.assertRaises(ValueError):parse_record('(;B[jj];W[ij];B[hh])')
        actions=[180,0,1,181,182,2,3,183,184,20,21,185]
        self.assertTrue(record_env(actions).game_over)
        with self.assertRaises(ValueError):record_env(actions+[22])

    def test_only_first_variation_is_replayed(self):
        self.assertEqual(parse_record('(;SZ[19];B[jj](;W[ij];W[ji])(;W[aa];W[ab]))'),[180,179,161])


if __name__=='__main__':unittest.main()
