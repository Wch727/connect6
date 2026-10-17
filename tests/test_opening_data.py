import ast
from contextlib import redirect_stdout
import io
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from game.connect6_env import Connect6Env
import build
import gen_HTML_data

ROOT = Path(__file__).resolve().parents[1]


def html(player=1, wl=.8, trailing=False):
    cells=','.join(['0']*361)+(',' if trailing else '')
    return (f'const board = [{cells}]; const nextPla = {player}; '
            f"const moves = [{{'xy':[[14,3],[3,14],],'wl':{wl},'v':100,}},"
            "{'xy':[[2,1],],'wl':-0.2,'v':5,},];")


class OpeningDataTests(unittest.TestCase):
    def test_xy_is_column_then_row(self):
        _,_,action=gen_HTML_data.parse_html_content(html())
        self.assertEqual(action,3*19+14)

    def test_board_array_accepts_katagomo_trailing_comma(self):
        self.assertIsNotNone(gen_HTML_data.parse_html_content(html(trailing=True)))

    def test_white_losing_position_is_not_discarded_as_bad_black_score(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'position.html';file.write_text(html(player=2,wl=-.8))
            board,candidates,player,code=build.parse_html_file_final(file)
        self.assertEqual(code,0)
        self.assertEqual(candidates,[(3,14),(14,3)])
        self.assertEqual(player,-1)

    def test_exported_value_and_symmetric_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'position.html').write_text(html(player=1,wl=.8))
            output=root/'dataset.pkl'
            with patch.object(gen_HTML_data,'HTML_DIR',str(root)),patch.object(gen_HTML_data,'OUTPUT_FILE',str(output)),redirect_stdout(io.StringIO()):
                gen_HTML_data.process_html_files()
            # Deserialize only the file produced by this test in its own temporary directory.
            samples=pickle.loads(output.read_bytes())
        self.assertEqual(len(samples),8)
        _,policy,value=samples[0]
        self.assertAlmostEqual(value,-.8)
        self.assertAlmostEqual(policy[3*19+14],.5)
        self.assertAlmostEqual(policy[14*19+3],.5)
        self.assertAlmostEqual(policy.sum(),1)


class ParserBoundaryTests(unittest.TestCase):
    def test_white_and_black_values_are_opposite(self):
        from opening_data import parse_opening
        self.assertAlmostEqual(parse_opening(html(player=1)).value,-.8)
        self.assertAlmostEqual(parse_opening(html(player=2)).value,.8)

    def test_min_value_filters_current_player_perspective(self):
        from opening_data import parse_opening
        black=parse_opening(html(player=1),min_value=0)
        self.assertEqual(black.candidates,((1,2),))
        white=parse_opening(html(player=2),min_value=0)
        self.assertEqual(white.candidates,((3,14),(14,3)))

    def test_invalid_and_occupied_candidates_are_removed(self):
        from opening_data import parse_opening
        content=html().replace("[[14,3],[3,14],]", "[[19,3],[-1,2],[3,14],[3,14],]")
        result=parse_opening(content)
        self.assertEqual(result.candidates,((14,3),))
        # Occupy every point: no legal policy target can be produced.
        content=html().replace(','.join(['0']*361),','.join(['1']*361))
        self.assertIsNone(parse_opening(content))

    def test_missing_analysis_and_invalid_evaluation_are_rejected(self):
        from opening_data import parse_opening
        self.assertIsNone(parse_opening('<html>redirect</html>'))
        self.assertIsNone(parse_opening(html().replace("'wl':0.8", "'wl':2").replace("'v':5", "'v':0")))

    def test_build_preserves_existing_outputs_and_deduplicates_boards(self):
        from opening_data import prepare_outputs
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'one.html').write_text(html())
            nested=root/'nested';nested.mkdir();(nested/'two.html').write_text(html())
            book=root/'book.pkl';data=root/'data.pkl'
            with redirect_stdout(io.StringIO()):
                result=build.build_book([root],book,data)
            self.assertEqual(result['positions'],1)
            before=book.read_bytes()
            with self.assertRaises(FileExistsError):build.build_book([root],book,data)
            self.assertEqual(book.read_bytes(),before)
            with self.assertRaises(ValueError):prepare_outputs([root/'same',root/'same'])


def worker(filename):
    tree=ast.parse((ROOT/filename).read_text(encoding='utf-8-sig'))
    methods=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='ai_worker']
    assert len(methods)==1
    namespace={'FIRST_MOVE_TENGEN':False,'FIRST_MOVE_TENGEN_RATE':0,
               'BOOK_ACCEPT_RATE':1,'np':np,'random':SimpleNamespace(random=lambda:0),
               'time':SimpleNamespace(sleep=lambda _:None),
               'traceback':SimpleNamespace(print_exc=lambda:None)}
    exec(compile(ast.Module(body=methods,type_ignores=[]),filename,'exec'),namespace)
    return namespace['ai_worker']


class BookGUIRegressionTests(unittest.TestCase):
    def test_every_stone_looks_up_its_own_position(self):
        for filename in ['gui_competition.py','gui_hybrid.py']:
            with self.subTest(filename=filename):
                env=Connect6Env();env.step(180)
                lookups=[];fallback=[]
                def lookup():
                    lookups.append(len(env.last_moves))
                    return (True,[(3,14),(14,3)]) if len(lookups)==1 else (False,[])
                agent=SimpleNamespace(get_action=lambda e:(fallback.append(True) or 25))
                gui=SimpleNamespace(env=env,pending_book_moves=[],game_active=True,
                                    ai_thinking=True,check_opening_book_moves=lookup,
                                    post_move=lambda _:None)
                action=worker(filename)
                with redirect_stdout(io.StringIO()):
                    action(gui,agent)
                    if filename=='gui_competition.py':env.step(gui.ai_action_queue)
                    action(gui,agent)
                self.assertEqual(lookups,[1,2])
                self.assertEqual(fallback,[True])
                if filename=='gui_competition.py':self.assertEqual(gui.ai_action_queue,25)
                else:self.assertEqual(env.last_moves[-1],(1,6))


class BookSymmetryTests(unittest.TestCase):
    def test_all_gui_lookups_restore_rotated_candidates_and_filter_occupied_points(self):
        env=Connect6Env();env.board[1,2]=1;env.board[6,11]=-1;env.board[12,17]=1
        for filename in ['gui_competition.py','gui_hybrid.py','gui_additional.py']:
            tree=ast.parse((ROOT/filename).read_text(encoding='utf-8-sig'))
            symmetry=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SymmetryHandler')
            method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='check_opening_book_moves')
            namespace={'np':np,'BOARD_SIZE':19}
            # Execute only these trusted local definitions; no Pygame startup.
            exec(compile(ast.Module(body=[symmetry,method],type_ignores=[]),filename,'exec'),namespace)
            for k in range(4):
                for flip in [False,True]:
                    with self.subTest(gui=filename,rotation=k,flip=flip):
                        transformed=np.rot90(env.board,k)
                        marker=np.zeros((19,19),dtype=int);marker[4,15]=1
                        marker=np.rot90(marker,k)
                        if flip:transformed=np.fliplr(transformed);marker=np.fliplr(marker)
                        key=tuple(np.where(transformed==-1,2,transformed).flatten())
                        point=tuple(map(int,np.argwhere(marker==1)[0]))
                        occupied=tuple(map(int,np.argwhere(transformed!=0)[0]))
                        gui=SimpleNamespace(env=env,opening_book={key:[occupied,point]})
                        found,candidates=namespace['check_opening_book_moves'](gui)
                        self.assertTrue(found)
                        self.assertEqual(candidates,[(4,15)])


if __name__=='__main__':unittest.main()
