import time
import unittest
import numpy as np
from web_ui import GameSession

class BookTests(unittest.TestCase):
    def wait(self, game):
        deadline=time.monotonic()+3
        while game.state()['thinking'] and time.monotonic()<deadline: time.sleep(.005)
        self.assertFalse(game.state()['thinking'])
        self.assertFalse(game.state()['error'])

    def test_fresh_lookup_each_stone_without_loading_model(self):
        board=np.zeros((19,19),dtype=int);board[9,9]=1
        book={tuple(board.flat):[(4,5),(5,4)]}
        board[4,5]=2;book[tuple(board.flat)]=[(6,7)]
        def fail(_): raise AssertionError('book hit must bypass model loading')
        game=GameSession(ai_factory=fail,opening_book=book)
        game.new({});game.move(180);self.wait(game)
        self.assertEqual([(m['row'],m['col']) for m in game.state()['moves']],[(9,9),(4,5),(6,7)])
        self.assertEqual([m['source'] for m in game.export()['moves']],['human','book','book'])
        self.assertEqual(game.undo()['moves'],[])

    def test_disabled_book_and_miss_use_ai(self):
        class AI:
            def get_action(self,env): return int(env.get_valid_moves().argmax())
        board=np.zeros((19,19),dtype=int);board[9,9]=1
        for enabled in (True,False):
            game=GameSession(ai_factory=lambda _:AI(),opening_book={tuple(board.flat):[(4,5)]})
            game.new({'book':enabled});game.move(180);self.wait(game)
            self.assertEqual(game.state()['moves'][1]['source'],'book' if enabled else 'ai')
            self.assertEqual(game.state()['moves'][2]['source'],'ai')

    def test_symmetry_and_occupied_candidates(self):
        board=np.zeros((19,19),dtype=int);board[2,3]=1;board[7,8]=-1
        for turns in range(4):
            for flip in (False,True):
                raw=np.where(board==-1,2,board)
                transformed=np.rot90(raw,turns)
                marks=np.zeros_like(board);marks[4,6]=1
                marks=np.rot90(marks,turns)
                if flip: transformed=np.fliplr(transformed);marks=np.fliplr(marks)
                occupied=tuple(map(int,np.argwhere(transformed!=0)[0]))
                candidate=tuple(map(int,np.argwhere(marks==1)[0]))
                game=GameSession(opening_book={tuple(transformed.flat):[occupied,candidate]})
                self.assertEqual(game._book_moves(board),[(4,6)])

    def test_checkbox_and_hint_are_present(self):
        from web_ui import ASSETS
        html=(ASSETS/'index.html').read_text(encoding='utf-8')
        self.assertIn('id="book" type="checkbox" checked',html)
        self.assertIn('id="book-hint"',html)
