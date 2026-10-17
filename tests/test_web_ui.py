"""Real game rules and HTTP checks, with controlled lightweight AI workers."""
import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from web_ui import GameSession, InvalidRequest, make_server


class FirstEmpty:
    def get_action(self, env):
        return int(env.get_valid_moves().argmax())


class WebGameTests(unittest.TestCase):
    def session(self, factory=None, clock=time.monotonic):
        return GameSession(opening_book={}, ai_factory=factory or (lambda _: FirstEmpty()), clock=clock)

    def finished_thinking(self, session):
        deadline=time.monotonic()+3
        while session.state()['thinking'] and time.monotonic()<deadline:
            time.sleep(.005)
        self.assertFalse(session.state()['thinking'])

    def test_black_opening_then_ai_places_two_and_human_has_two(self):
        game=self.session();game.new({});game.move(180);self.finished_thinking(game)
        state=game.state()
        self.assertEqual([x['player'] for x in state['moves']],[1,-1,-1])
        self.assertEqual(state['remaining_stones'],2)
        game.move(181)
        self.assertEqual(game.state()['remaining_stones'],1)
        self.assertFalse(game.state()['thinking'])

    def test_human_white_receives_one_ai_opening_stone(self):
        game=self.session();game.new({'human':-1});self.finished_thinking(game)
        self.assertEqual(len(game.state()['moves']),1)
        self.assertEqual(game.state()['player'],-1)
        game.move(180);game.move(181);self.finished_thinking(game)
        self.assertEqual(len(game.state()['moves']),5)

    def test_undo_restores_before_human_action_including_ai_reply(self):
        game=self.session();game.new({});game.move(180);self.finished_thinking(game)
        state=game.undo()
        self.assertEqual(state['moves'],[])
        self.assertEqual(state['player'],1)
        self.assertEqual(state['remaining_stones'],1)

    def test_restart_discards_old_search_result(self):
        started,release,completed=threading.Event(),threading.Event(),threading.Event()
        class SlowAI:
            def get_action(self, env):
                started.set();release.wait(3);completed.set();return 0
        game=self.session(lambda _:SlowAI());game.new({'human':-1})
        self.assertTrue(started.wait(2))
        game.new({'human':1});release.set();self.assertTrue(completed.wait(2))
        # Acquiring the worker lock waits until its stale-commit check is complete.
        with game.inference_lock:
            self.assertEqual(game.state()['moves'],[])
            self.assertFalse(game.state()['thinking'])

    def test_timeout_wins_for_other_side_and_prevents_move(self):
        now=[0.0];game=self.session(clock=lambda:now[0]);game.new({'seconds':60})
        now[0]=61
        self.assertEqual(game.state()['winner'],-1)
        self.assertEqual(game.state()['reason'],'timeout')
        with self.assertRaises(InvalidRequest):game.move(180)

    def test_invalid_parameters_and_occupied_point(self):
        game=self.session()
        for data in [{'human':True},{'model':'../secret.pt'},{'simulations':0},{'seconds':1},{'rules':'false'}]:
            with self.assertRaises(InvalidRequest):game.new(data)
        game.new({});game.move(180);self.finished_thinking(game)
        with self.assertRaises(InvalidRequest):game.move(180)
        with self.assertRaises(InvalidRequest):game.move(-1)

    def test_model_failure_is_visible_and_clock_pauses(self):
        def fail(_):raise RuntimeError('test model loading failure')
        game=self.session(fail);game.new({'human':-1});self.finished_thinking(game)
        self.assertTrue(game.state()['error'])
        with self.assertRaises(InvalidRequest):game.move(180)
        self.assertFalse(game.new({'human':1})['error'])

    def test_export_is_replayable_and_uses_bottom_left_coordinates(self):
        game=self.session();game.new({});game.move(180);self.finished_thinking(game)
        record=game.export()
        self.assertEqual(record['moves'][0]['coordinate'],'J10')
        self.assertEqual(len(record['moves']),3)
        self.assertFalse(record['finished'])
        self.assertNotIn('path',record['settings'])


class WebHTTPTests(unittest.TestCase):
    def setUp(self):
        self.server=make_server(GameSession(opening_book={}, ai_factory=lambda _:FirstEmpty()),port=0)
        self.worker=threading.Thread(target=self.server.serve_forever,daemon=True);self.worker.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'
        self.client=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.worker.join(2)

    def request(self,path,data=None,headers=None):
        payload=json.dumps(data).encode() if data is not None else None
        hdr={'Content-Type':'application/json'} if data is not None else {}
        hdr.update(headers or {})
        return self.client.open(urllib.request.Request(self.base+path,data=payload,headers=hdr),timeout=3)

    def test_page_assets_and_json_routes(self):
        for path in ['/','/app.js','/style.css','/api/state','/api/models']:
            with self.request(path) as response:
                self.assertEqual(response.status,200)
                self.assertTrue(response.read())
        with self.request('/api/new',{}) as response:
            self.assertTrue(json.load(response)['active'])
        with self.request('/api/export') as response:
            self.assertEqual(json.load(response)['board_size'],19)

    def test_reject_cross_origin_host_and_non_object_json(self):
        for headers in [{'Origin':'https://other.example'},{'Host':'other.example'}]:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self.request('/api/new',{},headers)
            self.assertEqual(ctx.exception.code,403)
            ctx.exception.close()
        with self.assertRaises(urllib.error.HTTPError) as ctx:self.request('/api/new',[])
        self.assertEqual(ctx.exception.code,400)
        ctx.exception.close()


if __name__=='__main__':
    unittest.main()
