import threading
import time
import unittest
from katagomo_ui import Session


class FakeEngine:
    def get_action(self,env):return int(env.get_valid_moves().argmax())
    def recommend(self,env):
        import copy
        trial=copy.deepcopy(env);p=trial.current_player;result=[]
        while not trial.game_over and trial.current_player==p:
            a=self.get_action(trial);result.append(a);trial.step(a)
        return result
    def close(self):pass


class KatagomoSessionTests(unittest.TestCase):
    def session(self,factory=None):
        s=Session(engine_factory=factory or (lambda seconds:FakeEngine()))
        self.addCleanup(s.close)
        return s
    def ready(self,s):
        end=time.monotonic()+3
        while s.state()['busy'] and time.monotonic()<end:time.sleep(.005)
        self.assertFalse(s.state()['busy'])
        self.assertFalse(s.state()['error'])

    def test_black_and_white_opening_and_undo(self):
        s=self.session();s.new({'human':1});s.move(180);self.ready(s)
        self.assertEqual(len(s.state()['moves']),3)
        self.assertEqual(s.undo()['cursor'],0)
        s.new({'human':-1});self.ready(s)
        self.assertEqual(s.state()['cursor'],1)
        self.assertEqual(s.state()['player'],-1)

    def test_review_seek_and_branch_discards_future(self):
        s=self.session();s.load({'text':'(;SZ[19];B[jj];W[ij];W[ji];B[hh])'})
        self.assertEqual(s.state()['total'],4)
        self.assertEqual(s.seek(2)['remaining_stones'],1)
        s.move(0)
        self.assertEqual(s.state()['total'],3)
        self.assertEqual(s.state()['moves'][-1]['action'],0)
        s.seek(0);self.assertEqual(s.state()['cursor'],0)

    def test_recommendation_is_preview_until_applied(self):
        s=self.session();s.new({'mode':'review'});s.move(180)
        s.analyze();self.ready(s)
        state=s.state();self.assertEqual(state['cursor'],1);self.assertEqual(len(state['hints']),2)
        s.apply_hint();self.assertEqual(s.state()['cursor'],3)
        self.assertEqual(s.state()['hints'],[])

    def test_stale_ai_does_not_change_imported_record(self):
        started,release=threading.Event(),threading.Event()
        class Slow(FakeEngine):
            def get_action(self,env):started.set();release.wait(3);return super().get_action(env)
        s=self.session(lambda seconds:Slow());s.new({'human':-1});self.assertTrue(started.wait(2))
        s.load({'text':'(;SZ[19];B[jj];W[ij])'});release.set()
        with s.engine_lock:pass
        self.assertEqual(s.state()['cursor'],2)
        self.assertEqual(s.state()['mode'],'review')

    def test_bad_import_preserves_record_and_settings(self):
        s=self.session();s.new({'mode':'review'});s.move(180)
        for data in [{'text':'(;B[aa];W[aa])'},{'text':'not a record'}]:
            with self.assertRaises(ValueError):s.load(data)
        self.assertEqual(s.state()['cursor'],1)
        for bad in [{'human':True},{'seconds':0},{'seconds':float('nan')},{'mode':'unknown'}]:
            with self.assertRaises(ValueError):s.new(bad)

    def test_resume_truncates_future_and_places_ai_stones(self):
        s=self.session();s.load({'text':'(;B[jj];W[ij];W[ji];B[hh])'});s.seek(1)
        s.resume({'human':1});self.ready(s)
        self.assertEqual(s.state()['cursor'],3)
        self.assertEqual(s.state()['total'],3)
        self.assertEqual(s.state()['mode'],'play')



class KatagomoHTTPTests(unittest.TestCase):
    def setUp(self):
        import urllib.request
        from katagomo_ui import make_server
        self.session=Session(engine_factory=lambda seconds:FakeEngine())
        self.server=make_server(self.session,0)
        self.worker=threading.Thread(target=self.server.serve_forever,daemon=True);self.worker.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'
        self.client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.worker.join(2);self.session.close()
    def request(self,path,data=None,headers=None):
        import urllib.request,json
        body=json.dumps(data).encode() if data is not None else None
        hdr={'Content-Type':'application/json'};hdr.update(headers or {})
        with self.client.open(urllib.request.Request(self.base+path,data=body,headers=hdr),timeout=3) as response:
            return response.read()
    def test_assets_and_export_routes(self):
        import json
        for path in ['/','/katagomo.js','/katagomo.css','/style.css','/favicon.svg','/api/state']:
            self.assertTrue(self.request(path))
        self.request('/api/import',{'text':'(;B[jj];W[ij])'})
        self.assertEqual(len(json.loads(self.request('/api/export'))['moves']),2)
        self.assertIn(';B[jj]',json.loads(self.request('/api/export-sgf'))['sgf'])
    def test_cross_origin_and_invalid_requests_rejected(self):
        import urllib.error
        for headers in [{'Origin':'https://other.example'},{'Host':'other.example'}]:
            with self.assertRaises(urllib.error.HTTPError) as context:self.request('/api/new',{},headers)
            self.assertEqual(context.exception.code,403);context.exception.close()
        for path,data in [('/api/import',{'text':'(;B[aa];W[aa])'}),('/api/seek',{'index':True}),('/api/new',[])]:
            with self.assertRaises(urllib.error.HTTPError) as context:self.request(path,data)
            self.assertEqual(context.exception.code,400);context.exception.close()


if __name__=='__main__':unittest.main()
