"""Rejected POST bodies must be consumed so Windows can deliver the response."""
import io
from email.message import Message
import unittest
from web_ui import GameSession, make_server as game_server
from katagomo_ui import Session, make_server as katagomo_server


class RejectedPostTests(unittest.TestCase):
    def check_body_consumed(self,server,check_name):
        self.addCleanup(server.server_close)
        handler=object.__new__(server.RequestHandlerClass)
        handler.headers=Message();handler.headers['Content-Length']='2';handler.headers['Content-Type']='application/json'
        handler.rfile=io.BytesIO(b'{}');observed=[]
        def reject():
            observed.append(handler.rfile.tell());return False
        setattr(handler,check_name,reject)
        handler.do_POST()
        self.assertEqual(observed,[2])

    def test_game_ui_consumes_body_before_rejection(self):
        self.check_body_consumed(game_server(GameSession(opening_book={}),0),'local_request')

    def test_katagomo_ui_consumes_body_before_rejection(self):
        self.check_body_consumed(katagomo_server(Session(engine_factory=lambda _:None),0),'local')


if __name__=='__main__':unittest.main()
