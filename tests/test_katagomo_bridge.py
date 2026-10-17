import unittest
from game.katagomo import Katagomo, to_gtp, from_gtp
from game.connect6_env import Connect6Env


class FakeGTP:
    def __init__(self):self.moves=[];self.commands=[];self.closed=False
    def command(self,cmd,timeout=0):
        self.commands.append(cmd)
        if cmd=='clear_board':self.moves=[]
        elif cmd.startswith('play '):self.moves.append(from_gtp(cmd.split()[2]))
        elif cmd.startswith('genmove '):
            a=next(a for a in range(361) if a not in self.moves);self.moves.append(a);return to_gtp(a)
        elif cmd=='undo':self.moves.pop()
        return ''
    def close(self):self.closed=True


class KatagomoBridgeTests(unittest.TestCase):
    def agent(self):
        fake=FakeGTP()
        return Katagomo(transport_factory=lambda:fake),fake

    def test_all_gtp_coordinates_and_bad_values(self):
        for a in range(361):self.assertEqual(from_gtp(to_gtp(a)),a)
        self.assertEqual(to_gtp(180),'K10')
        for value in ('I10','A0','T20','pass','resign','../x'):
            with self.assertRaises(ValueError):from_gtp(value)

    def test_incremental_play_and_same_color_second_stone(self):
        ai,fake=self.agent();env=Connect6Env();env.step(180)
        a=ai.get_action(env);env.step(a);b=ai.get_action(env)
        self.assertEqual([cmd for cmd in fake.commands if cmd.startswith('genmove')],['genmove w','genmove w'])
        self.assertEqual(fake.commands.count('play b K10'),1)
        self.assertNotEqual(a,b)
        ai.close();self.assertTrue(fake.closed)

    def test_recommendations_do_not_change_real_board_or_engine_history(self):
        ai,fake=self.agent();env=Connect6Env();env.step(180)
        before=env.board.copy();hints=ai.recommend(env)
        self.assertEqual(len(hints),2);self.assertEqual(fake.moves,[180])
        self.assertTrue((before==env.board).all());self.assertEqual(len(env.last_moves),1)
        self.assertEqual(ai.get_action(env),hints[0])
        ai.close()

    def test_undo_and_branch_resynchronize_engine(self):
        ai,fake=self.agent();env=Connect6Env();env.step(180);a=ai.get_action(env);env.step(a)
        branch=Connect6Env();branch.step(181)
        ai.get_action(branch)
        self.assertEqual(fake.moves[0],181)
        self.assertGreaterEqual(fake.commands.count('clear_board'),2)
        ai.close()


class PackageDiscoveryTests(unittest.TestCase):
    def test_package_next_to_a_normally_cloned_repository(self):
        import os,tempfile
        from unittest.mock import patch
        from pathlib import Path
        import game.katagomo as module
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);repo=base/'workspace'/'repo';package=repo.parent/'Katagomo-test'
            for name in ('engine/connectsix19x_opencl.exe','weights/connectsix19x_b18trans.bin.gz','engine.cfg'):
                file=package/name;file.parent.mkdir(parents=True,exist_ok=True);file.touch()
            with patch.object(module,'ROOT',repo),patch.dict(os.environ,{'KATAGOMO_DIR':''}):
                self.assertEqual(module.find_package(),package.resolve())


if __name__=='__main__':unittest.main()
