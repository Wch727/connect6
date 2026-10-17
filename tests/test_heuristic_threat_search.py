import unittest
from unittest.mock import patch
import numpy as np
from game.connect6_env import Connect6Env
from game.heuristic_ai import HeuristicAI

class ThreatSearchTests(unittest.TestCase):
    def test_defences_include_the_free_second_stone(self):
        flat=np.ones(361,dtype=np.int8)
        flat[[0,1,2,3]]=0
        ai=HeuristicAI(noise_prob=0)
        replies=ai._defence_plans(flat,[frozenset({0})],limit=10)
        self.assertEqual(set(replies),{(0,1),(0,2),(0,3)})
        self.assertIsNone(ai._defence_plans(flat,[frozenset({0})],limit=2))

    def test_cannot_claim_a_fork_if_opponent_wins_first(self):
        ai=HeuristicAI(noise_prob=0)
        flat=np.zeros(361,dtype=np.int8)
        flat[9*19+3:9*19+7]=1
        flat[3*19+3:3*19+8]=-1
        self.assertFalse(ai._prove_plan(flat,1,(9*19+7,10*19+7),depth=2))

    def test_exhausted_budget_is_unknown(self):
        ai=HeuristicAI(noise_prob=0,threat_nodes=0)
        flat=np.zeros(361,dtype=np.int8);flat[180]=1
        self.assertIsNone(ai._find_forced_win(flat,1,2,depth=3))

    def test_action_cache_is_legal_and_turn_aware(self):
        ai=HeuristicAI(noise_prob=0)
        env=Connect6Env();env.step(180)
        action=ai.get_action(env)
        with patch.object(ai,'_point_scores',side_effect=AssertionError('cache missed')):
            self.assertEqual(ai.get_action(env),action)
        env.step(action)
        next_action=ai.get_action(env)
        self.assertNotEqual(next_action,action)
        self.assertEqual(env.board.ravel()[next_action],0)

    def test_search_does_not_change_board(self):
        ai=HeuristicAI(noise_prob=0)
        flat=np.zeros(361,dtype=np.int8);flat[180]=1
        original=flat.copy()
        ai._find_forced_win(flat,1,2,depth=2)
        np.testing.assert_array_equal(original,flat)

    def chain_position(self):
        own=[(6,6),(6,11),(7,6),(7,9),(7,10),(8,7),(8,9),(9,10),(9,11),(10,10),(11,7),(12,9),(12,10)]
        other=[(6,7),(6,9),(6,12),(7,7),(7,8),(7,12),(8,10),(8,11),(9,7),(10,6),(10,7),(10,8),(11,11),(12,11)]
        env=Connect6Env();env.turn=14
        for r,c in own:env.board[r,c]=1
        for r,c in other:env.board[r,c]=-1
        return env

    def test_continuation_is_verified_against_every_legal_defence(self):
        from itertools import combinations
        env=self.chain_position();flat=env.board.ravel().copy()
        ai=HeuristicAI(noise_prob=0,threat_nodes=192,threat_seconds=1)
        self.assertFalse(ai._threats(flat,1,2))
        plan=ai._find_forced_win(flat,1,2,depth=2)
        self.assertEqual(set(plan),{107,89})
        self.assertFalse(HeuristicAI(noise_prob=0)._prove_plan(flat,1,plan,depth=1))
        after=flat.copy();after[list(plan)]=1
        threats=ai._threats(after,1,2)
        # Independently enumerate all legal defensive pairs, rather than using _covers.
        replies=[pair for pair in combinations(map(int,np.flatnonzero(after==0)),2)
                 if all(any(a in threat for a in pair) for threat in threats)]
        self.assertEqual(replies,[(71,179)])
        for reply in replies:
            defended=after.copy();defended[list(reply)]=-1
            follow=ai._known_wins[(defended.tobytes(),1,2)]
            self.assertEqual(set(follow),{236,218})
            defended[list(follow)]=1
            self.assertFalse(ai._threats(defended,-1,2))
            follow_threats=ai._threats(defended,1,2)
            self.assertTrue(follow_threats)
            self.assertFalse(any(all(any(a in t for a in pair) for t in follow_threats)
                                 for pair in combinations(map(int,np.flatnonzero(defended==0)),2)))

    def test_proved_pair_is_completed_on_second_stone(self):
        env=self.chain_position();ai=HeuristicAI(noise_prob=0,threat_seconds=1)
        first=ai.get_action(env);self.assertIsNotNone(ai.last_proof)
        env.step(first)
        second=ai.get_action(env)
        self.assertEqual({first,second},{107,89})
        self.assertEqual(env.board.ravel()[second],0)

    def test_quiet_pair_is_completed_without_replanning(self):
        env=Connect6Env();env.step(180)
        ai=HeuristicAI(noise_prob=0,threat_depth=1)
        first=ai.get_action(env);env.step(first)
        with patch.object(ai,'_choose_action',side_effect=AssertionError('lost pair plan')):
            second=ai.get_action(env)
        self.assertEqual(env.board.ravel()[second],0)
        self.assertNotEqual(first,second)

    def test_forced_replies_cover_every_threat(self):
        ai=HeuristicAI(noise_prob=0)
        flat=np.zeros(361,dtype=np.int8);flat[100]=1
        replies=ai._reply_plans(flat,[10,11,12],(frozenset({20}),frozenset({30})))
        self.assertEqual(replies,[(20,30)])
        replies=ai._reply_plans(flat,[10,11,12],(frozenset({20}),))
        self.assertEqual(set(replies),{(10,20),(11,20),(12,20)})

    def test_pair_candidates_include_partner_after_first_stone(self):
        ai=HeuristicAI(noise_prob=0,beam_width=2)
        flat=np.zeros(361,dtype=np.int8)
        flat[[180,181]]=1;flat[[160,161]]=-1
        own,other=ai._counts(flat,1)
        ranked=ai._ranked(flat,ai._point_scores(own,other),2)
        before=flat.copy()
        plans=ai._pair_plans(flat,1,own,other,ranked)
        self.assertIn(tuple(sorted(ranked)),plans)
        extra=set()
        from game.heuristic_ai import _MEMBERSHIP
        for first in ranked:
            trial=flat.copy();trial[first]=1
            partners=ai._ranked(trial,ai._point_scores(own+_MEMBERSHIP[:,first],other),3)
            for second in partners:
                self.assertIn(tuple(sorted((first,second))),plans)
                if second not in ranked:extra.add(second)
        self.assertTrue(extra)
        self.assertLessEqual(len(plans),7)
        np.testing.assert_array_equal(before,flat)

    def test_avoids_allowing_a_proved_opponent_attack(self):
        flat=self.chain_position().board.ravel().copy()
        flat[[89,107]]=1;flat[[71,179]]=-1
        before=flat.copy()
        ai=HeuristicAI(noise_prob=0,threat_nodes=192,threat_seconds=float('inf'))
        self.assertFalse(ai._threats(flat,1,2))
        chosen=ai._avoid_forced_loss(flat,-1,[(0,1),(218,236)])
        self.assertEqual(chosen,(218,236))
        np.testing.assert_array_equal(before,flat)
        after=flat.copy();after[[0,1]]=-1
        self.assertIsInstance(ai._find_forced_win(after,1,2),tuple)

    def test_unknown_opponent_search_does_not_claim_forced_loss(self):
        flat=np.zeros(361,dtype=np.int8);flat[180]=1
        ai=HeuristicAI(noise_prob=0,threat_nodes=0)
        self.assertEqual(ai._avoid_forced_loss(flat,-1,[(179,199),(160,161)]),(179,199))

    def test_opponent_search_cannot_reset_the_decision_deadline(self):
        ai=HeuristicAI(noise_prob=0,threat_seconds=1)
        flat=np.zeros(361,dtype=np.int8);flat[180:185]=1
        ai._decision_deadline=0
        self.assertIsNone(ai._find_forced_win(flat,1,2))
