import unittest
import numpy as np
from game.connect6_env import Connect6Env, Player
from game.heuristic_ai import HeuristicAI

class HeuristicTests(unittest.TestCase):
    def position(self, own=(), other=(), remaining=2):
        env=Connect6Env();env.turn=4;env.move_count=2-remaining
        for r,c in own: env.board[r,c]=1
        for r,c in other: env.board[r,c]=-1
        return env

    def test_win_in_two_before_blocking_opponent(self):
        env=self.position([(9,c) for c in (3,4,6,7)],[(2,c) for c in range(2,7)])
        ai=HeuristicAI(noise_prob=0)
        for _ in range(2):
            if not env.game_over: env.step(ai.get_action(env))
        self.assertTrue(env.game_over)
        self.assertEqual(env.winner,Player.BLACK)

    def test_blocks_two_gap_threat_before_building_own_four(self):
        env=self.position([(12,c) for c in (5,6,7)],[(4,c) for c in (2,3,6,7)])
        action=HeuristicAI(noise_prob=0).get_action(env)
        self.assertIn(divmod(action,19),[(4,4),(4,5)])

    def test_random_exploration_cannot_skip_immediate_win(self):
        env=self.position([(9,c) for c in range(3,8)])
        env.step(HeuristicAI(noise_prob=1).get_action(env))
        self.assertEqual(env.winner,Player.BLACK)

    def test_last_stone_blocks_instead_of_starting_two_stone_attack(self):
        env=self.position([(9,c) for c in (3,4,6,7)],[(2,c) for c in range(2,7)],remaining=1)
        self.assertIn(divmod(HeuristicAI(noise_prob=0).get_action(env),19),[(2,1),(2,7)])

    def test_open_pattern_scores_above_blocked_pattern(self):
        ai=HeuristicAI(noise_prob=0)
        board=np.zeros((19,19),dtype=np.int8);board[3,3:6]=1;board[3,2]=-1;board[3,7]=-1
        board[12,3:6]=1
        self.assertGreater(ai._evaluate_point(board,12,6,1),ai._evaluate_point(board,3,6,1))

    def test_search_preserves_board_and_handles_full_board(self):
        env=self.position([(9,9)],[(8,8)])
        before=env.board.copy();ai=HeuristicAI(noise_prob=0)
        self.assertEqual(before.flat[ai.get_action(env)],0)
        np.testing.assert_array_equal(env.board,before)
        env.board[:]=1;self.assertEqual(ai.get_action(env),-1)

    def test_two_stones_cover_two_separate_winning_lines(self):
        env=self.position([(3,1),(12,1)],[(r,c) for r in (3,12) for c in range(2,7)])
        ai=HeuristicAI(noise_prob=0)
        chosen=[]
        for _ in range(2):
            action=ai.get_action(env);chosen.append(divmod(action,19));env.step(action)
        self.assertEqual(set(chosen),{(3,7),(12,7)})
        self.assertFalse(ai._threats(env.board.ravel(),-1,2))

    def test_tactics_apply_to_both_colors_and_diagonals(self):
        ai=HeuristicAI(noise_prob=0)
        for color in (1,-1):
            env=self.position();env.current_player=Player(color)
            for offset in (0,1,3,4):env.board[3+offset,6+offset]=color
            for _ in range(2):env.step(ai.get_action(env))
            self.assertEqual(env.winner,Player(color))

    def test_single_remaining_legal_cell(self):
        env=self.position();env.board[:]=1;env.board[0,0]=0
        self.assertEqual(HeuristicAI(noise_prob=0).get_action(env),0)
