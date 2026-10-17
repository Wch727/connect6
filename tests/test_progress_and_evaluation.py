"""Targeted regression checks; the training-loop check needs no accelerator."""
import ast
import contextlib
import io
import random
import time
import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def training_loop():
    # Execute the real method in isolation, without importing PyTorch.
    tree = ast.parse((ROOT / 'training/trainer.py').read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AlphaZeroTrainer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {'time': time}
    exec(compile(module, 'training/trainer.py', 'exec'), namespace)
    return namespace['train']


class ResumeTests(unittest.TestCase):
    def run_loop(self, completed, extra, interval):
        visited, saved = [], []
        trainer = SimpleNamespace(
            iteration=completed, device='cpu', num_self_play_games=1,
            num_simulations=1, batch_size=2, num_epochs=1,
            num_self_play_workers=1, replay_buffer=[],
        )
        trainer.self_play = lambda: visited.append(trainer.iteration) or 0
        trainer.train_network = lambda: {}
        trainer.save_checkpoint = lambda final=False: saved.append((trainer.iteration, final))
        with contextlib.redirect_stdout(io.StringIO()):
            training_loop()(trainer, num_iterations=extra, save_interval=interval)
        return trainer, visited, saved

    def test_resume_80_adds_rounds_81_and_82(self):
        trainer, visited, saved = self.run_loop(80, 2, 1)
        self.assertEqual(visited, [81, 82])
        self.assertEqual(saved, [(81, False), (82, False), (82, True)])
        self.assertEqual(trainer.iteration, 82)

    def test_fresh_training_starts_at_one(self):
        _, visited, _ = self.run_loop(0, 2, 10)
        self.assertEqual(visited, [1, 2])

    def test_checkpoint_interval_uses_absolute_iteration(self):
        _, _, saved = self.run_loop(79, 2, 10)
        self.assertEqual(saved, [(80, False), (81, True)])


class EvaluationTests(unittest.TestCase):
    def module(self):
        try:
            import evaluation
        except ModuleNotFoundError:
            self.fail('Shared MCTS evaluation is not implemented')
        return evaluation

    def test_network_player_performs_search_not_just_one_prediction(self):
        module = self.module()
        from game.connect6_env import Connect6Env
        class Network:
            calls = 0
            def predict(self, state):
                self.calls += 1
                return np.full(361, 1 / 361, dtype=np.float32), 0.0
        network = Network()
        env = Connect6Env()
        before = env.board.copy()
        action = module.NetworkMCTSPlayer(network, simulations=4, use_rule_based=False).get_action(env)
        self.assertGreater(network.calls, 1)
        self.assertEqual(env.get_valid_moves()[action], 1)
        np.testing.assert_array_equal(env.board, before)

    def test_match_series_alternates_colors_and_counts_every_game(self):
        module = self.module()
        class Player:
            def get_action(self, env):
                return int(np.flatnonzero(env.get_valid_moves())[0])
        result = module.play_match_series(Player(), Player(), num_games=3)
        self.assertEqual(result['games'], 3)
        self.assertEqual(result['games_as_black'], 2)
        self.assertEqual(result['games_as_white'], 1)
        self.assertEqual(result['wins'] + result['losses'] + result['draws'], 3)

    def test_invalid_move_fails_instead_of_looping(self):
        module = self.module()
        class Player:
            def get_action(self, env):
                return -1
        with self.assertRaises(ValueError):
            module.play_match_series(Player(), Player(), num_games=2)

    def test_evaluation_restores_training_mode_and_random_state(self):
        module = self.module()
        class Network:
            training = True
            def eval(self): self.training = False
            def train(self, mode=True): self.training = mode
            def predict(self, state):
                return np.full(361, 1 / 361, dtype=np.float32), 0.0
        class Player:
            def get_action(self, env):
                return int(np.flatnonzero(env.get_valid_moves())[0])
        network = Network()
        py_state = random.getstate()
        np_state = np.random.get_state()
        result = module.evaluate_network(network, num_games=2, simulations=1,
                                         opponent=Player(), use_rule_based=False)
        self.assertEqual(result['games'], 2)
        self.assertTrue(network.training)
        self.assertEqual(random.getstate(), py_state)
        self.assertEqual(np.random.get_state()[0], np_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        self.assertEqual(np.random.get_state()[2:], np_state[2:])

    def test_two_model_cli_saves_all_games_and_settings(self):
        # Execute the actual CLI with lightweight agents instead of loading weights.
        tree = ast.parse((ROOT / 'evaluate.py').read_text(encoding='utf-8-sig'))
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        class Player:
            def __init__(self, *args): pass
            def get_action(self, env):
                return int(np.flatnonzero(env.get_valid_moves())[0])
        namespace = {'argparse': argparse, 'os': os, 'AIPlayer': Player,
                     'RandomPlayer': Player, 'GreedyPlayer': Player}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'evaluate.py', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'match.json'
            argv = ['evaluate.py', '--model', 'a.pt', '--opponent-model', 'b.pt',
                    '--games', '3', '--simulations', '7', '--no-rule-based',
                    '--output', str(output)]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                namespace['main']()
            report = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(report['model'], 'a.pt')
        self.assertEqual(report['opponent'], 'b.pt')
        self.assertEqual(report['simulations'], 7)
        self.assertFalse(report['rule_based'])
        self.assertEqual(len(report['results']['game_records']), 3)
        self.assertEqual([r['player_color'] for r in report['results']['game_records']],
                         ['black', 'white', 'black'])


if __name__ == '__main__':
    unittest.main()
