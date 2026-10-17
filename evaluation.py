"""MCTS evaluation shared by experiments and the command-line evaluator."""
import random
from contextlib import contextmanager

import numpy as np

from game.connect6_env import Connect6Env, Player
from game.heuristic_ai import HeuristicAI
from mcts.mcts import MCTS


@contextmanager
def _evaluation_seed(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class NetworkMCTSPlayer:
    def __init__(self, network, simulations=400, use_rule_based=True):
        if simulations <= 0:
            raise ValueError('simulations must be positive')
        self.mcts = MCTS(network, num_simulations=simulations, temperature=0,
                         use_rule_based=use_rule_based)

    def get_action(self, env):
        # No exploration noise or temperature sampling during evaluation.
        return int(np.argmax(self.mcts.search(env, add_noise=False)))


def play_match_series(player, opponent, num_games=10, seed=42, verbose=False):
    """Alternate colors and report results from the first player's perspective."""
    if num_games <= 0:
        raise ValueError('num_games must be positive')
    wins = losses = draws = 0
    records = []
    with _evaluation_seed(seed):
        for index in range(num_games):
            env = Connect6Env()
            player_color = Player.BLACK if index % 2 == 0 else Player.WHITE
            agents = {player_color: player, Player(-player_color): opponent}
            moves = []
            while not env.game_over:
                action = int(agents[env.current_player].get_action(env))
                valid = env.get_valid_moves()
                if not 0 <= action < len(valid) or valid[action] == 0:
                    raise ValueError(f'Illegal action during evaluation: {action}')
                moves.append(action)
                env.step(action)
            if env.winner is None:
                draws += 1
                result = 'draw'
            elif env.winner == player_color:
                wins += 1
                result = 'win'
            else:
                losses += 1
                result = 'loss'
            color = 'black' if player_color == Player.BLACK else 'white'
            records.append({'game': index + 1, 'player_color': color,
                            'result': result, 'moves': moves})
            if verbose:
                print(f'Game {index + 1}/{num_games}: player={color}, result={result}')
    return {
        'games': num_games, 'wins': wins, 'losses': losses, 'draws': draws,
        'win_rate': wins / num_games,
        'score_rate': (wins + 0.5 * draws) / num_games,
        'games_as_black': (num_games + 1) // 2, 'games_as_white': num_games // 2,
        'game_records': records,
    }


def evaluate_network(network, num_games=10, simulations=400, use_rule_based=True,
                     seed=42, opponent=None):
    """Evaluate the full search player, restoring the network's previous mode."""
    was_training = network.training
    network.eval()
    try:
        player = NetworkMCTSPlayer(network, simulations, use_rule_based)
        if opponent is None:
            opponent = HeuristicAI(noise_prob=0.0)
        return play_match_series(player, opponent, num_games, seed)
    finally:
        network.train(was_training)
