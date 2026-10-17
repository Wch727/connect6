import argparse
import torch
import csv
import os
from training.trainer import AlphaZeroTrainer
from model.network import get_network
from evaluation import evaluate_network

def evaluate_model(network, num_games=10, simulations=400, use_rule_based=True, seed=42):
    return evaluate_network(network, num_games, simulations, use_rule_based, seed)['win_rate']

def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-name', type=str, required=True)
    parser.add_argument('--model-size', type=str, default='hybrid')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--no-rule-based', action='store_true')
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--eval-games', type=int, default=10)
    parser.add_argument('--eval-simulations', type=int, default=400)
    parser.add_argument('--eval-seed', type=int, default=42)
    parser.add_argument('--eval-no-rule-based', action='store_true',
                        help='仅在评测时关闭规则辅助；公平对比时两组使用相同评测设置')
    args = parser.parse_args()
    if args.eval_games <= 0 or args.eval_simulations <= 0:
        parser.error('--eval-games 和 --eval-simulations 必须为正数')

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints_exp", exist_ok=True) # 专门存实验模型

    log_path = os.path.join("logs", args.log_name)

    # === 关键：表头增加了 Loss 字段 ===
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['iteration', 'win_rate', 'policy_loss', 'value_loss', 'entropy',
                               'eval_games', 'eval_simulations', 'eval_rule_based', 'eval_seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = get_network(args.model_size).to(device)

    if args.pretrained:
        print(f"Loading: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        network.load_state_dict(ckpt['model_state_dict'])

    trainer = AlphaZeroTrainer(
        network=network,
        device=device,
        num_simulations=400,
        num_self_play_games=20,
        num_epochs=5,
        num_parallel=args.parallel,
        use_rule_based=not args.no_rule_based
    )

    for i in range(args.iterations):
        print(f"\n=== Iteration {i+1} ===")

        # 1. 自我对弈
        trainer.self_play()

        # 2. 训练网络 (获取 Loss 字典)
        train_stats = trainer.train_network()
        p_loss = train_stats.get('policy_loss', 0)
        v_loss = train_stats.get('value_loss', 0)
        entropy = train_stats.get('entropy', 0)

        # 3. 评估胜率
        win_rate = evaluate_model(network, args.eval_games, args.eval_simulations,
                                  not args.eval_no_rule_based, args.eval_seed)
        print(f"Stats -> Win: {win_rate:.2f} | P_Loss: {p_loss:.4f} | V_Loss: {v_loss:.4f}")

        # 4. 记录所有数据到 CSV
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([i+1, win_rate, p_loss, v_loss, entropy,
                                   args.eval_games, args.eval_simulations,
                                   not args.eval_no_rule_based, args.eval_seed])

        # === 关键：每 10 轮保存一个中间模型，防止白跑 ===
        if (i + 1) % 10 == 0:
            ckpt_name = f"{args.log_name.split('.')[0]}_iter_{i+1}.pt"
            save_path = os.path.join("checkpoints_exp", ckpt_name)
            torch.save({
                'model_state_dict': network.state_dict(),
                'iteration': i+1
            }, save_path)
            print(f"Checkpoint saved: {save_path}")

if __name__ == "__main__":
    run_experiment()
