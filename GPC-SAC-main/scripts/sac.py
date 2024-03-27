from experiment_utils.launch_experiment import launch_experiment

from experiment_configs.configs.q_learning.sac_config import get_config
from experiment_configs.algorithms.offline import get_offline_algorithm
import argparse
def main(args):
    experiment_kwargs = dict(
    exp_postfix='',
    use_gpu=True,
    log_to_tensorboard=False,
)
    variant = dict(
        algorithm='SAC',
        collector_type='step',
        env_name='hopper-random-v2',
        env_kwargs=dict(),
        replay_buffer_size=int(1e6),
        policy_kwargs=dict(
            layer_size=256,
            num_q_layers=3,
            num_p_layers=3,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
        ),
        offline_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            #num_expl_steps_per_train_loop=1000,
            #min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
            save_snapshot_freq=1000,
        ),
    )
    variant['algorithm'] = args.algorithm
    variant['env_name'] = args.env_name
    variant['seed'] = args.seed
    variant['offline_kwargs']['num_epochs'] = args.epoch
    # SAC-N
    exp_postfix = '_{}'
    variant['trainer_kwargs']['policy_lr'] = args.plr
    variant['trainer_kwargs']['qf_lr'] = args.qlr
    experiment_kwargs['exp_postfix'] = ''

    experiment_kwargs['exp_postfix'] = exp_postfix
    experiment_kwargs['data_args'] = {
        'reward_mean': args.reward_mean,
        'reward_std': args.reward_std,
    }
    launch_experiment(
        get_config=get_config,
        get_offline_algorithm=get_offline_algorithm,
        variant=variant,
        **experiment_kwargs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='sac', type=str)  # 修改处
    parser.add_argument('-e',
                        '--env_name',
                        default='halfcheetah-random-v2',
                        type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--log_to_tensorboard', action='store_true')
    parser.add_argument("--epoch", default=3000, type=int)
    parser.add_argument("--plr",
                        default=3e-4,
                        type=float,
                        help='policy learning rate')
    parser.add_argument("--qlr",
                        default=3e-4,
                        type=float,
                        help='Q learning rate')
    parser.add_argument("--reward_mean",
                        action='store_true',
                        help='normalize rewards to 0 mean')
    parser.add_argument("--reward_std",
                        action='store_true',
                        help='normalize rewards to 1 std')
    args = parser.parse_args()
    main(args)
