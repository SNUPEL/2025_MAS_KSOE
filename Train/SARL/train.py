import os
import time
import json
import torch
import argparse
import random
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from Environment.environment import Factory
from Environment.data import DataGenerator
from Agent.Agent1.heuristic import BSHeuristic
from Agent.Agent2.heuristic import BAHeuristic
from Agent.Agent3.heuristic import BLHeuristic
from Agent.Agent1.ppo import Agent1
from Agent.Agent2.ppo import Agent2
from Train.SARL.validate import evaluate


def get_config():
    parser = argparse.ArgumentParser(description="2025_MAS_KSOE")

    parser.add_argument("--no_vessl", action='store_true', help="Disable VESSL")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_record', action='store_true', help="Disable Recording events")
    parser.add_argument('--no_communication', action='store_true', help="Disable communication")
    parser.add_argument('--no_spatial_arrangement', action='store_true', help="Disable spatial arrangement")

    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument("--use_pretraining", action='store_true', help="Load the pre-trained models")
    parser.add_argument("--pretrained_model_path_agent1", type=str, default=None, help="agent1 model file path")
    parser.add_argument("--pretrained_model_path_agent2", type=str, default=None, help="agent2 model file path")

    parser.add_argument("--num_blocks", type=int, default=100, help="number of blocks")
    parser.add_argument("--time_horizon", type=int, default=30, help="time horizon")
    parser.add_argument("--use_fixed_time_horizon", action='store_true', help="Fix the time horizon")
    parser.add_argument("--iat_avg", type=float, default=0.1, help="average inter-arrival time")
    parser.add_argument("--buffer_avg", type=float, default=1.5, help="average buffer")
    parser.add_argument("--weight_factor", type=float, default=0.7, help="weight factor")
    parser.add_argument("--bay_data_path", type=str, default="./input/configurations/bay_data.xlsx", help="bay data")
    parser.add_argument("--block_data_path", type=str, default="./input/configurations/block_data.xlsx", help="block data")
    parser.add_argument("--val_dir", type=str, default=None, help="directory where the validation data are stored")

    parser.add_argument("--algorithm_agent1", type=str, default=None, help="agent1")
    parser.add_argument("--algorithm_agent2", type=str, default=None, help="agent2")
    parser.add_argument("--algorithm_agent3", type=str, default=None, help="agent2")

    parser.add_argument("--embed_dim", type=int, default=128, help="node embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="multi-head attention in HGT layers")
    parser.add_argument("--num_HGT_layers", type=int, default=2, help="number of HGT layers")
    parser.add_argument("--num_actor_layers", type=int, default=2, help="number of actor layers")
    parser.add_argument("--num_critic_layers", type=int, default=2, help="number of critic layers")

    parser.add_argument("--num_episodes", type=int, default=2000, help="number of episodes")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--lr_decay", type=float, default=1.0, help="learning rate decay ratio")
    parser.add_argument("--lr_step", type=int, default=100, help="step size to reduce learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount ratio")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--eps_clip", type=float, default=0.1, help="clipping parameter")
    parser.add_argument("--K_epoch", type=int, default=5, help="optimization epoch")
    parser.add_argument("--T_horizon", type=int, default=40, help="the number of steps to obtain samples")
    parser.add_argument("--P_coeff", type=float, default=1, help="coefficient for policy loss")
    parser.add_argument("--V_coeff", type=float, default=0.5, help="coefficient for value loss")
    parser.add_argument("--E_coeff", type=float, default=0.01, help="coefficient for entropy loss")
    parser.add_argument('--use_value_clipping', action='store_true', help="Use value clipping")

    parser.add_argument("--eval_every", type=int, default=50, help="Evaluate every x episodes")
    parser.add_argument("--save_every", type=int, default=500, help="Save a model every x episodes")
    parser.add_argument("--reset_every", type=int, default=1, help="Generate new instances every x episodes")

    return parser.parse_args()


def train(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    use_cuda = torch.cuda.is_available() and not config.no_cuda
    use_vessl = False if config.no_vessl else True
    use_saved_model = True if config.use_pretraining else False
    use_recording = False if config.no_record else True
    use_communication = False if config.no_communication else True
    use_spatial_arrangement = False if config.no_spatial_arrangement else True

    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if use_vessl:
        import vessl
        vessl.init(organization="snu-eng-dgx", project="2025_MAS_KSOE", hp=config)

    pretrained_model_path_agent1 = config.pretrained_model_path_agent1
    pretrained_model_path_agent2 = config.pretrained_model_path_agent2

    use_fixed_time_horizon = True if config.use_fixed_time_horizon else False
    num_blocks = config.num_blocks
    time_horizon = config.time_horizon
    iat_avg = config.iat_avg
    buffer_avg = config.buffer_avg
    weight_factor = config.weight_factor
    bay_data_path = config.bay_data_path
    block_data_path = config.block_data_path

    algorithm_agent1 = config.algorithm_agent1
    algorithm_agent2 = config.algorithm_agent2
    algorithm_agent3 = config.algorithm_agent3

    # 인공신경망 관련 파라미터
    embed_dim = config.embed_dim
    num_heads = config.num_heads
    num_HGT_layers = config.num_HGT_layers
    num_actor_layers = config.num_actor_layers
    num_critic_layers = config.num_critic_layers

    # 강화학습 알고리즘 관련 파라미터
    num_episodes = config.num_episodes
    lr = config.lr
    lr_decay = config.lr_decay
    lr_step = config.lr_step
    gamma = config.gamma
    lmbda = config.lmbda
    eps_clip = config.eps_clip
    K_epoch = config.K_epoch
    T_horizon = config.T_horizon
    P_coeff = config.P_coeff
    V_coeff = config.V_coeff
    E_coeff = config.E_coeff
    use_value_clipping = True if config.use_value_clipping else False

    eval_every = config.eval_every
    save_every = config.save_every
    reset_every = config.reset_every

    val_dir = config.val_dir

    config.ymd = time.strftime('%Y%m%d')
    config.hour = str(time.localtime().tm_hour)
    config.minute = str(time.localtime().tm_min)
    config.second = str(time.localtime().tm_sec)

    file_dir = './output/train/SARL/%s-%s-%s/' % (algorithm_agent1, algorithm_agent2, algorithm_agent3)
    model_dir = file_dir + '/%s_%sh_%sm_%ss/model/' % (config.ymd, config.hour, config.minute, config.second)
    log_dir = file_dir + '/%s_%sh_%sm_%ss/log/' % (config.ymd, config.hour, config.minute, config.second)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(log_dir + "parameters.json", 'w') as f:
        json.dump(vars(config), f, indent=4)

    if use_fixed_time_horizon:
        block_data_src = DataGenerator(
            block_data_path,
            time_horizon=time_horizon,
            iat_avg=iat_avg,
            buffer_avg=buffer_avg,
            weight_factor=weight_factor,
            fix_time_horizon=True)
    else:
        block_data_src = DataGenerator(
            block_data_path,
            num_blocks=num_blocks,
            iat_avg=iat_avg,
            buffer_avg=buffer_avg,
            weight_factor=weight_factor,
            fix_time_horizon=False)

    env = Factory(block_data_src,
                  bay_data_path,
                  device=device,
                  agent1=algorithm_agent1,
                  agent2=algorithm_agent2,
                  agent3=algorithm_agent3,
                  use_recording=use_recording,
                  use_communication=use_communication,
                  use_spatial_arrangement=use_spatial_arrangement)

    if algorithm_agent1 == "RL":
        agent1 = Agent1(meta_data=env.meta_data_agent1,
                        state_size=env.state_size_agent1,
                        num_nodes=env.num_nodes_agent1,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        num_HGT_layers=num_HGT_layers,
                        num_actor_layers=num_actor_layers,
                        num_critic_layers=num_critic_layers,
                        lr=lr,
                        lr_decay=lr_decay,
                        lr_step=lr_step,
                        gamma=gamma,
                        lmbda=lmbda,
                        eps_clip=eps_clip,
                        K_epoch=K_epoch,
                        P_coeff=P_coeff,
                        V_coeff=V_coeff,
                        E_coeff=E_coeff,
                        use_value_clipping=use_value_clipping,
                        use_parameter_sharing=True,
                        device=device)

    else:
        agent1 = BSHeuristic(algorithm_agent1)

    if algorithm_agent2 == "RL":
        agent2 = Agent2(meta_data=env.meta_data_agent2,
                        state_size=env.state_size_agent2,
                        num_nodes=env.num_nodes_agent2,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        num_HGT_layers=num_HGT_layers,
                        num_actor_layers=num_actor_layers,
                        num_critic_layers=num_critic_layers,
                        lr=lr,
                        lr_decay=lr_decay,
                        lr_step=lr_step,
                        gamma=gamma,
                        lmbda=lmbda,
                        eps_clip=eps_clip,
                        K_epoch=K_epoch,
                        P_coeff=P_coeff,
                        V_coeff=V_coeff,
                        E_coeff=E_coeff,
                        use_value_clipping=use_value_clipping,
                        use_parameter_sharing=True,
                        use_communication=use_communication,
                        device=device)
    else:
        agent2 = BAHeuristic(algorithm_agent2)

    if use_spatial_arrangement:
        agent3 = BLHeuristic(algorithm_agent3)
    else:
        agent3 = None

    if not use_vessl:
        writer = SummaryWriter(log_dir)

    if use_saved_model:
        if algorithm_agent1 == "RL":
            checkpoint = torch.load(pretrained_model_path_agent1)
            agent1.network.load_state_dict(checkpoint['model_state_dict'])
            agent1.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if algorithm_agent2 == "RL":
            checkpoint = torch.load(pretrained_model_path_agent2)
            agent2.network.load_state_dict(checkpoint['model_state_dict'])
            agent2.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    with open(log_dir + "train_log.csv", 'w') as f:
        f.write('episode, reward, loss, lr\n')
    with open(log_dir + "validation_log.csv", 'w') as f:
        f.write('episode, tardiness, load_deviation\n')

    for e in range(1, num_episodes + 1):
        if use_vessl:
            if algorithm_agent1 == "RL":
                vessl.log(payload={"Train/LearnigRate": agent1.scheduler.get_last_lr()[0]}, step=e)
            if algorithm_agent2 == "RL":
                vessl.log(payload={"Train/LearnigRate": agent2.scheduler.get_last_lr()[0]}, step=e)
        else:
            if algorithm_agent1 == "RL":
                writer.add_scalar("Training/LearningRate", agent1.scheduler.get_last_lr()[0], e)
            if algorithm_agent2 == "RL":
                writer.add_scalar("Training/LearningRate", agent2.scheduler.get_last_lr()[0], e)

        step = 0
        step_agent1 = 0
        step_agent2 = 0

        episode_reward = 0.0
        episode_average_loss = 0.0

        state_agent1 = env.reset()

        while True:
            if env.agent_mode == "agent1":
                mode = "agent1"
            elif env.agent_mode == "agent2":
                mode = "agent2"
            elif env.agent_mode == "agent3":
                mode = "agent3"
            else:
                raise Exception("Invalid agent mode")

            if mode == "agent1":
                step_agent1 += 1

                if algorithm_agent1 == "RL":
                    action_agent1, log_prob_agent1, value_agent1 = agent1.get_action(state_agent1)
                else:
                    action_agent1 = agent1.act(state_agent1)

                next_state_agent2, reward_agent1, done = env.step(action_agent1)
                episode_reward += reward_agent1
            elif mode == "agent2":
                step_agent2 += 1

                if algorithm_agent2 == "RL":
                    action_agent2, log_prob_agent2, value_agent2 = agent2.get_action(state_agent2)
                else:
                    action_agent2 = agent2.act(state_agent2)

                next_state_agent3, reward_agent2, done = env.step(action_agent2)
                episode_reward += reward_agent2
            elif mode == "agent3":
                if use_spatial_arrangement:
                    action_agent3 = agent3.act(state_agent3)
                else:
                    action_agent3 = None

                next_state_agent1, reward_agent3, done = env.step(action_agent3)

            if mode == "agent1":
                if algorithm_agent2 == "RL" and step_agent2 >= 1:
                    agent2.put_sample(state=state_agent2,
                                      action=action_agent2,
                                      reward=reward_agent2 + +reward_agent3 + reward_agent1,
                                      done=done,
                                      log_prob=log_prob_agent2,
                                      value=value_agent2)

                state_agent2 = next_state_agent2
            elif mode == "agent2":
                state_agent3 = next_state_agent3
            elif mode == "agent3":
                if algorithm_agent1 == "RL" and step_agent1 >= 1:
                    agent1.put_sample(state=state_agent1,
                                      action=action_agent1,
                                      reward=reward_agent1 + reward_agent2 + reward_agent3,
                                      done=done,
                                      log_prob=log_prob_agent1,
                                      value=value_agent1)

                state_agent1 = next_state_agent1

            if algorithm_agent1 == "RL" and (done or len(agent1.memory.actions) == T_horizon):
                if done:
                    last_value = 0.0
                else:
                    _, _, last_value = agent1.get_action(state_agent1)

                if len(agent1.memory.actions) > 0:
                    episode_average_loss += agent1.train(last_value)

            if algorithm_agent2 == "RL" and (done or len(agent2.memory.actions) == T_horizon):
                if done:
                    last_value = 0.0
                else:
                    _, _, last_value = agent2.get_action(state_agent2)

                if len(agent2.memory.actions) > 0:
                    episode_average_loss += agent2.train(last_value)

            step += 1

            if done:
                break

        print("episode: %d | reward: %.4f | loss: %.4f" % (e, episode_reward, episode_average_loss / step))
        with open(log_dir + "train_log.csv", 'a') as f:
            if algorithm_agent1 == "RL":
                f.write('%d, %1.4f, %1.4f, %f\n'
                        % (e, episode_reward, episode_average_loss / step, agent1.scheduler.get_last_lr()[0]))
            if algorithm_agent2 == "RL":
                f.write('%d, %1.4f, %1.4f, %f\n'
                        % (e, episode_reward, episode_average_loss / step, agent2.scheduler.get_last_lr()[0]))

        if use_vessl:
            vessl.log(payload={"Train/Reward": episode_reward,
                               "Train/Loss": episode_average_loss / step}, step=e)
        else:
            writer.add_scalar("Training/Reward", episode_reward, e)
            writer.add_scalar("Training/Loss", episode_average_loss / step, e)

        if algorithm_agent1 == "RL":
            agent1.scheduler.step()

        if algorithm_agent2 == "RL":
            agent2.scheduler.step()

        if e == 1 or e % eval_every == 0:
            average_tardiness, average_load_deviation = evaluate(agent1, agent2, agent3, val_dir, bay_data_src)

            with open(log_dir + "validation_log.csv", 'a') as f:
                f.write('%d,%1.4f,%1.4f\n' % (e, average_tardiness, average_load_deviation))

            if use_vessl:
                vessl.log(payload={"Perf/Tardiness": average_tardiness}, step=e)
                vessl.log(payload={"Perf/LoadDeviation": average_load_deviation}, step=e)
            else:
                writer.add_scalar("Validation/Tardiness", average_tardiness, e)
                writer.add_scalar("Validation/LoadDeviation", average_load_deviation, e)

        if e % save_every == 0:
            if algorithm_agent1 == "RL":
                agent1.save_network(e, model_dir)
            if algorithm_agent2 == "RL":
                agent2.save_network(e, model_dir)

        if e % reset_every == 0:
            env = Factory(block_data_src,
                          bay_data_path,
                          device=device,
                          agent1=algorithm_agent1,
                          agent2=algorithm_agent2,
                          agent3=algorithm_agent3,
                          use_recording=use_recording,
                          use_communication=use_communication,
                          use_spatial_arrangement=use_spatial_arrangement)

    if not use_vessl:
        writer.close()


if __name__ == "__main__":
    config = get_config()
    train(config)