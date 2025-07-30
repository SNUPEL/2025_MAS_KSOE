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


class Config():
    # parser = argparse.ArgumentParser(description="2025_MAS_KSOE")
    def __init__(self):
        self.params = dict()
        self.params["no_vessl"] = True
        self.params['no_cuda'] = False
        self.params['no_record'] = False
        self.params['no_communication'] = True
        self.params['no_spatial_arrangement'] = False
    
        self.params["seed"] = 42
        self.params["use_pretraining"] = False
        self.params["pretrained_model_path_agent1"] = None
        self.params["pretrained_model_path_agent2"] = None
    
        self.params["num_blocks"]=200
        self.params["time_horizon"]=30
        self.params["use_fixed_time_horizon"] = True
        self.params["iat_avg"] = 0.1
        self.params["buffer_avg"] = 1.5
        self.params["weight_factor"] = 0.7
        self.params["bay_data_path"] = "./input/configurations/bay_data.xlsx"
        self.params["block_data_path"] = "./input/configurations/block_data.xlsx"  # block data
        self.params["val_dir"]="./input/validation/"  # directory where the validation data are stored
    
        self.params["algorithm_agent1"]=None  # agent1
        self.params["algorithm_agent2"]="RL"  # agent2
        self.params["algorithm_agent3"]=None  # agent2
    
        self.params["embed_dim"]=64  # node embedding dimension
        self.params["num_heads"]=4  #multi-head attention in HGT layers
        self.params["num_HGT_layers"]=2
        self.params["num_actor_layers"]=2
        self.params["num_critic_layers"]=2
    
        self.params["reward_weight"] = (0.5, 0.5)
        self.params["num_episodes"]=2000
        self.params["lr"]=0.001
        self.params["lr_decay"]=1.0
        self.params["lr_step"]=100
        self.params["gamma"]=0.98
        self.params["lmbda"]=0.95
        self.params["eps_clip"]=0.1
        self.params["K_epoch"]=5
        self.params["T_horizon"]=40
        self.params["P_coeff"]=1
        self.params["V_coeff"]=0.5
        self.params["E_coeff"]=0.01
        self.params['use_value_clipping'] = True
    
        self.params["eval_every"] = 10
        self.params["save_every"] = 100
        self.params["reset_every"] = 1

    
def train(config):
    random.seed(config.params['seed'])
    np.random.seed(config.params['seed'])
    torch.manual_seed(config.params['seed'])
    torch.cuda.manual_seed_all(config.params['seed'])

    use_cuda = torch.cuda.is_available() and not config.params['no_cuda']
    use_vessl = False if config.params['no_vessl'] else True
    use_saved_model = True if config.params['use_pretraining'] else False
    use_recording = False if config.params['no_record'] else True
    use_communication = False if config.params['no_communication'] else True
    use_spatial_arrangement = False if config.params['no_spatial_arrangement'] else True

    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if use_vessl:
        import vessl
        vessl.init(organization="snu-eng-dgx", project="2025_MAS_KSOE", hp=config)

    pretrained_model_path_agent1 = config.params['pretrained_model_path_agent1']
    pretrained_model_path_agent2 = config.params['pretrained_model_path_agent2']

    use_fixed_time_horizon = True if config.params['use_fixed_time_horizon'] else False
    num_blocks = config.params['num_blocks']
    time_horizon = config.params['time_horizon']
    iat_avg = config.params['iat_avg']
    buffer_avg = config.params['buffer_avg']
    weight_factor = config.params['weight_factor']
    bay_data_path = config.params['bay_data_path']
    block_data_path = config.params['block_data_path']

    algorithm_agent1 = config.params['algorithm_agent1']
    algorithm_agent2 = config.params['algorithm_agent2']
    algorithm_agent3 = config.params['algorithm_agent3']

    # 인공신경망 관련 파라미터
    embed_dim = config.params['embed_dim']
    num_heads = config.params['num_heads']
    num_HGT_layers = config.params['num_HGT_layers']
    num_actor_layers = config.params['num_actor_layers']
    num_critic_layers = config.params['num_critic_layers']

    # 강화학습 알고리즘 관련 파라미터
    reward_weight = config.params['reward_weight']
    num_episodes = config.params['num_episodes']
    lr = config.params['lr']
    lr_decay = config.params['lr_decay']
    lr_step = config.params['lr_step']
    gamma = config.params['gamma']
    lmbda = config.params['lmbda']
    eps_clip = config.params['eps_clip']
    K_epoch = config.params['K_epoch']
    T_horizon = config.params['T_horizon']
    P_coeff = config.params['P_coeff']
    V_coeff = config.params['V_coeff']
    E_coeff = config.params['E_coeff']
    use_value_clipping = True if config.params['use_value_clipping'] else False

    eval_every = config.params['eval_every']
    save_every = config.params['save_every']
    reset_every = config.params['reset_every']

    val_dir = config.params['val_dir']

    config.params['ymd'] = time.strftime('%Y%m%d')
    config.params['hour'] = str(time.localtime().tm_hour)
    config.params['minute'] = str(time.localtime().tm_min)
    config.params['second'] = str(time.localtime().tm_sec)

    file_dir = './output/train/SARL/%s-%s-%s/' % (algorithm_agent1, algorithm_agent2, algorithm_agent3)
    model_dir = file_dir + '/%s_%sh_%sm_%ss/model/' % (config.params['ymd'],
                                                       config.params['hour'],
                                                       config.params['minute'],
                                                       config.params['second'])
    log_dir = file_dir + '/%s_%sh_%sm_%ss/log/' % (config.params['ymd'],
                                                   config.params['hour'],
                                                   config.params['minute'],
                                                   config.params['second'])
    # val_dir = file_dir + '/%s_%sh_%sm_%ss/validation/' % (config.params['ymd'],
    #                                                config.params['hour'],
    #                                                config.params['minute'],
    #                                                config.params['second'])

    print('File Stored in:',log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    with open(log_dir + "parameters.json", 'w') as f:
        json.dump(vars(config), f, indent=4)

    if use_fixed_time_horizon:
        block_data_src = DataGenerator(
            block_data_path,
            bay_data_path,
            time_horizon=time_horizon,
            iat_avg=iat_avg,
            buffer_avg=buffer_avg,
            weight_factor=weight_factor,
            fix_time_horizon=True)
    else:
        block_data_src = DataGenerator(
            block_data_path,
            bay_data_path,
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
                  reward_weight=reward_weight,
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


        if e == 1 or e % 10 == 0:
            # 저장 디렉토리
            save_path = log_dir + f'Episode{e}_MDP.json'

            # 기존 json 로드 or 초기화
            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    json_data = json.load(f)
            else:
                json_data = {}

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
                    action_agent2, log_prob_agent2, value_agent2, probs = agent2.get_action(state_agent2)

                    if e == 1 or e % 10 == 0:
                        node_feature_0 = state_agent2.graph_feature.node_stores[0]['x'].cpu().tolist()  # shape: [1,8]
                        node_feature_1 = state_agent2.graph_feature.node_stores[1]['x'].cpu().tolist()  # shape: [17,3]

                        # JSON에 저장할 state 구성
                        state = {
                            'node_Block': node_feature_0,
                            'node_Bay': node_feature_1
                        }
                        # action_prob는 예시로 빈 값 설정 (알고리즘 실행 결과로 채울 수 있음)
                        action_prob = probs.cpu().tolist()  # 예: [0.1, 0.3, 0.6] 처럼 softmax 결과

                        # JSON 데이터에 step 단위로 추가
                        json_data[str(step_agent2)] = {
                            'state': state,
                            'action': action_prob
                        }


                else:
                    action_agent2 = agent2.act(state_agent2)

                next_state_agent3, reward_agent2, done = env.step(action_agent2)
                # print(f"\tStep{step} action:{action_agent2}")
                # print(f"\tStep{step} log prob:{log_prob_agent2}")
                # print(f"\tStep{step} reward:{reward_agent2}")
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
                                      reward=reward_agent2 +reward_agent3 + reward_agent1,
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
                    _, _, last_value, _ = agent2.get_action(state_agent2)

                if len(agent2.memory.actions) > 0:
                    # print('Train called')
                    episode_average_loss += agent2.train(last_value)

            step += 1

            if done:


                break

        print("episode: %d | reward: %.4f | loss: %.4f" % (e, episode_reward / step, episode_average_loss / step))

        if e == 1 or e % 10 == 0:
            # JSON 파일 저장
            with open(save_path, 'w') as f:
                json.dump(json_data, f, indent=2)


        with open(log_dir + "train_log.csv", 'a') as f:
            if algorithm_agent1 == "RL":
                f.write('%d, %1.4f, %1.4f, %f\n'
                        % (e, episode_reward  / step, episode_average_loss / step, agent1.scheduler.get_last_lr()[0]))
            if algorithm_agent2 == "RL":
                f.write('%d, %1.4f, %1.4f, %f\n'
                        % (e, episode_reward / step, episode_average_loss / step, agent2.scheduler.get_last_lr()[0]))

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

        # if e == 1 or e % eval_every == 0:
        #     average_tardiness, average_load_deviation = evaluate(agent1, agent2, agent3, val_dir, bay_data_path)
        #     print("\tValidation tardiness : %.4f | load deviation : %.4f" % (average_tardiness, average_load_deviation))
        #
        #     with open(log_dir + "validation_log.csv", 'a') as f:
        #         f.write('%d,%1.4f,%1.4f\n' % (e, average_tardiness, average_load_deviation))
        #
        #     if use_vessl:
        #         vessl.log(payload={"Perf/Tardiness": average_tardiness}, step=e)
        #         vessl.log(payload={"Perf/LoadDeviation": average_load_deviation}, step=e)
        #     else:
        #         writer.add_scalar("Validation/Tardiness", average_tardiness, e)
        #         writer.add_scalar("Validation/LoadDeviation", average_load_deviation, e)

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
                          reward_weight=reward_weight,
                          use_recording=use_recording,
                          use_communication=use_communication,
                          use_spatial_arrangement=use_spatial_arrangement)

    if not use_vessl:
        writer.close()


if __name__ == "__main__":
    config = Config()
    train(config)