import os
import torch

from Environment.environment import Factory


def evaluate(agent1, agent2, agent3, val_dir, bay_data_src):
    if agent1.name == "RL":
        agent1.network.eval()
        use_communication = True
        device = agent1.device
    if agent2.name == "RL":
        agent2.network.eval()
        use_communication = agent2.use_communication
        device = agent2.device

    val_paths = os.listdir(val_dir)
    tardiness_lst = []
    load_deviation_lst = []

    with torch.no_grad():
        for path in val_paths:
            env = Factory(val_dir + path,
                          bay_data_src,
                          device=device,
                          agent1=agent1.name,
                          agent2=agent2.name,
                          agent3=agent3.name,
                          use_recording=False,
                          use_communication=use_communication)

            state_agent1, _ = env.reset()

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
                    if agent1.name == "RL":
                        action_agent1, _, _ = agent1.get_action(state_agent1)
                    else:
                        action_agent1 = agent1.act(state_agent1)

                    next_state_agent2, reward, done = env.step(action_agent1)
                elif mode == "agent2":
                    if agent2.name == "RL":
                        action_agent2, _, _ = agent2.get_action(state_agent2)
                    else:
                        action_agent2 = agent2.act(state_agent2)

                    next_state_agent3, reward, done = env.step(action_agent2)
                elif mode == "agent3":
                    action_agent3 = None
                    next_state_agent1, reward, done = env.step(action_agent3)

                if mode == "agent1":
                    state_agent2 = next_state_agent2
                elif mode == "agent2":
                    state_agent3 = next_state_agent3
                elif mode == "agent3":
                    state_agent1 = next_state_agent1

                if done:
                    break

            # 추후 변경 예정
            tardiness_lst.append(env.sink.completion_time)
            load_deviation_lst.append(env.sink.completion_time)

        tardiness_avg = sum(tardiness_lst) / len(tardiness_lst)
        load_deviation_avg = sum(load_deviation_lst) / len(load_deviation_lst)

        return tardiness_avg, load_deviation_avg