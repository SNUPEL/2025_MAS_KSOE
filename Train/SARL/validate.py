import os
import torch
import numpy as np

from Environment.environment import Factory
from Environment.utils import calculate_total_weighted_tardiness, calculate_average_workload_deviation


def evaluate(agent1, agent2, agent3, val_dir, bay_data_path):
    if agent1.name == "RL":
        agent1.network.eval()
        use_communication = True
        device = agent1.device
    if agent2.name == "RL":
        agent2.network.eval()
        use_communication = agent2.use_communication
        device = agent2.device

    if agent3 is not None:
        use_spatial_arrangement = True
    else:
        use_spatial_arrangement = False

    val_paths = os.listdir(val_dir)
    tardiness_lst = []
    load_deviation_lst = []

    with torch.no_grad():
        for path in val_paths:
            env = Factory(val_dir + path,
                          bay_data_path,
                          device=device,
                          agent1=agent1.name,
                          agent2=agent2.name,
                          agent3=agent3.name if agent3 is not None else None,
                          use_recording=False,
                          use_communication=use_communication,
                          use_spatial_arrangement=use_spatial_arrangement)

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
                    if use_spatial_arrangement:
                        action_agent3 = agent3.act(state_agent3)
                    else:
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

            delay_log = env.monitor.delay_log
            total_weighted_tardiness = calculate_total_weighted_tardiness(delay_log)

            working_log = env.monitor.working_log
            bay_capacity = np.zeros(env.num_bays)
            for i, row in env.df_bays.iterrows():
                bay_capacity[int(row["bay_id"])] = row["capacity_h1"] + row["capacity_h2"]
            average_load_deviation = calculate_average_workload_deviation(working_log, bay_capacity)

            tardiness_lst.append(total_weighted_tardiness)
            load_deviation_lst.append(average_load_deviation)

        tardiness_avg = sum(tardiness_lst) / len(tardiness_lst) if len(tardiness_lst) > 0 else 0
        load_deviation_avg = sum(load_deviation_lst) / len(load_deviation_lst) if len(load_deviation_lst) > 0 else 0

        return tardiness_avg, load_deviation_avg