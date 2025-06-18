import torch
import simpy
import numpy as np
import pandas as pd

from torch_geometric.data import HeteroData
from Environment.data import DataGenerator
from Environment.simulation import *


class State:
    def __init__(self, algorithm="RL"):
        self.algorithm = algorithm
        if algorithm == "RL":
            self.graph_feature = None
            self.pairwise_feature = None
            self.mask = None
        else:
            self.priority_idx = None
            self.mask = None

    def update(self,
               graph_feature=None,
               pairwise_feature=None,
               priority_idx=None,
               mask=None):

        if self.algorithm == "RL":
            self.graph_feature = graph_feature
            self.pairwise_feature = pairwise_feature if pairwise_feature is not None else None
            self.mask = mask if mask is not None else None
        else:
            self.priority_idx = priority_idx
            self.mask = mask


class Factory:
    def __init__(self,
                 block_data_src,
                 bay_data_src,
                 device='cpu',
                 agent1='RL',
                 agent2='RL',
                 agent3='NFP',
                 use_recording=False,
                 use_communication=True):

        self.block_data_src = block_data_src
        self.bay_data_src = bay_data_src
        self.device = device
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent3 = agent3
        self.use_recording = use_recording
        self.use_communication = use_communication

        if type(block_data_src) is DataGenerator:
            self.df_blocks = block_data_src.generate()
            self.df_bays = pd.read_excel(bay_data_src, sheet_name="bays", engine='openpyxl')
        else:
            self.df_blocks = pd.read_excel(block_data_src, sheet_name="blocks", engine='openpyxl')
            self.df_bays = pd.read_excel(bay_data_src, sheet_name="bays", engine='openpyxl')

        self.num_blocks = len(self.df_blocks)
        self.num_bays = len(self.df_bays)

        if agent1 == "RL":
            self.agent1_block_feature_dim = 8
            self.agent1_bay_feature_dim = 8

            self.agent1_meta_data = (
                ["block", "bay",],
                [("block", "block_to_block", "block"),
                 ("bay", "bay_to_block", "block"),
                 ("block", "block_to_bay", "bay")])

            self.agent1_state_size = {
                "block": self.agent1_block_feature_dim,
                "bay": self.agent1_bay_feature_dim,
            }

            self.agent1_num_nodes = {
                "block": self.num_blocks,
                "bay": self.num_bays
            }

        if agent2 == "RL":
            self.agent2_block_feature_dim = 8
            self.agent2_bay_feature_dim = 8

            self.agent2_meta_data = (
                ["block", "bay",],
                [("bay", "bay_to_bay", "bay"),
                 ("bay", "bay_to_block", "block"),
                 ("block", "block_to_bay", "bay")])

            self.agent2_state_size = {
                "block": self.agent1_block_feature_dim,
                "bay": self.agent1_bay_feature_dim,
            }

            self.agent2_num_nodes = {
                "block": self.num_blocks,
                "bay": self.num_bays
            }

    def step(self, action):
        if self.agent_mode == "agent1":
            block_id = action
            block = self.monitor.remove_from_queue(block_id, agent="agent1")

            if self.monitor.use_recording:
                self.monitor.record(self.sim_env.now,
                                    block=block.name,
                                    event="Block_Selected")

            self.agent_mode = "agent2"

        elif self.agent_mode == "agent2":
            bay_id = action
            block = self.monitor.remove_from_queue(agent="agent2")
            bay = self.bays[bay_id]

            self.source.call_for_machine_scheduling[block.id].succeed(bay.id)

            if self.monitor.use_recording:
                self.monitor.record(self.sim_env.now,
                                    block=block.name,
                                    bay=bay.name,
                                    event="Bay_Allocated")

            self.agent_mode = "agent3"

        else:
            # 공간 배치 알고리즘 추후 연결

            self.agent_mode = "agent1"

            mask = self._get_mask()
            if mask.any():
                self.monitor.set_scheduling_flag(scheduling_mode="machine_scheduling")

        done = False

        while True:
            if self.monitor.call_agent1 or self.monitor.call_agent2 or self.monitor.call_agent3:
                while self.sim_env.now in [event[0] for event in self.sim_env._queue]:
                    self.sim_env.step()

                if self.agent_mode == "agent1":
                    mask = self._get_mask()

                    if mask.any():
                        break
                    else:
                        self.monitor.call_agent1 = False
                else:
                    break

            if self.sink.num_blocks_completed == self.num_blocks:
                done = True
                break

            self.sim_env.step()

        next_local_observation = self._get_local_observation()
        reward = self._calculate_reward()

        if self.current_date != self.sim_env.now:
            self.current_date = self.sim_env.now

        return next_local_observation, reward, done

    def reset(self):
        (self.sim_env,
         self.blocks,
         self.source,
         self.bays,
         self.sink,
         self.monitor) = self._build_model()

        self.agent_mode = "agent1"
        self.current_date = 0

        while True:
            if self.monitor.call_agent1:
                while self.sim_env.now in [event[0] for event in self.sim_env._queue]:
                    self.sim_env.step()
                break
            self.sim_env.step()

        if self.current_date != self.sim_env.now:
            self.current_date = self.sim_env.now

        local_observation = self._get_local_observation()

        return local_observation

    def _get_mask(self):
        num_rows = self.num_bays
        num_columns = self.num_blocks

        mask = np.zeros((num_rows, num_columns), dtype=bool)

        if self.agent_mode == "agent1":
            blocks = [block for block in self.monitor.queue_for_agent1.values()]
            axis = 0
        elif self.agent_mode == "agent2":
            blocks = [self.monitor.queue_for_agent2]
            axis = 1
        else:
            raise Exception("Invalid agent mode")

        for block in blocks:
            process_type = block.process_type
            weight = block.weight
            breadth = block.breadth
            height = block.height
            workload_h1 = block.workload_h1
            workload_h2 = block.workload_h2

            for bay in self.bays.values():
                flag_size_constraint = (breadth <= bay.block_breadth) and (height <= bay.block_height)

                if process_type == "Final조립":
                    flag_weight_constraint = (weight <= bay.block_turnover_weight)
                else:
                    flag_weight_constraint = (weight <= bay.block_weight)

                flag_capacity_constraint = ((bay.workload_h1 + workload_h1 <= bay.capacity_h1)
                                            and (bay.workload_h2 + workload_h2 <= bay.capacity_h2))

                mask[bay.id, block.id] = flag_size_constraint & flag_weight_constraint & flag_capacity_constraint

        mask = torch.tensor(np.any(mask, axis=axis), dtype=torch.bool).to(self.device)

        return mask

    def _get_local_observation(self):
        if self.agent_mode == "agent1":
            if self.agent1 == "RL":
                pass
            else:
                priority_idx = np.zeros(self.num_blocks)

                # shortest processing time (SPT)
                if self.agent1 == "SPT":
                    for block in self.monitor.queue_for_agent1.values():
                        priority_idx[block.id] = 1 / block.duration
                # earliest due date (EDD)
                elif self.agent1 == "EDD":
                    for block in self.monitor.queue_for_agent1.values():
                        priority_idx[block.id] = 1 / block.due_date
                # modified due date (MDD)
                elif self.agent1 == "MDD":
                    for block in self.monitor.queue_for_agent1.values():
                        priority_idx[block.id] = 1 / max(block.due_date, self.current_date + block.duration)
                # least slack time (LST)
                elif self.agent1 == "LST":
                    for block in self.monitor.queue_for_agent1.values():
                        priority_idx[block.id] = 1 / (block.duration - self.current_date - block.duration) \
                            if block.duration - self.current_date - block.duration > 0 else 1
                # random (RAND)
                else:
                    for block in self.monitor.queue_for_agent1.values():
                        priority_idx[block.id] = 1

        elif self.agent_mode == "agent2":
            if self.agent2 == "RL":
                pass
            else:
                priority_idx = np.zeros(self.num_bays)

                # minimum number of blocks (MNB)
                if self.agent2 == "MNB":
                    for bay in self.bays.values():
                        priority_idx[bay.id] = 1 / len(bay.blocks_in_bay) if len(bay.blocks_in_bay) > 0 else 1
                # largest space remaining (LSR)
                elif self.agent2 == "LSR":
                    for bay in self.bays.values():
                        occupied_space_ratio = bay.occupied_space / (bay.length * bay.breadth) * 100
                        priority_idx[bay.id] = 1 / occupied_space_ratio if occupied_space_ratio > 0 else 1
                # lowest capacity remaining (LCR)
                elif self.agent2 == "LCR":
                    for bay in self.bays.values():
                        capacity_ratio_h1 = bay.workload_h1 / bay.capacity_h1 * 100
                        capacity_ratio_h2 = bay.workload_h2 / bay.capacity_h2 * 100
                        capacity_ratio_avg = (capacity_ratio_h1 + capacity_ratio_h2) / 2
                        priority_idx[bay.id] = 1 / capacity_ratio_avg if capacity_ratio_avg > 0 else 1
                # randon (RAND)
                else:
                    for bay in self.bays.values():
                        priority_idx[bay.id] = 1

        elif self.agent_mode == "agent3":
            # no-fit polygon (NFP)
            if self.agent3 == "NFP":
                pass
            # bottom left fill (BLF)
            elif self.agent3 == "BLF":
                pass
            # random (RAND)
            else:
                pass
        else:
            raise Exception("Invalid agent_mode")

        if self.agent_mode == "agent1":
            state = State(self.agent1)
            mask = self._get_mask()

            if self.agent1 == "RL":
                state.update(graph_feature=graph_feature,
                             mask=mask)
            else:
                state.update(priority_idx=priority_idx,
                             mask=mask)

        elif self.agent_mode == "agent2":
            state = State(self.agent2)
            mask = self._get_mask()

            if self.agent2 == "RL":
                state.update(graph_feature=graph_feature,
                             mask=mask)
            else:
                state.update(priority_idx=priority_idx,
                             mask=mask)

        else:
            state = None

        return state

    def _calculate_reward(self):
        pass

    def _build_model(self):
        sim_env = simpy.Environment()
        monitor = Monitor(self.use_recording)

        blocks = []
        for _, row in self.df_blocks.iterrows():
            block = Block(name=row['Block_Name'],
                          id=int(row['Block_ID']),
                          process_type=row['Process_Type'],
                          ship_type=row['Ship_Type'],
                          start_date=int(row['Start_Date']),
                          duration=int(row['Duration']),
                          due_date=float(row['Due_Date']),
                          weight=float(row['Weight']),
                          length=float(row['Length']),
                          breadth=float(row['Breadth']),
                          height=float(row['Height']),
                          workload_h1=int(row['Workload_H01']),
                          workload_h2=int(row['Workload_H02']))

            blocks.append(block)

        blocks = sorted(blocks, key=lambda x: x.start_date)

        bays = {}
        source = Source(sim_env,
                        name='Source',
                        blocks=blocks,
                        bays=bays,
                        monitor=monitor)

        sink = Sink(sim_env,
                    name='Sink',
                    monitor=monitor)

        for _, row in self.df_bays.iterrows():
            bay = Bay(sim_env,
                      name=row['Bay_Name'],
                      id=int(row['Bay_ID']),
                      team=row['Team_Name'],
                      capacity_h1=float(row['Capacity_H01']),
                      capacity_h2=float(row['Capacity_H02']),
                      length=float(row['Bay_Length']),
                      breadth=float(row['Bay_Breadth']),
                      block_breadth=float(row['Block_Breadth']),
                      block_height=float(row['Block_Height']),
                      block_weight=float(row['Block_Weight']),
                      block_turnover_weight=float(row['Block_T/O_Weight']),
                      sink=sink,
                      monitor=monitor)

            bays[bay.id] = bay

        return sim_env, blocks, source, bays, sink, monitor


if __name__ == '__main__':
    import random
    from Agent.Agent1.heuristic import BSHeuristic
    from Agent.Agent2.heuristic import BAHeuristic

    algorithm_agent1 = "SPT"
    algorithm_agent2 = "MNB"
    algorithm_agent3 = "BLF"

    agent1 = BSHeuristic(algorithm_agent1)
    agent2 = BAHeuristic(algorithm_agent2)

    # data_src = DataGenerator()
    block_data_src = "../input/block_data.xlsx"
    bay_data_src = "../input/bay_config.xlsx"
    env = Factory(block_data_src,
                  bay_data_src,
                  agent1=algorithm_agent1,
                  agent2=algorithm_agent2,
                  agent3=algorithm_agent3,
                  use_recording=True)

    step = 0
    episode_reward = 0
    random.seed(42)
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
            action_agent1 = agent1.act(state_agent1)
            next_state_agent2, reward_agent1, done = env.step(action_agent1)
            # episode_reward += reward_agent1
            # mask = state.mask.transpose(0, 1).flatten()
            # candidates = np.where(mask == True)[0]
            # action = np.random.choice(candidates)
        elif mode == "agent2":
            action_agent2 = agent2.act(state_agent2)
            next_state_agent3, reward_agent2, done = env.step(action_agent2)
            # episode_reward += reward_agent2
            # mask = state.mask
            # candidates = np.where(mask == True)[0]
            # action = np.random.choice(candidates)
        else:
            action_agent3 = None
            next_state_agent1, reward_agent3, done = env.step(action_agent3)

        if mode == "agent1":
            state_agent2 = next_state_agent2
        elif mode == "agent2":
            state_agent3 = next_state_agent3
        else:
            state_agent1 = next_state_agent1

        step += 1

        print(step, episode_reward)

        if done:
            break