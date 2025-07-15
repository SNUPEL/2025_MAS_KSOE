import torch
import simpy
import shapely
import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon
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
                 use_communication=True,
                 use_spatial_arrangement=False):

        self.block_data_src = block_data_src
        self.bay_data_src = bay_data_src
        self.device = device
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent3 = agent3
        self.use_recording = use_recording
        self.use_communication = use_communication
        self.use_spatial_arrangement = use_spatial_arrangement

        if type(block_data_src) is DataGenerator:
            self.df_blocks = block_data_src.generate()
            self.df_bays = pd.read_excel(bay_data_src, sheet_name="bays", engine='openpyxl')
        else:
            self.df_blocks = pd.read_excel(block_data_src, sheet_name="blocks", engine='openpyxl')
            self.df_bays = pd.read_excel(bay_data_src, sheet_name="bays", engine='openpyxl')

        self.num_blocks = len(self.df_blocks)
        self.num_bays = len(self.df_bays)

        self.eligibility_matrix = self._get_eligibility_matrix()

        if agent1 == "RL":
            self.block_feature_dim_agent1 = 8
            self.bay_feature_dim_agent1 = 8

            self.meta_data_agent1 = (
                ["block", "bay",],
                [("block", "block_to_block", "block"),
                 ("bay", "bay_to_block", "block"),
                 ("block", "block_to_bay", "bay")])

            self.state_size_agent1 = {
                "block": self.block_feature_dim_agent1,
                "bay": self.bay_feature_dim_agent1,
            }

            self.num_nodes_agent1 = {
                "block": self.num_blocks,
                "bay": self.num_bays
            }

        if agent2 == "RL":
            self.block_feature_dim_agent2 = 8
            self.bay_feature_dim_agent2 = 8
            self.pairwise_feature_dim = 2

            self.meta_data_agent2 = (
                ["block", "bay",],
                [("bay", "bay_to_bay", "bay"),
                 ("bay", "bay_to_block", "block"),
                 ("block", "block_to_bay", "bay")])

            self.state_size_agent2 = {
                "block": self.block_feature_dim_agent2,
                "bay": self.bay_feature_dim_agent2,
            }

            self.num_nodes_agent2 = {
                "block": 1,
                "bay": self.num_bays
            }

    def step(self, action):
        if self.agent_mode == "agent1":
            block_id = action
            _, block = self.monitor.remove_from_queue(block_id, agent="agent1")

            if self.monitor.use_recording:
                self.monitor.record(self.sim_env.now,
                                    block=block.name,
                                    event="Block_Selected")

            self.agent_mode = "agent2"

        elif self.agent_mode == "agent2":
            bay_id = action
            _, block = self.monitor.remove_from_queue(agent="agent2")
            bay = self.bays[bay_id]

            self.source.call_for_machine_scheduling[block.id].succeed(bay.id)

            if self.monitor.use_recording:
                self.monitor.record(self.sim_env.now,
                                    block=block.name,
                                    bay=bay.name,
                                    event="Bay_Allocated")

            self.agent_mode = "agent3"

        else:
            if self.use_spatial_arrangement:
                x, y = action
                bay, block = self.monitor.remove_from_queue(agent="agent3")

                bay.call_for_spatial_arrangement[block.id].succeed((x, y))

                if self.monitor.use_recording:
                    self.monitor.record(self.sim_env.now,
                                        block=block.name,
                                        bay=bay.name,
                                        x_coordinate=x,
                                        y_coordinate=y,
                                        event="Block_Located")

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

    def _get_eligibility_matrix(self):
        eligibility_matrix = np.zeros((self.num_blocks, self.num_bays), dtype=bool)

        for _, block_info in self.df_blocks.iterrows():
            for _, bay_info in self.df_bays.iterrows():
                flag_size_constraint = (int(block_info["Breadth"]) <= int(bay_info["Block_Breadth"])
                                        and int(block_info["Height"]) <= int(bay_info["Block_Height"]))

                if block_info["Process_Type"] == "Final조립":
                    flag_weight_constraint = (int(block_info["Weight"]) <= int(bay_info["Block_T/O_Weight"]))
                else:
                    flag_weight_constraint = (int(block_info["Weight"]) <= int(bay_info["Block_Weight"]))

                flag_capacity_constraint = ((int(block_info["Workload_H01"]) <= int(bay_info["Capacity_H01"])
                                            and (int(block_info["Workload_H02"]) <= int(bay_info["Capacity_H02"]))))

                eligibility_matrix[int(block_info["Block_ID"]),int(bay_info["Bay_ID"])] \
                    = flag_size_constraint & flag_weight_constraint & flag_capacity_constraint

        return eligibility_matrix

    def _get_mask(self):
        if self.agent_mode == "agent1" or self.agent_mode == "agent2":
            num_rows = self.num_bays
            num_columns = self.num_blocks

            mask = np.zeros((num_rows, num_columns), dtype=bool)

            if self.agent_mode == "agent1":
                blocks = [block for block in self.monitor.queue_for_agent1.values()]
                axis = 0
            elif self.agent_mode == "agent2":
                blocks = [self.monitor.queue_for_agent2]
                axis = 1

            for block in blocks:
                process_type = block.process_type
                weight = block.weight
                length = block.length
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

                    flag_space_constraint = (bay.occupied_space + (length * breadth)
                                             <= (bay.length * bay.breadth) * 0.8)

                    mask[bay.id, block.id] = (flag_size_constraint & flag_weight_constraint
                                              & flag_capacity_constraint & flag_space_constraint)

            mask = torch.tensor(np.any(mask, axis=axis), dtype=torch.bool).to(self.device)

        elif self.agent_mode == "agent3":
            target_bay, target_block = self.monitor.queue_for_agent3

            num_rows = len(target_bay.x_list)
            num_columns = len(target_bay.y_list)

            mask = np.zeros((num_rows, num_columns), dtype=bool)

            for i, x in enumerate(target_bay.x_list):
                for j, y in enumerate(target_bay.y_list):
                    if (x + target_block.length > target_bay.length
                            or y + target_block.breadth > target_bay.breadth):
                        continue  # 캔버스 범위 초과

                    poly = Polygon((Point(x, y),
                                    Point(x + target_block.length, y),
                                    Point(x + target_block.length, y + target_block.breadth),
                                    Point(x, y + target_block.breadth),
                                    Point(x, y)))

                    poly = shapely.affinity.scale(poly, xfact=0.99, yfact=0.99)

                    flag_collide = False
                    for item in target_bay.allocated_blocks_polygon_dict.values():
                        if poly.intersects(item):
                            flag_collide = True
                            break

                    mask[i, j] = not flag_collide

            mask = torch.tensor(mask, dtype=torch.bool).to(self.device)

        else:
            raise Exception("Invalid agent mode")

        return mask

    def _get_local_observation(self):
        if self.agent_mode == "agent1":
            if self.agent1 == "RL":
                # 노드 특성 벡터 생성
                block_feature = np.zeros((self.num_blocks, self.block_feature_dim_agent1))
                bay_feature = np.zeros((self.num_bays, self.bay_feature_dim_agent1))

                # 그래프 내 노드 간 엣지 모델링
                edge_block_to_block = [[], []]
                edge_block_to_bay, edge_bay_to_block = [[], []], [[], []]

                # node feature 추가


                # block 노드와 bay 노드 간 엣지 구성
                for _, block_info in self.df_blocks.iterrows():
                    for _, bay_info in self.df_bays.iterrows():
                        if self.eligibility_matrix[int(block_info["Block_ID"]),int(bay_info["Bay_ID"])]:
                            edge_block_to_bay[0].append(int(block_info["Block_ID"]))
                            edge_block_to_bay[1].append(int(bay_info["Bay_ID"]))
                            edge_bay_to_block[0].append(int(bay_info["Bay_ID"]))
                            edge_bay_to_block[1].append(int(block_info["Block_ID"]))

                # block 노드 간 엣지 구성
                for block_from in self.monitor.queue_for_agent1.values():
                    for block_to in self.monitor.queue_for_agent1.values():
                        edge_block_to_block[0].append(block_from.id)
                        edge_block_to_block[1].append(block_to.id)

                # 이종 그래프 객체 생성
                block_feature = torch.from_numpy(block_feature).type(torch.float32).to(self.device)
                bay_feature = torch.from_numpy(bay_feature).type(torch.float32).to(self.device)
                edge_block_to_block = torch.from_numpy(np.array(edge_block_to_block)).type(torch.long).to(self.device)
                edge_block_to_bay = torch.from_numpy(np.array(edge_block_to_bay)).type(torch.long).to(self.device)
                edge_bay_to_block = torch.from_numpy(np.array(edge_bay_to_block)).type(torch.long).to(self.device)

                graph_feature = HeteroData()
                graph_feature["block"].x = block_feature
                graph_feature["bay"].x = bay_feature
                graph_feature["block", "block_to_block", "block"].edge_index = edge_block_to_block
                graph_feature["block", "block_to_bay", "bay"].edge_index = edge_block_to_bay
                graph_feature["bay", "bay_to_block", "block"].edge_index = edge_bay_to_block

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
                # 노드 특성 벡터 생성
                block_feature = np.zeros((1, self.block_feature_dim_agent2))
                bay_feature = np.zeros((self.num_bays, self.bay_feature_dim_agent2))
                # 노드 조합 특성 벡터 생성
                pairwise_feature = np.zeros((1, self.num_bays, self.pairwise_feature_dim))

                # node feature 추가

                # pairwise feature 추가


                # 그래프 내 노드 간 엣지 모델링
                edge_bay_to_bay = [[], []]
                edge_block_to_bay, edge_bay_to_block = [[], []], [[], []]

                # block 노드와 bay 노드 간 엣지 구성
                target_block = self.monitor.queue_for_agent2
                for _, bay_info in self.df_bays.iterrows():
                    if self.eligibility_matrix[target_block.id, int(bay_info["Bay_ID"])]:
                        edge_block_to_bay[0].append(0)
                        edge_block_to_bay[1].append(int(bay_info["Bay_ID"]))
                        edge_bay_to_block[0].append(int(bay_info["Bay_ID"]))
                        edge_bay_to_block[1].append(0)

                # bay 노드 간 엣지 구성
                for bay_from in self.bays.values():
                    for bay_to in self.bays.values():
                        edge_bay_to_bay[0].append(bay_from.id)
                        edge_bay_to_bay[1].append(bay_to.id)

                # 이종 그래프 객체 생성
                block_feature = torch.from_numpy(block_feature).type(torch.float32).to(self.device)
                bay_feature = torch.from_numpy(bay_feature).type(torch.float32).to(self.device)
                edge_bay_to_bay = torch.from_numpy(np.array(edge_bay_to_bay)).type(torch.long).to(self.device)
                edge_block_to_bay = torch.from_numpy(np.array(edge_block_to_bay)).type(torch.long).to(self.device)
                edge_bay_to_block = torch.from_numpy(np.array(edge_bay_to_block)).type(torch.long).to(self.device)

                graph_feature = HeteroData()
                graph_feature["block"].x = block_feature
                graph_feature["bay"].x = bay_feature
                graph_feature["bay", "bay_to_bay", "bay"].edge_index = edge_bay_to_bay
                graph_feature["block", "block_to_bay", "bay"].edge_index = edge_block_to_bay
                graph_feature["bay", "bay_to_block", "block"].edge_index = edge_bay_to_block

                pairwise_feature = torch.from_numpy(pairwise_feature).type(torch.float32).to(self.device)

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
            if self.use_spatial_arrangement:
                target_bay, target_block = self.monitor.queue_for_agent3
                priority_idx = np.zeros((len(target_bay.x_list), len(target_bay.y_list)))

                # no-fit polygon (NFP)
                if self.agent3 == "NFP":
                    pass
                # minimum distance to origin (MDO)
                elif self.agent3 == "MDO":
                    for i, x in enumerate(target_bay.x_list):
                        for j, y in enumerate(target_bay.y_list):
                            priority_idx[i, j] = 1 / np.sqrt(x ** 2 + y ** 2) if np.sqrt(x ** 2 + y ** 2) != 0 else 1
                # bottom left fill (BLF)
                elif self.agent3 == "BLF":
                    for i, x in enumerate(target_bay.x_list):
                        for j, y in enumerate(target_bay.y_list):
                            priority_idx[i, j] = 1 / (100 * y + x) if 100 * y + x != 0 else 1
                # random (RAND)
                else:
                    for i, x in enumerate(target_bay.x_list):
                        for j, y in enumerate(target_bay.y_list):
                            priority_idx[i, j] = 1
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
                             pairwise_feature=pairwise_feature,
                             mask=mask)
            else:
                state.update(priority_idx=priority_idx,
                             mask=mask)

        elif self.agent_mode == "agent3":
            if self.use_spatial_arrangement:
                state = State(self.agent3)
                mask = self._get_mask()

                state.update(priority_idx=priority_idx,
                             mask=mask)
            else:
                state = None

        else:
            raise Exception("Invalid agent_mode")

        return state

    def _calculate_reward(self):
        return 0.0

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

            # mask = state_agent1.mask
            # candidates = np.where(mask == True)[0]
            # action_agent1 = np.random.choice(candidates)
            # next_state_agent2, reward_agent1, done = env.step(action_agent1)
        elif mode == "agent2":
            action_agent2 = agent2.act(state_agent2)
            next_state_agent3, reward_agent2, done = env.step(action_agent2)
            # episode_reward += reward_agent2

            # mask = state_agent2.mask
            # candidates = np.where(mask == True)[0]
            # action_agent2 = np.random.choice(candidates)
            # next_state_agent3, reward_agent2, done = env.step(action_agent2)
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