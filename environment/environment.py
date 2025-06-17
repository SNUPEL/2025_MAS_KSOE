import torch
import simpy
import copy
import numpy as np
import pandas as pd

from torch_geometric.data import HeteroData
from environment.data import DataGenerator
from environment.simulation import *


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
    def __init__(self, data_src,
                 device='cpu',
                 agent1='RL',
                 agent2='RL',
                 agent3='NFP',
                 use_recording=False,
                 use_communication=True):

        self.data_src = data_src
        self.device = device
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent3 = agent3
        self.use_recording = use_recording
        self.use_communication = use_communication

        if type(data_src) is DataGenerator:
            self.df_blocks, self.df_bays, self.df_teams = data_src.generate()
        else:
            self.df_blocks = pd.read_excel(data_src, sheet_name="blocks", engine='openpyxl')
            self.df_bays = pd.read_excel(data_src, sheet_name="bays", engine='openpyxl')
            self.df_teams = pd.read_excel(data_src, sheet_name="teams", engine='openpyxl')

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

            mask = self._get_mask()
            if mask.any():
                self.monitor.set_scheduling_flag(scheduling_mode="machine_scheduling")

            self.agent_mode = "agent1"

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
        pass

    def _get_local_observation(self):
        pass

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
                          start_date=int(row['Start Date']),
                          duration=int(row['Duration']),
                          due_date=None,
                          weight=float(row['W']),
                          length=float(row['L']),
                          breadth=float(row['B']),
                          height=float(row['H']),
                          workload_h1=int(row['H01']),
                          workload_h2=int(row['H02']))

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
            df_temp = self.df_teams[self.df_teams['Name'] == row['Team_Name']]
            team = Team(name=df_temp['Team_Name'],
                        num_workers_h1=int(df_temp['Num_Workers_H01']),
                        num_workers_h2=int(df_temp['Num_Workers_H01']),
                        capacity_h1=float(df_temp['Capacity_H01']),
                        capacity_h2=float(df_temp['Capacity_H02']))

            bay = Bay(sim_env,
                      name=row['Bay_Name'],
                      id=int(row['Bay_ID']),
                      team=team,
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