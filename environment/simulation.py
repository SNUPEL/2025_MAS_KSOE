import pandas as pd


class Block:
    def __init__(self,
                 name=None,
                 id=None,
                 process_type=None,
                 ship_type=None,
                 start_date=None,
                 duration=None,
                 due_date=None,
                 weight=None,
                 length=None,
                 breadth=None,
                 height=None,
                 workload_h1=None,
                 workload_h2=None):

        self.name = name
        self.id = id
        self.process_type = process_type
        self.ship_type = ship_type
        self.start_date = start_date
        self.duration = duration
        self.due_date = due_date
        self.weight = weight
        self.length = length
        self.breadth = breadth
        self.height = height
        self.workload_h1 = workload_h1
        self.workload_h2 = workload_h2

        self.allocated_bay = None


class Team:
    def __init__(self,
                 name=None,
                 num_workers_h1=None,
                 num_workers_h2=None,
                 capacity_h1=None,
                 capacity_h2=None):

        self.name = name
        self.num_workers_h1 = num_workers_h1
        self.num_workers_h2 = num_workers_h2
        self.capacity_h1 = capacity_h1
        self.capacity_h2 = capacity_h2


class Source:
    def __init__(self,
                 env,
                 name='Source',
                 blocks=None,
                 bays=None,
                 monitor=None):

        self.env = env
        self.name = name
        self.blocks = blocks
        self.bays = bays
        self.monitor = monitor

        self.sent = 0
        self.call_for_machine_scheduling = {}
        self.process = env.process(self._generate())

    def _generate(self):
        for block in self.blocks:
            self.monitor.blocks_unscheduled[block.id] = block

        while True:
            block = self.blocks[self.sent]

            inter_arrival_time = block.start_date - self.env.now
            if inter_arrival_time > 0:
                yield self.env.timeout(inter_arrival_time)

            self.env.process(self._run(block))

            self.sent += 1
            if len(self.blocks) == self.sent:
                break

    def _run(self, block):
        self.monitor.add_to_queue(block, agent="agent1")
        self.monitor.set_scheduling_flag(scheduling_mode="machine_scheduling")
        self.call_for_machine_scheduling[block.id] = self.env.event()

        bay_name = yield self.call_for_machine_scheduling[block.id]
        block.allocated_bay = bay_name

        self.bays[bay_name].put(block)

        del self.call_for_machine_scheduling[block.id]
        del self.monitor.blocks_unscheduled[block.id]


class Bay:
    def __init__(self,
                 env,
                 name=None,
                 id=None,
                 team=None,
                 length=None,
                 breadth=None,
                 block_breadth=None,
                 block_height=None,
                 block_weight=None,
                 block_turnover_weight=None,
                 sink=None,
                 monitor=None):

        self.env = env
        self.name = name
        self.id = id
        self.team = team
        self.length = length
        self.breadth = breadth
        self.block_breadth = block_breadth
        self.block_height = block_height
        self.block_weight = block_weight
        self.block_turnover_weight = block_turnover_weight
        self.sink = sink
        self.monitor = monitor

        self.occupied_space = 0
        self.workload_h1 = 0
        self.workload_h2 = 0

        self.processes = {}
        self.blocks_in_bay = {}
        self.call_for_spatial_arrangement = {}

    def put(self, block):
        self.monitor.blocks_working[block.id] = block

        self.processes[block.id] = self.env.process(self._work(block))
        self.blocks_in_bay[block.id] = block

        self.occupied_space += block.length * block.breadth
        self.workload_h1 += block.workload_h1
        self.workload_h2 += block.workload_h2

    def _work(self, block):
        # 공간 배치 알고리즘 추후 연결
        # self.monitor.add_to_queue(block, agent="agent3")
        # self.monitor.set_scheduling_flag(scheduling_mode="spatial_arrangement")
        #
        # self.call_for_spatial_arrangement[block.id] = self.env.event()
        # x, y = yield self.call_for_spatial_arrangement[block.id]
        #
        # del self.call_for_spatial_arrangement[block.id]

        if self.monitor.use_recording:
            self.monitor.record(self.env.now,
                                block=block.name,
                                bay=self.name,
                                team=self.team.name,
                                event="Working_Started")

        yield self.env.timeout(block.duration)

        if self.monitor.use_recording:
            self.monitor.record(self.env.now,
                                block=block.name,
                                bay=self.name,
                                team=self.team.name,
                                event="Working_Finished")

        del self.monitor.blocks_working[block.id]
        self.occupied_space -= block.length * block.breadth
        self.workload_h1 -= block.workload_h1
        self.workload_h2 -= block.workload_h2

        self.sink.put(block)


class Sink:
    def __init__(self,
                 env,
                 name='Sink',
                 monitor=None):

        self.env = env
        self.name = name
        self.monitor = monitor

        self.num_blocks_completed = 0
        self.completion_date = 0

    def put(self, block):
        self.monitor.blocks_done[block.id] = block

        self.num_blocks_completed += 1
        self.completion_date = self.env.now


class Monitor:
    def __init__(self,
                 use_recording=True):

        self.use_recording = use_recording

        self.queue_for_agent1 = {}
        self.queue_for_agent2 = None
        self.queue_for_agent3 = None

        self.call_agent1 = False
        self.call_agent2 = False
        self.call_agent3 = False

        self.time = []
        self.block = []
        self.bay = []
        self.team = []
        self.event = []

    def set_scheduling_flag(self,
                            scheduling_mode='machine_scheduling'):

        if scheduling_mode == 'machine_scheduling':
            self.call_agent1 = True
        elif scheduling_mode == 'spatial_arrangement':
            self.call_agent3 = True
        else:
            print("Invalid scheduling mode")

    def add_to_queue(self,
                     block,
                     agent="agent1"):

        if agent == "agent1":
            self.queue_for_agent1[block.id] = block
        elif agent == "agent2":
            self.queue_for_agent2 = block
        elif agent == "agent3":
            self.queue_for_agent3 = block
        else:
            print("Invalid agent")

    def remove_from_queue(self,
                          block_id=None,
                          agent='agent1'):

        if agent == "agent1":
            assert block_id is not None

            block = self.queue_for_agent1[block_id]
            del self.queue_for_agent1[block_id]
            self.call_agent1 = False

            self.queue_for_agent2 = block
            self.call_agent2 = True

        elif agent == "agent2":
            block = self.queue_for_agent2
            self.queue_for_agent2 = None
            self.call_agent2 = False

        elif agent == "agent3":
            block = self.queue_for_agent3
            self.queue_for_agent3 = None
            self.call_agent3= False

        else:
            print("Invalid scheduling mode")

        return block

    def record(self,
               time,
               block=None,
               bay=None,
               team=None,
               event=None):

        self.time.append(time)
        self.block.append(block)
        self.bay.append(bay)
        self.team.append(team)
        self.event.append(event)

    def get_logs(self,
                 file_path=None):

        df_log = pd.DataFrame(columns=['Time', 'Block', 'Bay', 'Team', 'Event'])

        df_log['Time'] = self.time
        df_log['Block'] = self.block
        df_log['Bay'] = self.bay
        df_log['Team'] = self.team
        df_log['Event'] = self.event

        if file_path is not None:
            df_log.to_excel(file_path, sheet_name="logs", index=False)

        return df_log