# blf_test_script.py
import math
import random
import shapely
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point

from Environment.data import DataGenerator
from data import *


class MiniBlock:
    def __init__(self, length, breadth):
        self.length = length
        self.breadth = breadth
        self.x = None
        self.y = None

    def place(self, x, y):
        self.x = x
        self.y = y

    def get_polygon(self):
        return Polygon([
            (self.x, self.y),
            (self.x + self.length, self.y),
            (self.x + self.length, self.y + self.breadth),
            (self.x, self.y + self.breadth),
            (self.x, self.y)
        ])


class BLFAllocator:
    def __init__(self, width=100, height=20, step=1):
        self.canvas_width = width
        self.canvas_height = height
        self.x_list = list(range(0, width, step))
        self.y_list = list(range(0, height, step))
        self.allocated_blocks_polygon_list = []

    def _BLF_algorithm(self, block):
        candidates = []
        for x in self.x_list:
            for y in self.y_list:
                if x + block.length > self.canvas_width or y + block.breadth > self.canvas_height:
                    continue  # 캔버스 범위 초과

                poly = Polygon([
                    (x, y),
                    (x + block.length, y),
                    (x + block.length, y + block.breadth),
                    (x, y + block.breadth),
                    (x, y)
                ])
                poly = shapely.affinity.scale(poly, xfact=0.99, yfact=0.99)
                collides = False
                for item in self.allocated_blocks_polygon_list:
                    if poly.intersects(item):
                        collides = True
                        break
                if not collides:
                    candidates.append((x, y))
                    break  # 가장 아래쪽 y 찾으면 다음 x로

        if not candidates:
            raise RuntimeError("No valid placement found for block.")

        # 원점에서의 L2 거리 기준
        best = sorted(candidates, key=lambda p: math.hypot(p[0], p[1]))[0]
        return best

    def allocate(self, blocks):
        for block in blocks:
            x, y = self._BLF_algorithm(block)
            block.place(x, y)
            self.allocated_blocks_polygon_list.append(block.get_polygon())


def visualize_blocks(blocks, canvas_width, canvas_height):
    fig, ax = plt.subplots()
    for block in blocks:
        rect = plt.Rectangle((block.x, block.y), block.length, block.breadth,
                             edgecolor='black', facecolor='skyblue', alpha=0.7)
        ax.add_patch(rect)
        cx = block.x + block.length / 2
        cy = block.y + block.breadth / 2
        ax.text(cx, cy, f"({block.length}×{block.breadth})", fontsize=6, ha='center', va='center')

    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.set_aspect('equal')
    ax.set_title('Bottom-Left-Fill Block Placement')
    plt.grid(True)
    plt.show()


def main():
    # 배치할 블록 문제 정의: (length, breadth)
    problem_list = [(10, 5), (15, 10), (20, 7), (5, 5), (8, 12), (6, 6), (9, 4), (12, 8)]

    # 블록 인스턴스화
    blocks = [MiniBlock(length, breadth) for length, breadth in problem_list]

    # BLF 배치 실행
    allocator = BLFAllocator(width=100, height=20, step=1)
    allocator.allocate(blocks)

    # 결과 시각화
    visualize_blocks(blocks, canvas_width=100, canvas_height=100)


def calculate_total_weighted_tardiness(log):
    total_weighted_tardiness = 0.0

    for importance, finish_date, due_date in log.values():
        total_weighted_tardiness += importance * max(finish_date - due_date, 0)

    return total_weighted_tardiness


def calculate_average_workload_deviation(log, bay_capacity):
    time_horizon = max([finish_date for _, _, finish_date, _ in log.values()])
    workload = np.zeros((len(bay_capacity), int(time_horizon)))

    for bay_id, start_date, finish_date, daily_workload in log.values():
        workload[bay_id, int(start_date):int(finish_date) + 1] += daily_workload

    for bay_id, bay_capacity in enumerate(bay_capacity):
        workload[bay_id, :] = workload[bay_id, :] / bay_capacity

    workload_deviation = workload.std(axis=0)
    average_workload_deviation = workload_deviation.mean()

    return average_workload_deviation


def load_analysis(file_path, graph=False):
    df_blocks = pd.read_excel(file_path, sheet_name='blocks', engine='openpyxl')
    df_blocks["finish_date"] = df_blocks["start_date"] + df_blocks["duration"] - 1

    start = df_blocks["start_date"].min()
    finish = df_blocks["finish_date"].max()

    timeline = np.arange(start, finish)
    block = np.zeros(finish - start)
    area = np.zeros(finish - start)
    workload_h1 = np.zeros(finish - start)
    workload_h2 = np.zeros(finish - start)

    for i, row in df_blocks.iterrows():
        block[int(row["start_date"]):int(row["finish_date"])] += 1
        area[int(row["start_date"]):int(row["finish_date"])] += int(row["length"]) * int(row["breadth"])
        workload_h1[int(row["start_date"]):int(row["finish_date"])] += int(row["workload_h1"]) / int(row["duration"])
        workload_h2[int(row["start_date"]):int(row["finish_date"])] += int(row["workload_h2"]) / int(row["duration"])


    if graph:
        fig, ax = plt.subplots(1, figsize=(16, 6))
        ax.plot(timeline, block)
        plt.show()
        plt.close()

    return np.max(block), np.max(area), np.max(workload_h1), np.max(workload_h2), np.average(block), np.average(area), np.average(workload_h1), np.average(workload_h2)


if __name__ == "__main__":
    # main()
    # file_path = "../input/validation/instance-1.xlsx"
    # block_max, area_max, workload_h1_max, workload_h2_max = load_analysis(file_path, graph=True)
    # print(block_max, area_max, workload_h1_max, workload_h2_max)

    df_result = []
    columns = ['instance', 'num_blocks', 'start_date', 'finish_date',
               'block_max', 'area_max', 'workload_h1_max', 'workload_h2_max',
               'block_avg', 'area_avg', 'workload_h1_avg', 'workload_h2_avg']

    block_data_path = "../input/configurations/block_data.xlsx"
    bay_data_path = "../input/configurations/bay_data.xlsx"

    data_gen = DataGenerator(block_data_path, bay_data_path)

    for i in range(1, 21):
        file_path = f"../input/validation/instance-{i}.xlsx"
        instance_df = pd.read_excel(file_path)
        instance_df["finish_date"] = instance_df["start_date"] + instance_df["duration"] - 1

        num_blocks = instance_df.shape[0]
        start_date = instance_df["start_date"].min()
        finish_date = instance_df["finish_date"].max()

        block_max, area_max, workload_h1_max, workload_h2_max, block_avg, area_avg, workload_h1_avg, workload_h2_avg = load_analysis(file_path, graph=False)
        print(block_max, area_max, workload_h1_max, workload_h2_max, block_avg, area_avg, workload_h1_avg, workload_h2_avg)

        row = [i, num_blocks, start_date, finish_date,
               block_max, area_max, workload_h1_max, workload_h2_max,
               block_avg, area_avg, workload_h1_avg, workload_h2_avg]

        df_result.append(row)

    df_result = pd.DataFrame(df_result, columns=columns)

    save_path = f'../data/validation_result/validation_result_iat{data_gen.iat_avg}_buffer{data_gen.buffer_avg}.xlsx'

    df_result.to_excel(save_path, sheet_name=f'iat {data_gen.iat_avg}_buffer {data_gen.buffer_avg}', index=False)