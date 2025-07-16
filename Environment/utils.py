# blf_test_script.py
import math
import random
import shapely
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point


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


def load_analysis(file_path, graph=False):
    df_blocks = pd.read_excel(file_path, sheet_name='blocks', engine='openpyxl')
    df_blocks["finish_date"] = df_blocks["start_date"] + df_blocks["duration"] - 1

    start = df_blocks["start_date"].min()
    finish = df_blocks["finish_date"].max()

    timeline = np.arange(start, finish)
    load_block_num = np.zeros(finish - start)
    load_workload_h1 = np.zeros(finish - start)
    load_workload_h2 = np.zeros(finish - start)

    for i, row in df_blocks.iterrows():
        load_block_num[int(row["start_date"]):int(row["finish_date"])] += 1
        load_workload_h1[int(row["start_date"]):int(row["finish_date"])] += int(row["workload_h1"]) / int(row["duration"])
        load_workload_h2[int(row["start_date"]):int(row["finish_date"])] += int(row["workload_h2"]) / int(row["duration"])

    if graph:
        fig, ax = plt.subplots(1, figsize=(16, 6))
        ax.plot(timeline, load_block_num)
        plt.show()
        plt.close()

    return np.max(load_block_num), np.max(load_workload_h1), np.max(load_workload_h2)


if __name__ == "__main__":
    # main()
    file_path = "../input/validation/instance-1.xlsx"
    block_num_max, workload_h1_max, workload_h2_max = load_analysis(file_path, graph=True)
    print(block_num_max, workload_h1_max, workload_h2_max)