import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def visualize_state_action_from_json(
    json_path,
    output_dir,
    episode,
    step_range,
    dpi=300
):
    # 1. JSON 파일 읽기
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 2. 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    for step in step_range:
        step_str = str(step)
        if step_str not in data:
            print(f"[Warning] Step {step_str} not found in JSON.")
            continue

        entry = data[step_str]
        block_node = entry["state"]["node_Block"]     # shape: [1, 8]
        bay_node = entry["state"]["node_Bay"]         # shape: [17, 3]
        action_prob = entry["action"]                 # shape: [17]

        # 3. Plot 설정
        fig, axs = plt.subplots(nrows=3, figsize=(10, 5), gridspec_kw={'height_ratios': [1, 3, 1]})

        # Block Node 테이블 (1x8)
        df_block = pd.DataFrame(np.round(block_node, 4), columns=[f'B{i}' for i in range(1, 9)])
        axs[0].axis('off')
        axs[0].table(cellText=df_block.values,
                     colLabels=df_block.columns,
                     loc='center',
                     cellLoc='center')

        # Bay Node 테이블 (17x3)
        df_bay = pd.DataFrame(np.round(bay_node, 4).transpose(), columns=['Bay'+str(i) for i in range(17)])
        axs[1].axis('off')
        axs[1].table(cellText=df_bay.values,
                     rowLabels=[f'Feature{i}' for i in range(1, 4)],
                     loc='center',
                     cellLoc='center')

        # Action prob 테이블 (1x17)
        df_action = pd.DataFrame([np.round(action_prob, 4)], columns=[f'Bay{i}' for i in range(1, 18)])
        axs[2].axis('off')
        axs[2].table(cellText=df_action.values,
                     colLabels=df_action.columns,
                     loc='center',
                     cellLoc='center')

        # plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)  # 서브플롯 간 여백 줄이기
        plt.tight_layout(pad=0.2)  # 전체 패딩 최소화
        fig.patch.set_facecolor('white')  # 배경 흰색 (투명 방지)

        # 4. 저장
        save_path = os.path.join(output_dir, f"episode_{episode}_step_{step_str}.png")
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
        print(f"[Saved] {save_path}")

# 예시 호출
visualize_state_action_from_json(
    json_path="../output/train/SARL/None-RL-None/20250729_19h_17m_7s/log/Episode60_MDP.json",
    output_dir='../output/train/SARL/None-RL-None/20250729_19h_17m_7s',
    episode = 60,
    step_range=range(1, 100),
    dpi=300
)
