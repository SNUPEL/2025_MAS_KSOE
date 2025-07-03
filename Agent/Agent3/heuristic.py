import numpy as np

# TODO: 블록 배치에 알맞는 단어 선택? 임시로 Block Placement에 해당하는 단어로 할당함
class BPHeuristic:
    """
    블록 배치(Block Placement)를 결정하는 휴리스틱 클래스입니다.
    """
    def __init__(self, name):
        self.name = name

    def act(self, state):
        priority_idx = state.priority_idx
        mask = state.mask.cpu().numpy()
        priority_idx[~mask] = 0.0

        candidates = np.where(priority_idx == np.max(priority_idx))[0]
        action = np.random.choice(candidates)

        return int(action)