from typing import List

class Box:
    def __init__(self, center: List[float], size: List[float], ID: int):
        self.size = size
        self.center = center
        self.color = [.2, .2, .2]
        self.ID = ID

    def to_g(self) -> str:
        size3 = self.size + [1]
        center3 = self.center + [0.5]
        out = ("obs{} {{X: <t({})>, shape:ssBox, size:[{}  0.05],"
               "color:[{}], contact}}\n").format(
            self.ID,
            ' '.join(map(str, center3)),
            ' '.join(map(str, size3)),
            ' '.join(map(str, self.color)))

        return out

