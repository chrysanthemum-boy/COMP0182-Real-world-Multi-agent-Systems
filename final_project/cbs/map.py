import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def plot_rrt_path(tree, path, num, length=10, width=10, obstacles=None, start=None, goal=None):
    """
    绘制RRT生成的路径以及障碍物、起点和目标点，并添加网格
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制障碍物（如果有的话）
    if obstacles:
        for obs in obstacles:
            ax.add_patch(Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, color="black", alpha=1))
        # 绘制额外的一圈障碍物（位于地图四周）
    for i in range(length):
        ax.add_patch(Rectangle((i - 0.5, -1 - 0.5), 1, 1, color="gray", alpha=0.5))  # 底边
        ax.add_patch(Rectangle((i - 0.5, width - 0.5), 1, 1, color="gray", alpha=0.5))  # 顶边

    for i in range(width):
        ax.add_patch(Rectangle((-1 - 0.5, i - 0.5), 1, 1, color="gray", alpha=0.5))  # 左边
        ax.add_patch(Rectangle((length - 0.5, i - 0.5), 1, 1, color="gray", alpha=0.5))  # 右边

        # 四个角的障碍物
    ax.add_patch(Rectangle((-1 - 0.5, -1 - 0.5), 1, 1, color="gray", alpha=0.5))  # 左下角
    ax.add_patch(Rectangle((length - 0.5, -1 - 0.5), 1, 1, color="gray", alpha=0.5))  # 右下角
    ax.add_patch(Rectangle((-1 - 0.5, width - 0.5), 1, 1, color="gray", alpha=0.5))  # 左上角
    ax.add_patch(Rectangle((length - 0.5, width - 0.5), 1, 1, color="gray", alpha=0.5))  # 右上角

    grid = np.zeros((10, 10), dtype=int)

    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.imshow(grid, cmap="Greys", origin="lower")

    # 绘制起点和目标点
    if start:
        ax.plot(start[0], start[1], 'go', label='Start', markersize=10)
    if goal:
        ax.plot(goal[0], goal[1], 'ro', label='Goal', markersize=10)

    # 绘制路径
    if path:
        path_x = [state["x"] for state in path]
        path_y = [state["y"] for state in path]
        ax.plot(path_x, path_y, 'b-', label='Path', linewidth=2)
        ax.plot(path_x, path_y, 'b.', label='Path Points')

    # 绘制树
    if tree:
        tree_x = [state.location.x for state in tree]
        tree_y = [state.location.y for state in tree]
        ax.plot(tree_x, tree_y, 'k.', label='RRT Tree', alpha=0.2)

    # 设置图形显示参数
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))
    ax.set_title("RRT & CBS Path Planning")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc='lower right')

    # 保存图像
    plt.savefig(f"rrt{num + 1}.png", dpi=300)
    plt.show()
# 示例数据
obstacles = [
    (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
    (0, 4), (1, 4), (2, 4), (3, 4), (4, 4),
    (7, 4), (8, 4), (9, 4),
    (3, 6), (4, 6), (5, 6), (6, 6), (7, 6),
    (0, 8), (1, 8), (2, 8), (5, 8), (6, 8),
    (7, 8), (8, 8), (9, 8)
]
start = (0, 0)
goal = (9, 9)
tree = [
    # 树中示例节点
    type('Node', (), {'location': type('Point', (), {'x': 1, 'y': 1})})(),
    type('Node', (), {'location': type('Point', (), {'x': 2, 'y': 2})})()
]
path = [{"x": 0, "y": 0}, {"x": 2, "y": 2}, {"x": 5, "y": 5}, {"x": 9, "y": 9}]

# 调用函数
plot_rrt_path(tree, path, num=0, obstacles=obstacles, start=start, goal=goal)
