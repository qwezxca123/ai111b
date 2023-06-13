import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)

def predict(a, xt):
    return a[0] + a[1] * xt

def MSE(a, x, y):
    total = 0
    for i in range(len(x)):
        total += (y[i] - predict(a, x[i])) ** 2
    return total

def optimize():
    p = [0.0, 0.0]  # 初始參數值
    best_loss = MSE(p, x, y)  # 初始最佳損失值

    # 迭代次數和停止條件
    max_iterations = 1000
    epsilon = 1e-6

    for i in range(max_iterations):
        # 生成當前參數的鄰近解
        delta = 0.01  # 鄰域的步長
        neighbors = [
            [p[0] + delta, p[1]],  # 增加 a[0]
            [p[0] - delta, p[1]],  # 減少 a[0]
            [p[0], p[1] + delta],  # 增加 a[1]
            [p[0], p[1] - delta]   # 減少 a[1]
        ]

        # 選擇損失函數最小的鄰近解
        for neighbor in neighbors:
            neighbor_loss = MSE(neighbor, x, y)
            if neighbor_loss < best_loss:
                p = neighbor
                best_loss = neighbor_loss

        # 檢查是否收斂
        if best_loss < epsilon:
            break

    return p

p = optimize()

# Print the linear function
print(f"線性函數：y = {p[0]} + {p[1]} * x")

# Plot the graph
y_predicted = list(map(lambda t: p[0] + p[1] * t, x))
print('y_predicted=', y_predicted)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()
