import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


# --- ФУНКЦИИ ЛР4 ВАРИАНТ 1 ---
def f(X):
    x1, x2 = X
    return 2 * x1 ** 2 + 2 * x1 * x2 + 3 * x2 ** 2 - 10 * x1 - 10 * x2 + 15


def grad_f(X):
    x1, x2 = X
    return np.array([4 * x1 + 2 * x2 - 10, 2 * x1 + 6 * x2 - 10])


# --- МЕТОДЫ ОПТИМИЗАЦИИ (Функции вместо классов) ---

def coordinate_descent(x0, eps=0.0001, max_iter=50):
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for _ in range(max_iter):
        x_prev = x.copy()

        # Спуск по x1
        res1 = minimize_scalar(lambda alpha: f([x[0] + alpha, x[1]]))
        x[0] += res1.x
        history.append(x.copy())

        # Спуск по x2
        res2 = minimize_scalar(lambda alpha: f([x[0], x[1] + alpha]))
        x[1] += res2.x
        history.append(x.copy())

        if np.linalg.norm(x - x_prev) < eps:
            break
    return np.array(history)


def gradient_descent(x0, h=0.1, eps=0.0001, max_iter=50):
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for _ in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < eps:
            break
        x = x - h * g
        history.append(x.copy())

    return np.array(history)


def steepest_descent(x0, eps=0.0001, max_iter=50):
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for _ in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < eps:
            break

        # Одномерная минимизация для поиска оптимального шага (из лекции)
        res = minimize_scalar(lambda h: f(x - h * g))
        x = x - res.x * g
        history.append(x.copy())

    return np.array(history)


# --- ЗАПУСК И ВИЗУАЛИЗАЦИЯ (ДЗ 2.1) ---
x_start = [-2.0, -2.0]

hist_coord = coordinate_descent(x_start)
hist_grad = gradient_descent(x_start, h=0.1)
hist_steep = steepest_descent(x_start)

# Построение графиков с линиями уровня
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
X1, X2 = np.meshgrid(np.linspace(-3, 4, 100), np.linspace(-3, 4, 100))
Z = f([X1, X2])

histories = [("Покоординатный", hist_coord),
             ("Градиентный (h=0.1)", hist_grad),
             ("Наискорейший", hist_steep)]

for ax, (title, hist) in zip(axes, histories):
    ax.contour(X1, X2, Z, levels=30, cmap='viridis')
    ax.plot(hist[:, 0], hist[:, 1], 'r.-', label='Траектория')
    ax.plot(hist[0, 0], hist[0, 1], 'bo', label='Старт')
    ax.plot(hist[-1, 0], hist[-1, 1], 'g*', markersize=10, label='Минимум')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()

# --- МЕТОД НЬЮТОНА (ДЗ 2.2) ---
print("\n=== Метод Ньютона (ДЗ 2.2) ===")


def grad_z(X):
    x, y = X
    return np.array([3 * x ** 2 + 6 * x, 3 * y ** 2 - 28 * y + 64])


def hessian_z(X):
    x, y = X
    return np.array([[6 * x + 6, 0], [0, 6 * y - 28]])


x_newton = np.array([-0.5, 5.5], dtype=float)  # M0
for i in range(10):
    g = grad_z(x_newton)
    if np.linalg.norm(g) < 0.0001:
        break
    H_inv = np.linalg.inv(hessian_z(x_newton))
    x_newton = x_newton - H_inv @ g
    print(f"Итерация {i + 1}: x = {x_newton[0]:.4f}, y = {x_newton[1]:.4f}")

print(f"Найденный оптимум (Ньютон): x = {x_newton[0]:.4f}, y = {x_newton[1]:.4f}")