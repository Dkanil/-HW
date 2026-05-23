import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from abc import ABC, abstractmethod


# ==============================================================
# БЛОК ДЗ 2.3: ОБЪЕКТНАЯ МОДЕЛЬ (КЛАССЫ И НАСЛЕДОВАНИЕ)
# ==============================================================

class OptimizationProblem:
    """Класс, инкапсулирующий целевую функцию и её градиент (Агрегация)"""

    def __init__(self, func, grad):
        self.func = func
        self.grad = grad


class Optimizer(ABC):
    """Абстрактный базовый класс для методов оптимизации"""

    def __init__(self, eps=0.0001, max_iter=50):
        self.eps = eps
        self.max_iter = max_iter
        self.history = []

    @abstractmethod
    def optimize(self, problem: OptimizationProblem, x0):
        """Абстрактный метод, который должен быть переопределен в наследниках"""
        pass


class CoordinateDescent(Optimizer):
    """Класс метода покоординатного спуска (Наследование)"""

    def optimize(self, problem: OptimizationProblem, x0):
        x = np.array(x0, dtype=float)
        self.history = [x.copy()]

        for _ in range(self.max_iter):
            x_prev = x.copy()

            # Шаг по x1
            res1 = minimize_scalar(lambda alpha: problem.func([x[0] + alpha, x[1]]))
            x[0] += res1.x
            self.history.append(x.copy())

            # Шаг по x2
            res2 = minimize_scalar(lambda alpha: problem.func([x[0], x[1] + alpha]))
            x[1] += res2.x
            self.history.append(x.copy())

            if np.linalg.norm(x - x_prev) < self.eps:
                break
        return x


class GradientDescent(Optimizer):
    """Класс метода градиентного спуска (Наследование)"""

    def __init__(self, learning_rate=0.1, eps=0.0001, max_iter=50):
        super().__init__(eps, max_iter)
        self.learning_rate = learning_rate

    def optimize(self, problem: OptimizationProblem, x0):
        x = np.array(x0, dtype=float)
        self.history = [x.copy()]

        for _ in range(self.max_iter):
            g = problem.grad(x)
            if np.linalg.norm(g) < self.eps:
                break
            x = x - self.learning_rate * g
            self.history.append(x.copy())
        return x


class SteepestDescent(Optimizer):
    """Класс метода наискорейшего спуска (Наследование)"""

    def optimize(self, problem: OptimizationProblem, x0):
        x = np.array(x0, dtype=float)
        self.history = [x.copy()]

        for _ in range(self.max_iter):
            g = problem.grad(x)
            if np.linalg.norm(g) < self.eps:
                break

            res = minimize_scalar(lambda h: problem.func(x - h * g))
            x = x - res.x * g
            self.history.append(x.copy())
        return x


# ==============================================================
# ФУНКЦИИ ВАРИАНТА 1 И МЕТОД НЬЮТОНА (ДЗ 2.2)
# ==============================================================

def f_var1(X):
    return 2 * X[0] ** 2 + 2 * X[0] * X[1] + 3 * X[1] ** 2 - 10 * X[0] - 10 * X[1] + 15


def grad_f_var1(X):
    return np.array([4 * X[0] + 2 * X[1] - 10, 2 * X[0] + 6 * X[1] - 10])


def z_func(X):
    x, y = X
    return x ** 3 + 3 * x ** 2 + y ** 3 - 14 * y ** 2 + 64 * y - 100


def grad_z(X):
    x, y = X
    return np.array([3 * x ** 2 + 6 * x, 3 * y ** 2 - 28 * y + 64])


def hessian_z(X):
    x, y = X
    return np.array([[6 * x + 6, 0], [0, 6 * y - 28]])


def newton_method(x0, eps=0.0001, max_iter=50):
    print("\n--- Метод Ньютона (ДЗ 2.2) ---")
    x = np.array(x0, dtype=float)
    for i in range(1, max_iter + 1):
        g = grad_z(x)
        if np.linalg.norm(g) < eps:
            print(f"Оптимум найден! Итераций: {i}")
            break
        H_inv = np.linalg.inv(hessian_z(x))
        x = x - np.dot(H_inv, g)
        print(f"Итерация {i}: x = {x[0]:.4f}, y = {x[1]:.4f}, |grad| = {np.linalg.norm(g):.6f}")
    return x


# ==============================================================
# ЗАПУСК И ВИЗУАЛИЗАЦИЯ
# ==============================================================
if __name__ == '__main__':
    # Создаем объект задачи
    problem = OptimizationProblem(f_var1, grad_f_var1)
    x_start = [-2.0, -2.0]

    # Создаем объекты алгоритмов
    optimizers = {
        "Покоординатный спуск": CoordinateDescent(),
        "Градиентный спуск (h=0.1)": GradientDescent(learning_rate=0.1),
        "Наискорейший спуск": SteepestDescent()
    }

    # Подготовка к графикам (ДЗ 2.1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    X1, X2 = np.meshgrid(np.linspace(-3, 4, 100), np.linspace(-3, 4, 100))
    Z = f_var1([X1, X2])

    # Запускаем каждый метод и рисуем график
    for ax, (title, optimizer) in zip(axes, optimizers.items()):
        # ООП вызов метода optimize
        optimizer.optimize(problem, x_start)
        hist = np.array(optimizer.history)

        # Рисовка
        ax.contour(X1, X2, Z, levels=30, cmap='viridis')
        ax.plot(hist[:, 0], hist[:, 1], 'r.-', linewidth=2, markersize=8, label='Траектория')
        ax.plot(hist[0, 0], hist[0, 1], 'bo', label='Старт (-2, -2)')
        ax.plot(hist[-1, 0], hist[-1, 1], 'g*', markersize=12, label='Минимум')
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Запуск Ньютона (ДЗ 2.2)
    M0 = [-0.5, 5.5]
    optimum = newton_method(M0)
    print(f"Итоговая точка минимума: x = {optimum[0]:.6f}, y = {optimum[1]:.6f}")