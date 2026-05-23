import numpy as np
import matplotlib.pyplot as plt


# Функция и ее производная
def f(x):
    return x ** 2 - 3 * x + x * np.log(x)


def df(x):
    return 2 * x - 2 + np.log(x)


# --- МЕТОД КВАДРАТИЧНОЙ АППРОКСИМАЦИИ (ЛР 3) ---
def quadratic_approximation(x1, dx, eps=0.0001, max_iter=10):
    print("=== Квадратичная аппроксимация ===")
    history = []

    for i in range(max_iter):
        x2 = x1 + dx
        f1, f2 = f(x1), f(x2)

        # Шаг 4 по алгоритму из лекции
        if f1 > f2:
            x3 = x1 + 2 * dx
        else:
            x3 = x1 - dx

        f3 = f(x3)

        # Шаг 7: Формула вершины параболы
        num = (x2 ** 2 - x3 ** 2) * f1 + (x3 ** 2 - x1 ** 2) * f2 + (x1 ** 2 - x2 ** 2) * f3
        den = (x2 - x3) * f1 + (x3 - x1) * f2 + (x1 - x2) * f3

        if den == 0: break
        x_bar = 0.5 * (num / den)
        f_bar = f(x_bar)

        print(f"Итерация {i + 1}: Точки ({x1:.4f}, {x2:.4f}, {x3:.4f}), x_min = {x_bar:.6f}, f(x_min) = {f_bar:.6f}")
        history.append(x_bar)

        # Шаг 8: Критерий останова
        f_min = min(f1, f2, f3)
        x_min = [x1, x2, x3][np.argmin([f1, f2, f3])]

        if abs((f_min - f_bar) / f_bar) < eps and abs((x_min - x_bar) / x_bar) < eps:
            break

        x1 = x_bar
        dx = dx / 2  # сужаем шаг

    return history


# --- МЕТОД КУБИЧЕСКОЙ АППРОКСИМАЦИИ (ДЗ 1.1) ---
def cubic_approximation(x1, x2, eps=0.0001, max_iter=10):
    print("\n=== Кубическая аппроксимация ===")
    history = []

    for i in range(max_iter):
        y1, y2 = f(x1), f(x2)
        y11, y12 = df(x1), df(x2)

        # Формулы строго из PDF (слайд 1)
        z = y11 + y12 - 3 * (y2 - y1) / (x2 - x1)
        w = np.sqrt(z ** 2 - y11 * y12)

        mu = (w + z - y11) / (2 * w - y11 + y12)
        x_new = x1 + mu * (x2 - x1)

        print(f"Итерация {i + 1}: Отрезок [{x1:.4f}, {x2:.4f}], x_new = {x_new:.6f}, f(x_new) = {f(x_new):.6f}")
        history.append(x_new)

        # Проверка остановки
        if abs(df(x_new)) < eps:
            break

        # Сужение отрезка
        if df(x_new) < 0:
            x1 = x_new
        else:
            x2 = x_new

    return history


# Запуск и графики (ДЗ 1.2)
hist_quad = quadratic_approximation(x1=1.0, dx=0.5)
hist_cubic = cubic_approximation(x1=1.0, x2=2.0)

x_plt = np.linspace(0.5, 2.5, 100)
y_plt = f(x_plt)

plt.figure(figsize=(8, 5))
plt.plot(x_plt, y_plt, 'k-', label="f(x) = x^2 - 3x + x*ln(x)")
plt.scatter(hist_quad, f(np.array(hist_quad)), color='red', label='Квадратичная')
plt.scatter(hist_cubic, f(np.array(hist_cubic)), color='blue', marker='x', s=80, label='Кубическая')
plt.title("Визуализация аппроксимаций")
plt.legend()
plt.grid()
plt.show()