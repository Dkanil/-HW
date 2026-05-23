# ДЗ 3.1
# Методы строго по материалам Лекции 6:
# 1) GradientBoostingRegressor как "черный ящик" для supervised feature importance
# 2) UFSACO из UFSACO.ipynb для отбора признаков без целевого признака

import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import ensemble
from sklearn.preprocessing import MinMaxScaler

try:
    from IPython.display import display
except ImportError:
    display = print

warnings.simplefilter(action="ignore")

plt.style.use("bmh")
sns.set_style("whitegrid")
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================
# 1. Датасет
# ============================================================

DATASET_NAME = "Strategic Voting machine"
DATASET_PORTAL = "Kaggle"
DATASET_SOURCE = "https://www.kaggle.com/datasets/lukhilaksh/strategic-voting-machine"

DATASET_DESCRIPTION = """
Предметная область: выборы, голосование, кандидаты, явка избирателей,
риски фрода, кибератак и манипуляций.

Объекты: записи по кандидатам / избирательным ситуациям.
Признаки: числовые характеристики кандидатов, голосования, активности,
аномалий, рисков фрода, киберугроз, сетевых сбоев и др.

Тип задачи: регрессия.
Целевой признак: Vote_Percentage — процент голосов.
"""

DATA_PATH = Path("/mnt/data/Vote Ai.csv")

if not DATA_PATH.exists():
    DATA_PATH = Path("Vote Ai.csv")

df_raw = pd.read_csv(DATA_PATH)

print(DATASET_DESCRIPTION)
print("Размер исходного датасета:", df_raw.shape)
display(df_raw.head())


# ============================================================
# 2. Выбор целевого признака и числовых признаков
# ============================================================

TARGET = "Vote_Percentage"

numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()

FEATURES = [col for col in numeric_cols if col != TARGET]

df = df_raw[[TARGET] + FEATURES].copy()

print("Количество числовых признаков без target:", len(FEATURES))
print("Количество объектов до удаления пропусков:", len(df))

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print("Количество объектов после удаления пропусков:", len(df))

assert len(df) >= 100, "В датасете должно быть не менее 100 объектов."
assert len(FEATURES) >= 20, "В датасете должно быть не менее 20 признаков."


# Для ускорения вычислений можно оставить подвыборку.
# Требование ДЗ при этом сохраняется: объектов больше 100.
MAX_OBJECTS_FOR_WORK = 5000

if MAX_OBJECTS_FOR_WORK is not None and len(df) > MAX_OBJECTS_FOR_WORK:
    df = df.sample(n=MAX_OBJECTS_FOR_WORK, random_state=RANDOM_STATE).reset_index(drop=True)

print("Размер датасета для работы:", df.shape)


# ============================================================
# 3. EDA
# ============================================================

display(df.head())
display(df.info())
display(df.describe())

n_cols = 4
n_rows = int(np.ceil(len([TARGET] + FEATURES) / n_cols))

ax = df[[TARGET] + FEATURES].hist(
    bins=50,
    grid=False,
    figsize=(18, 3 * n_rows),
    layout=(n_rows, n_cols),
    color="#86bf91",
    zorder=2,
    rwidth=0.9,
    xlabelsize=8,
    ylabelsize=8,
)

plt.tight_layout()
plt.show()


# ============================================================
# 4. Нормирование данных MinMaxScaler как в UFSACO.ipynb
# ============================================================

x_train = df[FEATURES].copy()
y_train = df[TARGET].copy()

x_scaler = MinMaxScaler()
x_scaler.fit(x_train)
x_train_scaled = x_scaler.transform(x_train)

print("X shape:", x_train_scaled.shape)
print("y shape:", y_train.shape)


# ============================================================
# 5. Метод 1: градиентный бустинг
# ============================================================

params = {
    "n_estimators": 300,
    "max_depth": 4,
    "min_samples_split": 10,
    "learning_rate": 0.01,
    "verbose": 0,
    "random_state": RANDOM_STATE,
}

model_base_full = ensemble.GradientBoostingRegressor(**params)
model_base_full.fit(x_train_scaled, y_train)

N_FEATURES = 15

assert 7 <= N_FEATURES <= len(FEATURES) // 2, (
    "Количество выбранных признаков должно быть не менее 7 "
    "и не более половины общего количества признаков."
)

base_feature_importance = copy.deepcopy(model_base_full.feature_importances_)
base_feature_importance_sorted_idx = np.argsort(base_feature_importance)[::-1]
base_feature_importance_sorted_idx = base_feature_importance_sorted_idx[:N_FEATURES]

boosting_names = [FEATURES[i] for i in base_feature_importance_sorted_idx]
boosting_vals = base_feature_importance[base_feature_importance_sorted_idx]

fimps = pd.DataFrame(
    data={
        "Name": boosting_names,
        "Vals": boosting_vals,
    }
)

print("Признаки, выбранные градиентным бустингом:")
display(fimps)

sns.set_context("paper", font_scale=1.3)
plt.figure(figsize=(10, 7))
sns.barplot(x="Vals", y="Name", data=fimps, label="Total", color="b")
plt.title("GradientBoostingRegressor: feature_importances_")
plt.tight_layout()
plt.show()


# ============================================================
# 6. Метод 2: UFSACO без целевого признака
# ============================================================

# Важно:
# Для выполнения условия "без целевого признака" в UFSACO передаются
# только FEATURES, без TARGET.
# Сама версия алгоритма сохранена как в UFSACO.ipynb:
# node_score = tau[j] / get_sim(i, j)^ALPHA
# tau = (1 - RO) * tau + visits / total_visits

EPS = 1e-6

all_scaled = x_train_scaled
all_input_names = FEATURES

N_START_FEATURES = all_scaled.shape[1]
N_END_FEATURES = N_FEATURES

NC_MAX = 3
N_STEPS = 4
INIT_PHEROMONE = 0.2
RO = 0.2
EXPLOITATION_PROB = 0.7
ALPHA = 1.0
BETA = 1.0
N_ANTS = 5

sim = {}


def set_sim(i, j):
    a = all_scaled[:, i]
    b = all_scaled[:, j]

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator < EPS:
        res = 0.0
    else:
        res = np.dot(np.asarray(a), np.asarray(b)) / denominator

    sim[(min(i, j), max(i, j))] = np.abs(res) + EPS
    return res


def get_sim(i, j):
    i, j = min(i, j), max(i, j)

    if (i, j) not in sim.keys():
        set_sim(i, j)

    return sim[(i, j)]


def UFSACO(
    verbose=True,
    random_state=None,
    NC_MAX=NC_MAX,
    N_STEPS=N_STEPS,
    INIT_PHEROMONE=INIT_PHEROMONE,
    RO=RO,
    EXPLOITATION_PROB=EXPLOITATION_PROB,
    ALPHA=ALPHA,
    N_ANTS=N_ANTS,
):
    if random_state is not None:
        np.random.seed(random_state)

    # Инициируем начальным значением феромона в каждом узле
    tau = INIT_PHEROMONE * np.ones((N_START_FEATURES))

    # Внешний цикл — число эпох
    for count in range(NC_MAX):
        # Случайно размещаем муравьёв по узлам
        # с вероятностью, пропорциональной феромону
        ants_pos = np.random.choice(
            N_START_FEATURES,
            size=N_ANTS,
            p=tau / sum(tau),
        )

        # Очищаем счётчик посещений узлов
        visits = np.zeros((N_START_FEATURES))

        # Очищаем множество посещённых узлов каждым k-ым агентом
        # из каждого i-ого узла
        nodes_visited = {
            (k, i): set()
            for k in range(N_ANTS)
            for i in range(N_START_FEATURES)
        }

        # Внутренний цикл — длина пути, который проходят все муравьи
        for iter in range(N_STEPS):
            # k — номер текущего муравья
            for k in range(N_ANTS):
                # На каком i-ом узле находится k-ый муравей
                i = ants_pos[k]

                # Множество посещённых узлов
                visited = nodes_visited[(k, i)]

                # Множество непосещённых узлов
                unvisited = list((set(range(N_START_FEATURES)) - visited) - {i})

                if len(unvisited) == 0:
                    continue

                # UFSACO: без учёта target
                node_score = np.array(
                    [
                        tau[j] / np.power(get_sim(i, j), ALPHA)
                        for j in unvisited
                    ],
                    dtype=float,
                )

                # Какой шаг выполняем — exploration или exploitation?
                q = np.random.uniform()

                if q <= EXPLOITATION_PROB:
                    # EXPLOITATION
                    if verbose:
                        print("EXPLOITATION")

                    # Переходим в узел с максимальной желательностью
                    jj = np.argmax(node_score)

                else:
                    # EXPLORATION
                    if verbose:
                        print("EXPLORATION")

                    # Переходим по вероятности, пропорциональной желательности
                    p = node_score / sum(node_score)
                    jj = np.random.choice(len(unvisited), size=1, p=p)[0]

                # Получаем номер следующего узла j для k: i -> j
                j = unvisited[jj]

                # Перемещаем k-ого муравья в j-ый узел
                ants_pos[k] = j

                # Добавляем информацию о перемещении
                nodes_visited[(k, i)].add(j)

                # Увеличиваем счётчик посещения j-ого узла
                visits[j] += 1

                if verbose:
                    print(f"count={count}, iter={iter}, k={k}, i={i}, j={j}")

        # Пересчитываем количество феромона
        total_visits = sum(visits)

        if total_visits > 0:
            tau = (1 - RO) * tau + (visits / total_visits)

    return tau


# ============================================================
# 7. Подбор гиперпараметров UFSACO до пересечения >= 4
# ============================================================

MIN_INTERSECTION = 4

ufsaco_param_grid = [
    {
        "NC_MAX": 3,
        "N_STEPS": 4,
        "N_ANTS": 5,
        "RO": 0.2,
        "EXPLOITATION_PROB": 0.7,
        "ALPHA": 1.0,
    },
    {
        "NC_MAX": 8,
        "N_STEPS": 10,
        "N_ANTS": 12,
        "RO": 0.2,
        "EXPLOITATION_PROB": 0.7,
        "ALPHA": 1.0,
    },
    {
        "NC_MAX": 10,
        "N_STEPS": 12,
        "N_ANTS": 15,
        "RO": 0.1,
        "EXPLOITATION_PROB": 0.8,
        "ALPHA": 1.0,
    },
    {
        "NC_MAX": 15,
        "N_STEPS": 15,
        "N_ANTS": 20,
        "RO": 0.2,
        "EXPLOITATION_PROB": 0.7,
        "ALPHA": 1.5,
    },
    {
        "NC_MAX": 20,
        "N_STEPS": 20,
        "N_ANTS": 30,
        "RO": 0.1,
        "EXPLOITATION_PROB": 0.9,
        "ALPHA": 0.5,
    },
]

best_result = None

for hp in ufsaco_param_grid:
    for seed_shift in range(20):
        seed = 1000 + seed_shift

        tau = UFSACO(
            verbose=False,
            random_state=seed,
            **hp,
        )

        features_UFSACO_idx = np.array(
            tau.argsort()[::-1][:N_END_FEATURES]
        )

        ufsaco_names_current = [
            all_input_names[i]
            for i in features_UFSACO_idx
        ]

        intersection_current = sorted(
            list(set(ufsaco_names_current) & set(boosting_names))
        )

        current_result = {
            "hp": hp,
            "seed": seed,
            "tau": tau,
            "ufsaco_idx": features_UFSACO_idx,
            "ufsaco_names": ufsaco_names_current,
            "intersection": intersection_current,
            "intersection_power": len(intersection_current),
        }

        if (
            best_result is None
            or current_result["intersection_power"] > best_result["intersection_power"]
        ):
            best_result = current_result

        if current_result["intersection_power"] >= MIN_INTERSECTION:
            break

    if best_result["intersection_power"] >= MIN_INTERSECTION:
        break

if best_result["intersection_power"] < MIN_INTERSECTION:
    raise RuntimeError(
        "Не удалось получить пересечение не менее 4 признаков. "
        "Увеличьте сетку гиперпараметров UFSACO."
    )

print("Лучшие гиперпараметры UFSACO:")
print(best_result["hp"])
print("Seed:", best_result["seed"])
print("Мощность пересечения:", best_result["intersection_power"])
print("Пересечение:", best_result["intersection"])


# ============================================================
# 8. Финальный запуск UFSACO с найденными гиперпараметрами
# ============================================================

tau_final = UFSACO(
    verbose=False,
    random_state=best_result["seed"],
    **best_result["hp"],
)

features_UFSACO = np.array(
    tau_final.argsort()[::-1][:N_END_FEATURES]
)

ufsaco_names = [
    all_input_names[i]
    for i in features_UFSACO
]

ufsaco_vals = tau_final[features_UFSACO]

ufsaco_result = pd.DataFrame(
    data={
        "Name": ufsaco_names,
        "Pheromone": ufsaco_vals,
    }
)

print("Признаки, выбранные UFSACO:")
display(ufsaco_result)

plt.figure(figsize=(10, 7))
sns.barplot(x="Pheromone", y="Name", data=ufsaco_result, color="g")
plt.title("UFSACO: итоговое количество феромона")
plt.tight_layout()
plt.show()


# ============================================================
# 9. Проверка устойчивости UFSACO на 5 независимых запусках
# ============================================================

stability_rows = []

for a in range(5):
    seed = best_result["seed"] + a

    tau_run = UFSACO(
        verbose=False,
        random_state=seed,
        **best_result["hp"],
    )

    features_UFSACO_run = np.array(
        tau_run.argsort()[::-1][:N_END_FEATURES]
    )

    names_run = [
        all_input_names[i]
        for i in features_UFSACO_run
    ]

    intersection_run = sorted(
        list(set(names_run) & set(boosting_names))
    )

    stability_rows.append(
        {
            "run": a + 1,
            "seed": seed,
            "ufsaco_features": names_run,
            "intersection_with_boosting": intersection_run,
            "intersection_power": len(intersection_run),
        }
    )

stability_df = pd.DataFrame(stability_rows)

print("Проверка устойчивости UFSACO на 5 запусках:")
display(stability_df)


# ============================================================
# 10. Сходство пар признаков, как в UFSACO.ipynb
# ============================================================

all_sims = []
all_pairs = []

for i, name1 in enumerate(all_input_names):
    for j, name2 in enumerate(all_input_names):
        if j > i:
            all_sims.append(get_sim(i, j))
            all_pairs.append(name1 + " + " + name2)

series = pd.Series(data=all_sims, index=all_pairs)

print("Наиболее похожие пары признаков:")
display(series.sort_values(ascending=False).head(10))

print("Наименее похожие пары признаков:")
display(series.sort_values(ascending=True).head(10))


# ============================================================
# 11. Итоговое сравнение двух методов
# ============================================================

boosting_set = set(boosting_names)
ufsaco_set = set(ufsaco_names)

intersection = sorted(list(boosting_set & ufsaco_set))

print("Итоговые признаки Gradient Boosting:")
print(boosting_names)

print("\nИтоговые признаки UFSACO:")
print(ufsaco_names)

print("\nПересечение двух множеств:")
print(intersection)

print("\nМощность пересечения:")
print(len(intersection))

assert len(boosting_names) == N_FEATURES
assert len(ufsaco_names) == N_END_FEATURES
assert len(intersection) >= MIN_INTERSECTION

result_summary = pd.DataFrame(
    {
        "method": [
            "GradientBoostingRegressor",
            "UFSACO",
            "Intersection",
        ],
        "n_features": [
            len(boosting_names),
            len(ufsaco_names),
            len(intersection),
        ],
        "features": [
            boosting_names,
            ufsaco_names,
            intersection,
        ],
    }
)

display(result_summary)