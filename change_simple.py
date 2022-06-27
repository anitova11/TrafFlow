import numpy as np
from numba import njit, cuda
from main import *

# Часть 1. CPU без jit

def update_cell(speed, accel, df):
    # скорость обновления в зависимости от ускорения
    speed += accel * DT
    speed = min(vmax, max(vmin, speed))

    # ускорение обновления
    d_safe = max(speed * time_reaction, car_size + SAFE_MARGIN)
    accel = ALPHA * (df - d_safe)

    return speed, accel


# Определить, есть ли машина перед текущей машиной
def check_front(road, position, num_cell, cell_size):
    i = position[0]
    j = position[1]

    j_max = j + int(max_forward / cell_size) + 1  # максимальное расстояние обнаружения
    df = max_forward

    # проверить наличие автомобиля в каждой ячейке, по одному
    for j_f in range(j + 1, j_max):
        if j_f >= num_cell:
            j_f_temp = j_f - num_cell
        else:
            j_f_temp = j_f

        # если есть маина
        if road[i][j_f_temp]:
            df = (j_f - j) * cell_size  # расстояние до передней машины
            return df

    return df  # текущее расстояние с передним авто и индекс положения переднего авто


# Расчет смещения текущего автомобиля, который может двигаться вперед
def cal_displacement(j, speed, distance2front, num_cell, cell_size):
    displacement = speed * DT
    displacement = max(min(displacement, distance2front - car_size - SAFE_MARGIN), 0)
    dj = int(displacement / cell_size)

    # определить новый индекс ячейки
    if (j + dj) >= num_cell:
        j_new = j + dj - num_cell
    else:
        j_new = j + dj

    return j_new


def forward(road, speed, accel, cell_size):
    num_cell = road.shape[1]
    road_update = np.zeros((num_lane, num_cell), dtype=np.int32)
    road_copy = road.copy()

    for i in range(num_lane):
        for j in range(num_cell):
            if road[i][j] and road_update[i][j] == 0:
                # Расчет расстояния до впереди идущего автомобиля
                df = check_front(road_copy, (i, j), num_cell, cell_size)

                # Расчет нового индекса позиции
                j_new = cal_displacement(j, speed[i][j], df, num_cell, cell_size)

                road[i][j_new] = road[i][j]
                road_update[i][j_new] = 1
                speed[i][j_new], accel[i][j_new] \
                    = update_cell(speed[i][j], accel[i][j], df)

                if j_new != j:
                    road[i][j] = 0
                    speed[i][j] = 0
                    accel[i][j] = 0

    return road, speed, accel


# Часть 2. CPU с jit

@njit
def update_cell_h(speed, accel, df):
    # скорость обновления в зависимости от ускорения
    speed += accel * DT
    speed = min(vmax, max(vmin, speed))

    # ускорение обновления
    d_safe = max(speed * time_reaction, car_size + SAFE_MARGIN)
    accel = ALPHA * (df - d_safe)

    return speed, accel


@njit
# Определить, есть ли машина перед текущей машиной
def check_front_h(road, position, num_cell, cell_size):
    i = position[0]
    j = position[1]

    j_max = j + int(max_forward / cell_size) + 1  # максимальное расстояние обнаружения
    df = max_forward

    # проверить наличие автомобиля в каждой ячейке, по одному
    for j_f in range(j + 1, j_max):
        if j_f >= num_cell:
            j_f_temp = j_f - num_cell
        else:
            j_f_temp = j_f

        # если есть машина
        if road[i][j_f_temp]:
            df = (j_f - j) * cell_size  # расстояние до впереди идущей машины
            return df

    return df  # текущее расстояние с передним вагоном и индекс положения переднего вагона


@njit
# Рассчитайте смещение текущего автомобиля, которое может двигаться вперед
def cal_displacement_h(j, speed, distance2front, num_cell, cell_size):
    displacement = speed * DT
    displacement = max(min(displacement, distance2front - car_size - SAFE_MARGIN), 0)
    dj = int(displacement / cell_size)

    # определить новый индекс ячейки
    if (j + dj) >= num_cell:
        j_new = j + dj - num_cell
    else:
        j_new = j + dj

    return j_new


@njit
def forward_h(road, speed, accel, cell_size):
    num_cell = road.shape[1]
    road_update = np.zeros((num_lane, num_cell), dtype=np.int32)
    road_copy = road.copy()

    for i in range(num_lane):
        for j in range(num_cell):
            if road[i][j] and road_update[i][j] == 0:
                # Рассчитать расстояние до впереди идущего автомобиля
                df = check_front_h(road_copy, (i, j), num_cell, cell_size)

                # Рассчитать новый индекс позиции
                j_new = cal_displacement_h(j, speed[i][j], df, num_cell, cell_size)

                road[i][j_new] = road[i][j]
                road_update[i][j_new] = 1
                speed[i][j_new], accel[i][j_new] \
                    = update_cell_h(speed[i][j], accel[i][j], df)

                if j_new != j:
                    road[i][j] = 0
                    speed[i][j] = 0
                    accel[i][j] = 0

    return road, speed, accel


# Часть 3. CUDA с jit

@cuda.jit(device=True)
# скорость обновления и ускорение
def update_cell_d(speed, accel, df):
    # скорость обновления в зависимости от ускорения
    speed += accel * DT
    speed = min(vmax, max(vmin, speed))

    # ускорение обновления
    d_safe = max(speed * time_reaction, car_size + SAFE_MARGIN)
    accel = ALPHA * (df - d_safe)

    return speed, accel


@cuda.jit(device=True)
# Определить, есть ли машина перед текущей машиной
def check_front_d(road, position, num_cell, cell_size):
    i = position[0]
    j = position[1]

    j_max = j + np.int(max_forward / cell_size) + 1  # максимальное расстояние обнаружения
    df = max_forward

    # проверить наличие автомобиля в каждой ячейке, по одному
    for j_f in range(j + 1, j_max):
        if j_f >= num_cell:
            j_f_temp = j_f - num_cell
        else:
            j_f_temp = j_f

        # если есть машина
        if road[i][j_f_temp]:
            df = (j_f - j) * cell_size  # расстояние до впереди идущего авто
            return df

    return df  # текущее расстояние с передним вагоном и индекс положения переднего вагона


@cuda.jit(device=True)
# Рассчитайте смещение текущего автомобиля, которое может двигаться вперед
def cal_displacement_d(j, speed, distance2front, num_cell, cell_size):
    displacement = speed * DT
    displacement = max(min(displacement, distance2front - car_size - SAFE_MARGIN), 0)
    dj = int(displacement / cell_size)

    # определить новый индекс ячейки
    if (j + dj) >= num_cell:
        j_new = j + dj - num_cell
    else:
        j_new = j + dj

    return j_new
