import numpy as np
from numba import njit
from main import *


# Построение матриц дороги, скорости и ускорения при t = 0
@njit
def initialize(road_length, cell_size, traffic_density):
    num_cell = int(road_length * 1000 / cell_size)               # количество продольных ячеек
    traffic_density = min(traffic_density, 90)                   # плотность движения не должна превышать разумный порог
    n_car_per_lane = road_length * traffic_density               # количество автомобилей на полосе
    spacing = np.int(num_cell / n_car_per_lane)                  # пространство между каждым автомобилем

    road = np.zeros((num_lane, num_cell), dtype=np.int32)
    speed = np.zeros((num_lane, num_cell), dtype=np.float32)
    accel = np.zeros((num_lane, num_cell), dtype=np.float32)

    n_car = int(num_lane * np.arange(0, num_cell - spacing, spacing).shape[0])
    pos_x = np.zeros(n_car, dtype=np.int32)
    pos_y = np.zeros(n_car, dtype=np.int32)
    rands = np.zeros(n_car, dtype=np.float32)

    if spacing <= car_size + SAFE_MARGIN:
        raise ValueError('Настройка плотности слишком высока. Пожалуйста, '
                         'уменьшите плотность.')

    np.random.seed(0)
    counter = 0
    while counter < n_car:
        i = np.random.randint(num_lane)
        j = np.random.randint(num_cell)
        position_valid = check_spacing(road, (i, j), num_cell, cell_size)
        if position_valid:
            road[i][j] = counter + 1
            speed[i][j] = max(vmin, min(vmax, np.random.normal(VM, DV)))
            accel[i][j] = 0
            pos_x[counter] = j
            pos_y[counter] = i
            rands[counter] = np.random.uniform(0, 1)
            counter += 1

    return road, speed, accel, pos_x, pos_y, rands


@njit
def check_spacing(road, position, num_cell, cell_size):
    i = position[0]
    j = position[1]
    position_valid = True

    j_max = j + int((car_size + SAFE_MARGIN) / cell_size) + 1
    j_min = j - int((car_size + SAFE_MARGIN) / cell_size) - 1
    for j in range(j_min, j_max + 1):
        if j >= num_cell:
            j_temp = j - num_cell
        else:
            j_temp = j
        if road[i][j_temp]:
            position_valid = False
            return position_valid
    return position_valid


def initialize_no_jit(road_length, cell_size, traffic_density):
    num_cell = int(road_length * 1000 / cell_size)  # количество продольных ячеек
    traffic_density = min(traffic_density, 90)  # плотность движения не должна превышать разумный порог
    n_car_per_lane = road_length * traffic_density  # количество автомобилей на полосе
    spacing = np.int(num_cell / n_car_per_lane)  # пространство между каждым автомобилем

    road = np.zeros((num_lane, num_cell), dtype=np.int32)
    speed = np.zeros((num_lane, num_cell), dtype=np.float32)
    accel = np.zeros((num_lane, num_cell), dtype=np.float32)

    n_car = int(num_lane * np.arange(0, num_cell - spacing, spacing).shape[0])
    pos_x = np.zeros(n_car, dtype=np.int32)
    pos_y = np.zeros(n_car, dtype=np.int32)
    rands = np.zeros(n_car, dtype=np.float32)

    if spacing <= car_size + SAFE_MARGIN:
        raise ValueError('Настройка плотности слишком высока. Пожалуйста, уменьшите плотность.')

    np.random.seed(0)
    counter = 0
    while counter < n_car:
        i = np.random.randint(num_lane)
        j = np.random.randint(num_cell)
        position_valid = check_spacing_no_jit(road, (i, j), num_cell, cell_size)
        if position_valid:
            road[i][j] = counter + 1
            speed[i][j] = max(vmin, min(vmax, np.random.normal(VM, DV)))
            accel[i][j] = 0
            pos_x[counter] = j
            pos_y[counter] = i
            rands[counter] = np.random.uniform(0, 1)
            counter += 1

    return road, speed, accel, pos_x, pos_y, rands


def check_spacing_no_jit(road, position, num_cell, cell_size):
    i = position[0]
    j = position[1]
    position_valid = True

    j_max = j + int((car_size + SAFE_MARGIN) / cell_size) + 1
    j_min = j - int((car_size + SAFE_MARGIN) / cell_size) - 1
    for j in range(j_min, j_max + 1):
        if j >= num_cell:
            j_temp = j - num_cell
        else:
            j_temp = j
        if road[i][j_temp]:
            position_valid = False
            return position_valid
    return position_valid
