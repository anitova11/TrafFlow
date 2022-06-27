import numpy as np
from numba import njit, cuda
from main import *


# Часть 1. CPU

@njit
# Определить, есть ли машина перед текущей машиной
def check_front_h(road, position, num_cell, cell_size):
    i = position[0]
    j = position[1]

    j_max = j + int(max_forward / cell_size) + 1                     # максимальное расстояние обнаружения
    df = max_forward
                                                                        # проверить наличие автомобиля в каждой ячейке, по одному
    for j_f in range(j + 1, j_max):
        if j_f >= num_cell:
            j_f_temp = j_f - num_cell
        else:
            j_f_temp = j_f
                                                                                    # если есть авто
        if road[i][j_f_temp]:
            df = (j_f - j) * cell_size                                   # расстояние до впереди идущего авто
            return df

    return df                                                        # текущее расстояние с передним авто и индекс положения переднего авто


@njit
def check_necessity_h(road, position, speed, num_cell, cell_size):
    i = position[0]
    j = position[1]

    left_necessary = False
    right_necessary = False
    df = check_front_h(road, (i, j), num_cell, cell_size)

    left_eligible = i > 0
    if left_eligible:
        df_l = check_front_h(road, (i - 1, j), num_cell, cell_size)
        left_necessary = (df_l - df) > TC * speed

    right_eligible = i < num_lane - 1
    if right_eligible:
        df_r = check_front_h(road, (i + 1, j), num_cell, cell_size)
        right_necessary = (df_r - df) > TC * speed

    return left_necessary, right_necessary


@njit
def check_safe_h(direct, road, position, speed, num_cell, cell_size):
    assert direct == 1 or direct == -1
    i = position[0]
    j = position[1]

    d_safe_0 = max(speed * time_reaction, car_size + SAFE_MARGIN)
    j_min = j - int(d_safe_0 / cell_size)
    safe_0 = True

    j_b = j
    i_b = i + direct
    while safe_0 and j_b > j_min:
        if road[i_b][j_b]:
            safe_0 = False
        j_b -= 1

    i_s = i + direct + direct
    if i_s >= 0 and i_s < num_lane:
        d_safe_1 = car_size + 2 * SAFE_MARGIN
        j_max = j + int(d_safe_1 / cell_size)
        j_min = j - int(d_safe_1 / cell_size)
        safe_1 = True

        j_s = j_min
        while safe_1 and j_s <= j_max:
            if j_s >= num_cell:
                j_s_temp = num_cell - j_s
            else:
                j_s_temp = j_s
            if road[i_s][j_s_temp]:
                safe_1 = False
            j_s += 1

    return (safe_0 and safe_1)


@njit
# обновить ячейку, используемую для переадресации
def update_cell_h(speed, accel, df):
    # скорость обновления в зависимости от ускорения
    speed += accel * DT
    speed = min(vmax, max(vmin, speed))

    # ускорение обновления
    d_safe = max(speed * time_reaction, car_size + SAFE_MARGIN)
    accel = ALPHA * (df - d_safe)

    return speed, accel


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
# Правило смены полосы
def changeLane_h(road, speed, accel, pos_x, pos_y, rands, cell_size):
    num_cell = road.shape[1]
    n_car = pos_y.shape[0]
    road_copy = road.copy()

    for k in range(n_car):
        i = pos_y[k]
        j = pos_x[k]
        v = speed[i][j]
        a = accel[i][j]

        left_necessary, right_necessary = \
            check_necessity_h(road_copy, (i, j), v, num_cell, cell_size)

        if left_necessary:
            left_safe = check_safe_h(-1, road_copy, (i, j), v, num_cell, cell_size)
            left = left_necessary and left_safe
        else:
            left = False

        if right_necessary:
            right_safe = check_safe_h(1, road_copy, (i, j), v, num_cell, cell_size)
            right = right_necessary and right_safe
        else:
            right = False

        if left and right:  # обе стороны подходят для смены полосы движения
            bl, br = PC / 2, 1 - PC / 2
        elif left and (not right):
            bl, br = PC, 1
        elif (not left) and right:
            bl, br = 0, 1 - PC
        else:
            bl, br = 0, 1

        x = rands[k]
        if x < bl:
            di = -1
        elif x > br:
            di = 1
        else:
            di = 0
        i_new = i + di

        road[i][j] = 0
        speed[i][j] = 0
        accel[i][j] = 0

        road[i_new][j] = k + 1
        speed[i_new][j] = v
        accel[i_new][j] = a
        pos_y[k] = i_new

    return road, speed, accel, pos_y


@njit
def forward_h(road, speed, accel, pos_x, pos_y, cell_size):
    num_cell = road.shape[1]
    n_car = pos_y.shape[0]
    road_copy = road.copy()

    for k in range(n_car):
        i = pos_y[k]
        j = pos_x[k]
        v = speed[i][j]
        a = accel[i][j]

        df = check_front_h(road_copy, (i, j), num_cell, cell_size)
        j_new = cal_displacement_h(j, v, df, num_cell, cell_size)

        road[i][j] = 0
        speed[i][j] = 0
        accel[i][j] = 0

        pos_x[k] = j_new
        road[i][j_new] = k + 1
        speed[i][j_new], accel[i][j_new] = update_cell_h(v, a, df)

    return road, speed, accel, pos_x


# Часть 2. CUDA
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

        # если есть авто
        if road[i][j_f_temp]:
            df = (j_f - j) * cell_size  # расстояние до впереди идущего авто
            return df

    return df  # текущее расстояние с передним вагоном и индекс положения переднего вагона


@cuda.jit(device=True)
def check_necessity_d(road, position, speed, num_cell, cell_size):
    i = position[0]
    j = position[1]

    left_necessary = False
    right_necessary = False
    df = check_front_d(road, (i, j), num_cell, cell_size)

    left_eligible = i > 0
    if left_eligible:
        df_l = check_front_d(road, (i - 1, j), num_cell, cell_size)
        left_necessary = (df_l - df) > TC * speed

    right_eligible = i < num_lane - 1
    if right_eligible:
        df_r = check_front_d(road, (i + 1, j), num_cell, cell_size)
        right_necessary = (df_r - df) > TC * speed

    return left_necessary, right_necessary


@cuda.jit(device=True)
def check_safe_d(direct, road, position, speed, num_cell, cell_size):
    assert direct == 1 or direct == -1
    i = position[0]
    j = position[1]

    # проверьте, безопасно ли для непосредственной соседней полосы
    # (нужно только оглянуться назад, т.к. check_necessity делала проверку вперед)
    d_safe_0 = max(speed * time_reaction, car_size + SAFE_MARGIN)
    j_min = j - int(d_safe_0 / cell_size)
    safe_0 = True

    j_b = j
    i_b = i + direct
    while safe_0 and j_b > j_min:
        if road[i_b][j_b]:
            safe_0 = False
        j_b -= 1

    # проверьте, безопасна ли полоса рядом с соседней полосой, если применимо
    # (необходимо смотреть как сзади, так и спереди, чтобы две машины
    # не перестроились на одну полосу и не приблизились друг к другу)
    i_s = i + direct + direct
    if i_s >= 0 and i_s < num_lane:
        d_safe_1 = car_size + 2 * SAFE_MARGIN
        j_max = j + int(d_safe_1 / cell_size)
        j_min = j - int(d_safe_1 / cell_size)
        safe_1 = True

        j_s = j_min
        while safe_1 and j_s <= j_max:
            if j_s >= num_cell:
                j_s_temp = num_cell - j_s
            else:
                j_s_temp = j_s
            if road[i_s][j_s_temp]:
                safe_1 = False
            j_s += 1

    return (safe_0 and safe_1)


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
