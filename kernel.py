from numba import cuda
from change import check_front_d, check_necessity_d, check_safe_d, \
    update_cell_d, cal_displacement_d
from main import *


@cuda.jit
# Правило движения автомобилей вперед
def detect_lane_change_d(road, speed, accel, pos_x, pos_y, pos_y_cp, rands, cell_size):
    k0 = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x

    num_cell = road.shape[1]
    n_car = pos_x.shape[0]

    for k in range(k0, n_car, stride):
        i = pos_y[k]
        j = pos_x[k]
        v = speed[i][j]
        a = accel[i][j]

        left_necessary, right_necessary = check_necessity_d(road, (i, j), v, num_cell, cell_size)

        if left_necessary:
            left_safe = check_safe_d(-1, road, (i, j), v, num_cell, cell_size)
            left = left_necessary and left_safe
        else:
            left = False

        if right_necessary:
            right_safe = check_safe_d(1, road, (i, j), v, num_cell, cell_size)
            right = right_necessary and right_safe
        else:
            right = False

        if left and right:  # И левый, и правый хороши для смены полосы движения
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
        pos_y_cp[k] = i_new
        speed[i][j] = 0
        accel[i][j] = 0
        speed[i_new][j] = v
        accel[i_new][j] = a


@cuda.jit
# Правило движения автомобилей вперед
def perform_lane_change_d(road, pos_x, pos_y, pos_y_cp):
    k0 = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x

    NC = road.shape[1]
    n_car = pos_x.shape[0]

    for k in range(k0, n_car, stride):
        i = pos_y[k]
        j = pos_x[k]
        i_new = pos_y_cp[k]

        road[i][j] = 0
        road[i_new][j] = k + 1
        pos_y[k] = i_new


@cuda.jit
# Правило движения автомобилей вперед
def detect_d(road, speed, accel, pos_x, pos_y, pos_x_cp, cell_size):
    k0 = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x

    NC = road.shape[1]
    n_car = pos_x.shape[0]

    for k in range(k0, n_car, stride):
        i = pos_y[k]
        j = pos_x[k]
        v = speed[i][j]
        a = accel[i][j]

        df = check_front_d(road, (i, j), NC, cell_size)
        j_new = cal_displacement_d(j, v, df, NC, cell_size)
        pos_x_cp[k] = j_new

        # скорость и ускорение и быть написанным прямо
        speed[i][j] = 0
        accel[i][j] = 0
        speed[i][j_new], accel[i][j_new] = update_cell_d(v, a, df)


@cuda.jit
# Правило движения вперед
def move_d(road, pos_x, pos_y, pos_x_cp):
    k0 = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x

    NC = road.shape[1]
    n_car = pos_x.shape[0]

    for k in range(k0, n_car, stride):
        i = pos_y[k]
        j = pos_x[k]
        j_new = pos_x_cp[k]

        road[i][j] = 0
        road[i][j_new] = k + 1
        pos_x[k] = j_new
