from numba import cuda
from change_simple import check_front_d, cal_displacement_d, update_cell_d
from main import *

@cuda.jit
# Правило движения автомобилей вперед
def detect_d(road, speed, accel, road_record, speed_record, accel_record, cell_size):
    i = cuda.threadIdx.y
    j0 = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x

    num_cell = road.shape[1]

    if j0 < num_cell and i < num_lane:
        for j in range(j0, num_cell, stride):
            if road[i][j]:
                # Рассчитать расстояние до впереди идущего автомобиля
                df = check_front_d(road, (i, j), num_cell, cell_size)
                j_new = cal_displacement_d(j, speed[i][j], df, num_cell, cell_size)
                speed_new, accel_new \
                    = update_cell_d(speed[i][j], accel[i][j], df)

                road_record[i][j] = j_new
                speed_record[i][j] = speed_new
                accel_record[i][j] = accel_new
            else:
                road_record[i][j] = -1
                speed_record[i][j] = -1
                accel_record[i][j] = -1


@cuda.jit
# Правило движения автомобилей вперед
def move_d(road, speed, accel, road_record, speed_record, accel_record):
    i = cuda.threadIdx.y
    j0 = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x

    num_cell = road.shape[1]

    if j0 < num_cell and i < num_lane:
        for j in range(j0, num_cell, stride):
            j_new = road_record[i][j]
            if j_new != -1:
                temp = road[i][j]
                road[i][j] = 0
                speed[i][j] = 0
                accel[i][j] = 0

                road[i][j_new] = temp
                speed[i][j_new] = speed_record[i][j]
                accel[i][j_new] = accel_record[i][j]
