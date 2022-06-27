from numba import cuda
from activation import initialize, initialize_no_jit
from change_simple import forward, forward_h
from kernel_simple import detect_d, move_d
from main import *


##### CPU no JIT #####
def traffic_cpu_nojit_naive(sim_steps=100, road_length=1, cell_size=1, traffic_density=30):
    # initilize matrices
    road, speed, accel, _, _, _ = initialize_no_jit(road_length, cell_size, traffic_density)
    counter = 0

    while counter < sim_steps:
        road, speed, accel = \
            forward(road, speed, accel, cell_size)
        counter += 1

    return road, speed, accel


##### CPU JIT #####
def traffic_cpu_naive(sim_steps=100, road_length=1, cell_size=1, traffic_density=30):
    # initilize matrices
    road, speed, accel, _, _, _ = initialize(road_length, cell_size, traffic_density)
    counter = 0

    while counter < sim_steps:
        road, speed, accel = \
            forward_h(road, speed, accel, cell_size)
        counter += 1

    return road, speed, accel


##### CUDA #####
def traffic_cuda_naive(sim_steps=1, road_length=1, cell_size=1, traffic_density=30):
    # initilize matrices
    road, speed, accel, _, _, _ = initialize(road_length, cell_size, traffic_density)

    # send matrices to device
    road_d = cuda.to_device(road)
    speed_d = cuda.to_device(speed)
    accel_d = cuda.to_device(accel)
    road_record_d = cuda.to_device(road.copy())
    speed_record_d = cuda.to_device(speed.copy())
    accel_record_d = cuda.to_device(accel.copy())

    # define blocks
    dim_block_x = 128
    dim_block_y = num_lane
    dim_block = (dim_block_x, dim_block_y)

    dim_grid_x = 80
    dim_grid_y = 1
    dim_grid = (dim_grid_x, dim_grid_y)

    # start simulation
    counter = 0
    while counter < sim_steps:
        counter += 1

        detect_d[dim_grid, dim_block](road_d, speed_d, accel_d, road_record_d, speed_record_d, accel_record_d,
                                      cell_size)

        move_d[dim_grid, dim_block](road_d, speed_d, accel_d, road_record_d, speed_record_d, accel_record_d)

    # cudaDeviceSync
    cuda.synchronize()

    # move results to host
    road = road_d.copy_to_host()
    speed = speed_d.copy_to_host()
    accel = accel_d.copy_to_host()

    return road, speed, accel
