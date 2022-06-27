import sys
#import time
import numpy as np
from modeling import traffic_cpu_lc, traffic_cuda_lc


def run(sim_steps=1, road_length=1, cell_size=1, traffic_density=30):
    sys.stdout.write('\n')
    sys.stdout.write('Введенные данные \n')
    sys.stdout.write('Всего шагов моделирования: {}\n'.format(sim_steps))
    sys.stdout.write('Длина дороги: {} км\n'.format(road_length))
    sys.stdout.write('Размер ячейки: {} м\n'.format(cell_size))
    sys.stdout.write('Плотность трафика: {} авто/км \n'.format(traffic_density))

    road, speed, accel = traffic_cuda_lc(sim_steps, road_length, cell_size, traffic_density, verbose=True)
    road0, speed0, accel0 = traffic_cpu_lc(sim_steps, road_length, cell_size, traffic_density, verbose=True)

    ncar = np.sum(np.array(road, dtype=np.bool))
    vm = np.sum(speed) / ncar * 3.6

    ncar0 = np.sum(np.array(road0, dtype=np.bool))
    vm0 = np.sum(speed0) / ncar0 * 3.6

    sys.stdout.write('Проверка \n')
    sys.stdout.write('Вывод с GPU. Число автомобилей: {}\n'.format(ncar))
    sys.stdout.write('Вывод с CPU. Число автомобилей: {}\n'.format(ncar0))
    sys.stdout.write('Вывод с GPU. Средняя скорость: {:.1f} км/ч\n'.format(vm))
    sys.stdout.write('Вывод с CPU. Средняя скорость: {:.1f} км/ч\n'.format(vm0))
    sys.stdout.write('Вывод с GPU. Транспортный поток: {:.0f} автом/час \n'.format(traffic_density * vm))
    sys.stdout.write('Вывод с CPU. Транспортный поток: {:.0f} автом/час \n'.format(traffic_density * vm0))
    sys.stdout.write('\n')
