# необходимые параметры для модели

num_lane = 4 # количество полос

time_reaction = 2 # in sec, время реакции водителя, сек
max_forward = 100 # на автомобиль будет воздействовать автомобиль на расстоянии до 100 м перед ним, м
ALPHA = 1 # параметр для обновления ускорения
SAFE_MARGIN = 3 # минимальное расстояние между автомобилями, м

vmax = 30 # максимальная скорость, единица измерения м/сек，~108км/час ~ 67.5 мили/час
vmin = 0
VM = 5 # начальная средняя скорость，м/сек
DV = 5 # стандартная начальная скорость，м/сек
car_size = 5 # длина машины， 5м


PC = 0.5 # возможность смены полосы движения
TC = 1 # выигрыш во времени для смены полосы движения, сек

DT = 0.1 # время шага, сек
