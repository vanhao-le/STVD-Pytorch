import numpy as np
import uuid

"""
generate the random number under conditions:

Length of a random sequence: 30
The random interval: 1
Min and max random value: 15 - 25
Thresold for down / up trend: 0.5
Noise: a random number of 0-5

Y = y_previous +/- noise

"""
def random_walk():
    steps = 30
    positions = [25]
    for i in range(steps):
        rr = np.random.random(steps)
        fps = np.random.randint(0, 5)
        if rr[i] > 0.5:
            new_value = positions[i] + fps
            while new_value > 25:
                new_value = positions[i] + np.random.randint(0, 5)
            positions.append(new_value)
        elif rr[i] < 0.5:
            new_value = positions[i] - fps
            while new_value < 15:
                new_value = positions[i] - np.random.randint(0, 5)
            positions.append(new_value)   
    return np.array(positions)


def hashing_uuid():
    res = uuid.uuid4().hex
    return res