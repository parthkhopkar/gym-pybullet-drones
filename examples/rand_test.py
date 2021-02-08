import numpy as np

def analyze_rand(n_samples):
    times_y_is_greater = 0
    for i in range(n_samples):
        pos = np.random.RandomState().uniform(-2,2,2)
        if abs(pos[1]) > abs(pos[0]):
            times_y_is_greater += 1

    return times_y_is_greater

if __name__ == '__main__':
    samples = 10
    times_y_is_greater = analyze_rand(samples)
    print(times_y_is_greater/samples)

