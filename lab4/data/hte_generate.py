import numpy as np
import pandas as pd

def append_repeat(arr, item, n):
    for _ in range(n):
        arr.append(item)

def get_dataset(counts=(1, 1, 1, 1)):
    data = []

    # [X, T, Y, ITE]
    append_repeat(data, [0, 1, 2.5, 2.3], counts[0])
    append_repeat(data, [1, 1, 0.3, -3.7], counts[1])
    append_repeat(data, [1, 0, 4.0, -3.7], counts[2])
    append_repeat(data, [0, 0, 0.2, 2.3], counts[3])

    data_arr = np.array(data)

    # Add some noise to X and Y.
    data_arr[:, 0] += np.random.normal(size = data_arr.shape[0], loc = 0, scale = 0.1)
    data_arr[:, 2] += np.random.normal(size = data_arr.shape[0], loc = 0, scale = 0.1)

    return data_arr

if __name__ == "__main__":
    train = get_dataset((4, 11, 2, 12))
    test = get_dataset((5, 5, 5, 5))

    col_names = ['x', 't', 'y', 'ite']
    pd.DataFrame(train, columns=col_names).to_csv('hte_train.csv', index=False)
    pd.DataFrame(test, columns=col_names).to_csv('hte_test.csv', index=False)