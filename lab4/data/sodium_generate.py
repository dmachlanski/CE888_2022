import numpy as np

def generate_data(n=1000, seed=0, beta1=1.05, binary_treatment=True, binary_cutoff=3.5):
    np.random.seed(seed)
    age = np.random.normal(65, 5, n)
    sodium = age / 18 + np.random.normal(size=n)
    if binary_treatment:
        if binary_cutoff is None:
            binary_cutoff = sodium.mean()
        sodium = (sodium > binary_cutoff).astype(int)
    blood_pressure = beta1 * sodium + 2 * age + np.random.normal(size=n)

    return age, sodium, blood_pressure

def generate_data_cf(n=1000, seed=0, beta1=1.05, binary_cutoff=3.5):
    np.random.seed(seed)
    age = np.random.normal(65, 5, n)
    sodium = age / 18 + np.random.normal(size=n)

    if binary_cutoff is None:
        binary_cutoff = sodium.mean()

    sodium_f = (sodium > binary_cutoff).astype(int)
    sodium_cf = 1 - sodium_f

    blood_pressure_f = beta1 * sodium_f + 2 * age + np.random.normal(size=n)
    blood_pressure_cf = beta1 * sodium_cf + 2 * age + np.random.normal(size=n)

    return age, sodium, blood_pressure_f, blood_pressure_cf

if __name__ == "__main__":
    age, sodium, bp = generate_data(10000)
    np.savez('sodium_10k', x=age, t=sodium, y=bp)

    age, sodium, bp_f, bp_cf = generate_data_cf(10000)
    np.savez('sodium_10k_cf', x=age, t=sodium, yf=bp_f, ycf=bp_cf)