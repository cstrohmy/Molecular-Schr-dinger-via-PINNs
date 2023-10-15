import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
from hydrogen import radial_wavefunction
from scipy.special import sph_harm


"""
Trains physics-informed neural network to obtain solutions to the time-independent Schrödinger equation.
"""


def generate_eigenstate(n, l, m):
    """
    Generates eigenstate with quantum numbers n, l, m
    :param n: energy shell
    :param l: orbital
    :param m: magnetic quantum number
    :return: eigenstate (a function)
    """
    def eigenstate(r, theta, phi):
        radial_part = radial_wavefunction(n, l, r)
        angular_part = sph_harm(m, l, phi, theta)
        return radial_part * angular_part
    return eigenstate

# nth eigenvalue of the normalized time-independent Schrödinger operator for the hydrogen atom
def eigenvalue(n):
    return -1 / (2 * n ** 2)

# Build model
model = Sequential(layers=[
    Input(shape=(3,)),
    Dense(60, activation='tanh'),
    Dense(60, activation='tanh'),
    Dense(60, activation='tanh'),
    Dense(60, activation='tanh'),
    Dense(60, activation='tanh'),
    Dense(60, activation='tanh'),
    Dense(1)
])

# Specify which eigenstate we want
n, l, m = 2, 1, 1
E = eigenvalue(n)
E = tf.constant(E, dtype=tf.float32)

# Some hyperparameters for training model
epochs = 100
learning_rate = 1e-2
optimizer = Adam(learning_rate=learning_rate)

# Used to sample R^3
N_interior = 1000

# Used to avoid zero solution
N_regularizer = 100
regularization_point = tf.constant([[1., 0., 0.]], dtype=tf.float32)
regularization_point = regularization_point * tf.ones(shape=(N_regularizer, 3), dtype=tf.float32)
regularization_value = tf.constant([[1.]], dtype=tf.float32)
regularization_value = regularization_value * tf.ones(shape=(N_regularizer, 1), dtype=tf.float32)

# Training model using physics prior
for epoch in range(epochs):

    # Sampling parameters uniformly (note: does not yield uniform distribution on R^3)
    interior_data = tf.random.uniform(minval=-3, maxval=3, shape=(N_interior, 3), dtype=tf.float32)
    r = tf.norm(interior_data, axis=1, keepdims=True)

    # Compute loss
    with tf.GradientTape() as tape:

        # Calculate u_xx
        with tf.GradientTape() as tape_xx:
            tape_xx.watch(interior_data)
            with tf.GradientTape() as tape_x:
                tape_x.watch(interior_data)
                u = model(interior_data)
            u_x = tape_x.gradient(u, interior_data)[:, 0:1]
        u_xx = tape_xx.gradient(u_x, interior_data)[:, 1:2]

        # Calculate u_yy
        with tf.GradientTape() as tape_yy:
            tape_yy.watch(interior_data)
            with tf.GradientTape() as tape_y:
                tape_y.watch(interior_data)
                u = model(interior_data)
            u_y = tape_y.gradient(u, interior_data)[:, 1:2]
        u_yy = tape_yy.gradient(u_y, interior_data)[:, 1:2]

        # Calculate u_zz
        with tf.GradientTape() as tape_zz:
            tape_zz.watch(interior_data)
            with tf.GradientTape() as tape_z:
                tape_z.watch(interior_data)
                u = model(interior_data)
            u_z = tape_z.gradient(u, interior_data)[:, 2:3]
        u_zz = tape_zz.gradient(u_z, interior_data)[:, 2:3]

        # Calculate laplacian
        laplacian = u_xx + u_yy + u_zz

        # Calculate loss
        loss = tf.reduce_mean(tf.square(model(regularization_point) - regularization_value)) + \
            tf.reduce_mean(tf.square(r ** 2 * laplacian + r * (1 + E * r)))

    # Apply gradients
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Print
    print(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss}')











