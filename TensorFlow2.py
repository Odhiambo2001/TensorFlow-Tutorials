import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(22)

x = tf.linspace(-2, 2, 201)
x = tf.cast(x, tf.float32)


def f(x):
    y = x ** 2 + 2 * x - 5
    return y


y = f(x) + tf.random.normal(shape=[201])

plt.plot(x.numpy(), y.numpy(), '.', label='Data')
plt.plot(x.numpy(), f(x).numpy(), label='Ground truth')
plt.legend()
#Create a quadratic model with randomly initialized weights and a bias 


class Model(tf.Module):
    def __init__(self):
        #Randomly generate weight and bias terms
        rand_init = tf.random.uniform(shape=[3], minval=0., maxval=5., seed=22)
        #Initialize model parameters
        self.w_q = tf.Variable(rand_init[0])
        self.w_l = tf.Variable(rand_init[1])
        self.b = tf.Variable(rand_init[2])

    @tf.function
    def __call__(self, x):
        #Quadratic Model: quadratic_weight *x^2 +linear_weight * x+bias
        return self.w_q * (x ** 2) + self.w_l * x + self.b
        #First observe your model's performance before training


quad_model = Model()


def plot_preds(x, y, f, model, title):
    plt.figure()
    plt.plot(x, y, '.', label='Data')
    plt.plot(x, f(x), label='Ground truth')
    plt.plot(x, model(x), label='Predictions')
    plt.title(title)
    plt.legend()


plot_preds(x.numpy(), y.numpy(), f, quad_model, 'Before training')
#Now define a loss for your model


def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))
#Write a basic training loop for the model

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=x.shape[0]).batch(batch_size)
#Set training parameters
epochs = 100
learning_rate = 0.01
losses = []
#Format training loop

for epoch in range(epochs):
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            batch_loss = mse_loss(quad_model(x_batch), y_batch)
        #Update parameters with respect to the gradient calculations 
        grads = tape.gradient(batch_loss, quad_model.variables)
        for g, v in zip(grads, quad_model.variables):
            v.assign_sub(learning_rate * g)
    #Keep track of model loss per epoch

    loss = mse_loss(quad_model(x), y)
    losses.append(loss)
    if epoch % 10 == 0:
        print(f'Mean squared error for step {epoch}: {loss.numpy():0.3f}')
#Plot model results
plt.figure()
plt.plot(range(epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.title('MSE loss vs training iterations')
#Now observe your model's performance after training
plot_preds(x.numpy(), y.numpy(), f, quad_model, 'After training')

new_model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.stack([x, x ** 2], axis=1)),
    tf.keras.layers.Dense(units=1, kernel_initializer=tf.random.normal)
])

new_model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)
)

history = new_model.fit(x, y, epochs=100, batch_size=32, verbose=0)
new_model.save('./my_new_model')
#Observe your keras model's performance after training

plt.figure()
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss (Mean Squared Error)')
plt.title('Keras training progress')

plot_preds(x.numpy(), y.numpy(), f, new_model, 'After Training: Keras')
plt.show()
