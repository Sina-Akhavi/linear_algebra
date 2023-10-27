import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
from audio import data_new

sampling_rate, data = scipy.io.wavfile.read('voice1.wav')

samples = np.arange(data.shape[0])

channel1 = data[:, 0]
channel2 = data[:, 1]

# Plot the two channels together in the same axes
# plt.plot(samples, channel1)
# plt.plot(samples, channel2)
# plt.title('channel 1 and 2 in one figure')

# both channels using a single plt.plot
# plt.plot(samples, data)

# Plot the channels of the output data (data_new)
# plt.plot(samples, data_new)

# in separate figures but one window
figure, axis = plt.subplots(1, 2)
axis[0].plot(samples, channel1)
axis[0].set_title("channel 1")
axis[1].plot(samples, channel2)
axis[1].set_title("channel 2")

plt.show()

