from audio import *

reversed_data = data[::-1]

scipy.io.wavfile.write('reversed_output1.wav', sampling_rate, reversed_data)


