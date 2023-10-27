from audio import *

new_sample_rate1 = sampling_rate + 5000
new_sample_rate2 = sampling_rate // 3

scipy.io.wavfile.write('output2_new_sample_rate.wav', new_sample_rate1, data)
scipy.io.wavfile.write('output3_new_sample_rate.wav', new_sample_rate2, data)

