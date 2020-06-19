# My model
network = models.Sequential()

network.add(layers.Conv1D(FEATURES,
                          kernel_size=KERNEL, 
                          activation='relu',
                          input_shape=(time_length,1)))

network.add(layers.GlobalMaxPooling1D())

network.add(layers.Dense(64, activation='softmax'))
network.add(layers.Dropout(0.6))
network.add(layers.Dense(64, activation='softmax'))
network.add(layers.Dense(1, activation='sigmoid'))
