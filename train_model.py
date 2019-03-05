from model.model_arch import *
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def main():
    model = build_model((384, 512, 3))
    # init Optimizer for the model
    EPOCHS = 50
    INIT_LR = 1e-3
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    # Compile the model with Adam optimizer and two loss function
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['acc'])

    print(model.summary())


if __name__ == '__main__':
    main()
