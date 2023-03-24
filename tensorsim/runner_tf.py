import tensorflow as tf
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# tf files for actual ML
import tensorsim.tf_models.UNet_tf
import tensorsim.loss.ULLoss as LDLU 

import tensorsim.data_loader as TSDL

def main(hparams):
    data = TSDL.DataLoader(hparams)

    model = UNet(input_channels=1, classes=1, bn=False)
    model.summary()

    model.compile(
        optimizer='adam', 
        loss=LDLU, 
        metrics=[tf.keras.metrics.MeanAbsoluteError()], 
    )

    history = model.fit(
        data.train_dataset,
        batch_size=hparams.batch_size,
        epochs=hparams.max_epochs,
        validation_data=data.val_dataset,
    )

    test_loss, test_mae = model.evaluate(
        data.test_dataset,
        verbose=2
    )


if __name__ == '__main__':
    main(hparams)