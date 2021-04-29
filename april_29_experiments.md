## Models used so far
FCN8s, DeepLabv3 ResNet50, UNet ResNet34, DeepLabv3+ ResNeXt101 32x16d

## Augmentation (Brightness)
Fixed parameters
- contrast = hue = saturation = 0.0
- model = DeepLabv3+ ResNeXt101 32x16d with instagram pretrain-weights
- batch = 8, lr = 1e-5
- input_size = 256x256
- loss function = cross entropy
- optimizer = adam
- seed = 21

|brightness|epoch|loss|val_loss|val_mIoU|LB|
|---|---|---|---|---|---|
|0.0|7|0.2008|0.3413|0.4963|0.6149|
|0.2|7|0.1981|0.3422|0.4897|N/A|
|0.4|9|0.1757|0.3173|0.5133|0.6158|
|0.6|7|0.2161|0.3362|0.5218|0.6082|
- observations
There are minor LB(less than 0.0010) score difference in between 
different values of brightness. However, it's not worth finding optimal
parameter.<br>
- thought
setting brightness = x, randomly applies brightness transformation to images what if I set up a parameter to have the same effect throughout all images?

## failures
model with loss function, -log(DiceLoss), is hard to train <br>
- thought <br>
consider mixture of cross entropy and dice, with non-equal weights something like 0.7ce - 0.3log(dice)

## Overall thoughts
- so far all the parameter tuning barely affects the performance <br>
- the larger model appears to perform better, thus try efficientnet <br>
- unclear which validation measure to evaluate the performance of the model on unseen data
