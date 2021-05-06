## PAN with input size 256x256

|모델|weight|img size|배치|시드|time/epoch|epoch|loss|val_mIoU|LB score|config|비고|
|------|---|---|---|---|---|---|---|---|---|---|---|
|PAN_resnext101_32x8d|swsl|256|8|7|177s|20|IoU+CE|0.5626|0.6105|config20|split|
|PAN_resnext101_32x8d|swsl|256|8|7|217s|19|IoU+CE|0.|0.6143|config22|trainall|

<br>
<br>

## PAN with input size 512x512

|모델|weight|img size|배치|시드|time/epoch|epoch|loss|val_mIoU|LB score|config|비고|
|------|---|---|---|---|---|---|---|---|---|---|---|
|PAN_resnext101_32x8d|swsl|512|8|7|388s|18|IoU+CE|0.6248|0.6267|config24|split|
|PAN_resnext101_32x8d|swsl|512|8|7|477s|17|IoU+CE|0.|0.6396|config25|trainall|

<br>
<br>

## PAN with input size 256x256 using pseudo labeling

|모델|weight|img size|배치|시드|time/epoch|epoch|loss|val_mIoU|LB score|config|비고|
|------|---|---|---|---|---|---|---|---|---|---|---|
|PAN_resnext101_32x8d|swsl|256|8|7|219s|20|IoU+CE|0.5768|0.6615|config26|split|
|PAN_resnext101_32x8d|swsl|256|8|7|257s|19|IoU+CE|0.|0.6625|config27|trainall|

<br>
<br>

## PAN with input size 512x512 using pseudo labeling

|모델|weight|img size|배치|시드|time/epoch|epoch|loss|val_mIoU|LB score|config|비고|
|------|---|---|---|---|---|---|---|---|---|---|---|
|PAN_resnext101_32x8d|swsl|512|8|7|489s|18|IoU+CE|0.6334|0.6786|config28|split|
|PAN_resnext101_32x8d|swsl|512|8|7|574s|17|IoU+CE|0.|0.6793|config29|trainall|

<br>
<br>

## Ensemble

|모델|앙상블 config|weight|LB score|비고|
|-----|-----|---|---|---|
|PAN_resnext101_32x8d|config27, config29|1:1|0.6904||
|PAN_resnext101_32x8d|config27, config29|1:2|0.6924||

<br>
<br>

## K-fold (input size  = 256, K = 5, include pseudo labeling data)
|k|epoch|val_loss|val_mIoU|config|비고|
|---|---|---|---|---|---|
|1|20|0.4224|0.5397|config26||
|2|20|0.3090|0.6186|config26||
|3|19|0.3559|0.5928|config26||
|4|18|0.3249|0.6513|config26||
|5|19|0.3729|0.5951|config26||

위를 앙상블 한 결과 -> __LB score : 0.6796__ 

(기존 config26 LB score : 0.6615 / config26을 train all한 LB score : 0.6625)
<br/>
혹시 몰라 val mIoU가 비교적 낮은 fold1을 빼고 해봤더니 LB score가 0.6800이 나옴

하지만 일반화(private)를 위해서 다 더한 결과를 사용하는 것이 나을 듯

<br>
<br>

## K-fold (input size  = 512, K = 5, include pseudo labeling data)
|k|epoch|val_loss|val_mIoU|config|비고|
|---|---|---|---|---|---|
|1|20|0.3884|0.5830|config28||
|2|16|0.2804|0.6444|config28||
|3|17|0.3275|0.6411|config28||
|4|18|0.2846|0.6884|config28||
|5|19|0.3128|0.6492|config28||

위를 앙상블 한 결과 -> __LB score : 0.6949__ 

(기존 config28 LB score : 0.6786 / config28을 train all한 LB score : 0.6793)

<br>
<br>

## ensemble kfold prediction(above two test, config26,28)

|모델|앙상블 config|weight|LB score|비고|
|-----|-----|---|---|---|
|PAN_resnext101_32x8d kfold|config26, config28|1:2|0.6956||

config28 kfold ensemble과 결과가 같다...