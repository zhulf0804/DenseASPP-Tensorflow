## About
A tensorflow implementation based on DenseASPP trained on CamVid dataset. The mIoU can be 0.58 evaluated on test set.

The CamVid dataset can be as follows:

```
|--CamVid
     |--train       (367)
          |--*.png
     |--trainannot  (367)
          |--*.png
     |--val         (101)
          |--*.png  
     |--valannot    (101)
          |--*.png
     |--test        (233)
          |--*.png
     |--testannot   (233)
          |--*.png
```

## To tfrecord

> python to_tfrecord.py

## Train

> python train.py

## Test

> python predict.py 

## Evaluation

> python eval.py

## Results

#### Segmentation

![](./testresults/test_1.png)

![](./testresults/test_2.png)

#### Learning rate

![](./testresults/lr.png)

#### Loss

![](./testresults/loss.png)

#### mIoU

![](./testresults/mIoU.png)