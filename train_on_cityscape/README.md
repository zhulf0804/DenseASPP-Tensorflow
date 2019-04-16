## About
A Tensorflow implementation of [denseASPP](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf).

+ **Backbone Network:** densenet-121
+ **Datasets:** Cityscape

## Dataset
The dataset is organized as follows:
```
|--cityscape
      |--leftImg8bit_trainvaltest
                 |--leftImg8bit (5,000)
                       |--train (2975)
                       |--val   (500)
                       |--test  (1525)
      |--gtFine_trainvaltest    (5000 * 4)
                 |--gtFine
                       |--train (2975 * 4)
                       |--val   (500 * 4)
                       |--test  (1525 * 4)
     
```

## Generate filename list
> python cityscape.py

## To TFRecord files
> python to_tfrecord.py

## Train
> pytho  train.py

## Test 
> python predict.py