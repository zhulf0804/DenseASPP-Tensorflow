## About
A Tensorflow implementation of [denseASPP](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf).

## Backbone

+ DenseNet121
    
    ![](./test_results/dense.png)
    
    | Backbone | Pretrained | Train set | Test set |mIoU |
    | :---: | :----: | :----: | :----: | :----: |
    | DenseNet121 | No | train | val | 53.3 |

+ ResNet101

    ![](./test_results/res.png)
    
    | Backbone | Pretrained | Train set | Test set |mIoU |
    | :---: | :----: | :----: | :----: | :----: |
    | ResNet101 | Yes | train+val | test | 64.9 |
    
 
 

