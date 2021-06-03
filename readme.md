# MESC-Net
+ The schematic diagrams of the learned affinity matrices under various regularization techniques. <br>
![image](https://user-images.githubusercontent.com/23076563/120636103-cf32f700-c49f-11eb-8072-496970cff4cb.png)
+ The contribution
  + The main contributions of our work are two folds. First, we propose a novel deep subspace clustering method using the maximum entropy principle, which can promote the connectivity of the learned affinity matrix within each subspace. We also theoretically prove that the learned affinity matrix satisfies the block-diagonal property under the independent subspaces assumption. Second, we design a novel deep clustering framework to explicitly decouple the auto-encoder module and the self-expressiveness module, which makes the training of deep subspace methods more efficient.

# Environment
+ Tensorflow
  + Tensorflow[2.1.0]
+ Python
  + Python[3.6.10]
+ Pytorch
  +   Pytorch implementation of DSC-Net by Guo(https://github.com/XifengGuo/DSC-Net).
# Remark
+ Error[ModuleNotFoundError: No module named 'tensorflow.contrib']
  +   As the contrib module doesn't exist in TF2.0, it is advised to use "tf.compat.v1.keras.initializers.he_normal()" as the initializer.
+ Error[which is resulted from the case that TensorFlow 1.x migrated to 2.x]
  +   It is advised to use the "tf.compat.v1.XXX" for code compatibility processing.
+ Error[RuntimeError: tf.placeholder() is not compatible with eager execution]
  +   It is advised to use the "tf.compat.v1.disable_eager_execution()".
+ If you find this work useful in your research,please cite:
  +   @article{peng2020maximum,
        author={Peng, Zhihao and Jia, Yuheng and Liu, Hui and Hou, Junhui and Zhang, Qingfu},
        title={Maximum Entropy Subspace Clustering Network},
        journal={arXiv preprint arXiv:2012.03176},
        year={2020}
    }
