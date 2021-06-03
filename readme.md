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
# Ours
+ If you find this work useful in your research, please cite:
+ @article{peng2020maximum,
  title={Maximum Entropy Subspace Clustering Network},
  author={Peng, Zhihao and Jia, Yuheng and Liu, Hui and Hou, Junhui and Zhang, Qingfu},
  journal={arXiv preprint arXiv:2012.03176},
  year={2020}
}
