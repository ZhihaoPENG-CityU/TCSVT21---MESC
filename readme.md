# Environment
+ Python[3.6.9] + Tensorflow[1.13.1]
# Remark
+ Error[ModuleNotFoundError: No module named 'tensorflow.contrib']
  +   As the contrib module doesn't exist in TF2.0, it is advised to use "tf.compat.v1.keras.initializers.he_normal()" as the initializer.
+ Error[which is resulted from the case that TensorFlow 1.x migrated to 2.x]
  +   It is advised to use the "tf.compat.v1.XXX" for code compatibility processing
+ Error[RuntimeError: tf.placeholder() is not compatible with eager execution]
  +   It is advised to use the "tf.compat.v1.disable_eager_execution()"
