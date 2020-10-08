from typing import Optional

import tensorflow as tf


@tf.function(experimental_relax_shapes=True)
def prune_components(labels: tf.Tensor, pihat: tf.Tensor, threshold: tf.Tensor, everything=False):
    toretain = tf.squeeze(tf.where(tf.reduce_any(pihat > threshold, axis=1)), axis=1)
    if not everything:
        toretain = tf.range(
            tf.reduce_max(toretain) + 1
        )  # we can not prune everything during optimization, then vhat3_cumsum would be wrong
        return tf.cast(toretain, labels.dtype), labels
    return prune_labels(labels, tf.cast(toretain, labels.dtype))


@tf.function(experimental_relax_shapes=True)
def prune_labels(labels: tf.Tensor, toretain: Optional[tf.Tensor] = None):
    if toretain is None:
        ulabels, _ = tf.unique(labels)
        toretain = tf.sort(ulabels)
    else:
        toretain = tf.sort(toretain)
    diffs = toretain[1:] - toretain[:-1]
    missing = tf.cast(tf.where(diffs > 1), labels.dtype)
    if tf.size(missing) > 0:
        missing = tf.squeeze(missing, axis=1)
        todrop = tf.TensorArray(labels.dtype, size=tf.size(missing), infer_shape=False)
        shift = tf.cast(0, labels.dtype)
        for i in tf.range(tf.size(missing)):
            m = missing[i]
            idx = tf.where(labels > toretain[m] - shift)
            shift += diffs[m] - 1
            labels = tf.tensor_scatter_nd_sub(labels, idx, tf.repeat(diffs[m] - 1, tf.size(idx)))
            todrop = todrop.write(i, tf.range(toretain[m] + 1, toretain[m] + diffs[m]))
        todrop = todrop.concat()
        if toretain[0] > 0:
            todrop = tf.concat((tf.range(toretain[0]), todrop), axis=0)
            labels = labels - toretain[0]
        idx = tf.squeeze(
            tf.sparse.to_dense(
                tf.sets.difference(
                    tf.range(tf.reduce_max(toretain) + 1)[tf.newaxis, :],
                    tf.cast(todrop[tf.newaxis, :], dtype=labels.dtype),
                )
            )
        )
    else:
        idx = tf.cast(tf.range(tf.size(toretain)), labels.dtype)
    return idx, labels
