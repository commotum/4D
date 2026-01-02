Euclidean geometry has a built-in symmetry:
All distances and inner products are invariant under sign reversal of coordinates.

Formally:

$$
\|\Delta t\|^2=\|-\Delta t\|^2
$$

and for any orthogonal rotation $R(\theta)$ :

$$
R(-\theta)=R(\theta)^{\top}
$$


Now look at the attention score:

$$
\ell(\Delta t)=q^{\top} R(\Delta t) k
$$


Swap past and future:

$$
\ell(-\Delta t)=q^{\top} R(-\Delta t) k=q^{\top} R(\Delta t)^{\top} k=(R(\Delta t) q)^{\top} k
$$

If you remove the causal mask in a standard Transformer:
- Future tokens contain perfect labels
- Attention can trivially copy
- Loss collapses
- Training degenerates

In MonSTER:
- Future tokens are geometrically repelled
- Even if present, they do not align in the representation space
- Copying is not linearly accessible

That's the entire point of using an indefinite metric.

A model will exploit future tokens only if doing so reduces loss more cheaply than not doing so. In MonSTER, once a head couples to time, future attention systematically increases loss, so SGD drives it to zero - even though it is technically allowed.

This is strictly stronger than a hard causal mask:
- A mask forbids peeking
- MonSTER makes peeking counterproductive

The model doesn't avoid the future because it's forbidden - it avoids the future because the geometry makes it a bad idea.

