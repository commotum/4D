### Quaternion conjugate

The **conjugate** of a quaternion is defined as the same quaternion with its **vector (imaginary) part reversed in sign**, while its **scalar (real) part is left unchanged**.

---

### Conjugate vs. Inverse

For any nonzero quaternion $q$, the inverse is given by:

$$
q^{-1}=\frac{q^*}{\|q\|^2}
$$


If $q$ is a unit quaternion, then by definition:

$$
\|q\|=1
$$


Substituting into the inverse formula gives:

$$
q^{-1}=\frac{q^*}{1^2}=q^*
$$


Therefore, for unit quaternions, the conjugate and the inverse are the same:

$$
q^*=q^{-1}
$$

---

### Consequence for reversibility

As a result, any multiplication of a quaternion $v$ by a unit quaternion $q$, is completely reversible **by multiplying by the conjugate $q^*$ on the same side**, since for unit quaternions $q^*=q^{-1}$.

In other words, a single left-multiplication $$ v^{\prime}=q v $$ is a linear, invertible map.

As a result, any multiplication of a quaternion $v$ by a unit quaternion $q$ is completely reversible by multiplying by the conjugate $q^*$ on the same side, since for unit quaternions $q^*=q^{-1}$. 

In other words, the single left-multiplication 

$$
v^{\prime}=q v
$$

is a linear, invertible map, whose inverse is given by left-multiplication by $q^*$ :

$$
v=q^* v^{\prime}
$$