
$D$ : embedding dimensions, total/num/count

$d$ : embedding dimension, index


$$\operatorname{PE}(p, d)= \begin{cases}\sin \left(\frac{p}{10000^{d / D}}\right), & d \text { even } \\ \cos \left(\frac{p}{10000^{(d-1) / D}}\right), & d \text { odd }\end{cases}$$

