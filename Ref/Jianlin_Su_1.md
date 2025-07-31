# The Transformer Positional Encodings That Have Researchers Racking Their Brains - Scientific Spaces

## Author: Su Jianlin
## Published: 2021-02-03
## Summary: A blog post surveying the many methods for adding positional information to Transformers, which are naturally permutation-invariant. It covers the two primary families of solutions: absolute encodings that are added to inputs (like in BERT) and relative encodings that modify the attention mechanism (like in T5 and XLNet). The author also explores several unconventional approaches and presents anovel fused encoding scheme that unifies absolute and relative properties.


---

Here is the translation in English:

---
title: "The Transformer Positional Encodings That Have Researchers Racking Their Brains - Scientific Spaces"
source: "[https://kexue.fm/archives/8130](https://kexue.fm/archives/8130)"
author: Su Jianlin
published: 2021-02-03
created: 2025-07-30
description: A blog post surveying the many methods for adding positional information to Transformers, which are naturally permutation-invariant. It covers the two primary families of solutions: absolute encodings that are added to inputs (like in BERT) and relative encodings that modify the attention mechanism (like in T5 and XLNet). The author also explores several unconventional approaches and presents anovel fused encoding scheme that unifies absolute and relative properties.
tags:
  - "clippings"
---
Feb 3

Unlike models such as RNNs and CNNs, the addition of positional encoding is essential for the Transformer model because a pure Attention module cannot capture the input order, i.e., it cannot distinguish between Tokens at different positions. For this, we have roughly two choices: 1. Find a way to integrate position information into the input, which constitutes the general approach of **absolute positional encoding**; 2. Find a way to fine-tune the Attention structure so that it has the ability to distinguish Tokens at different positions, which constitutes the general approach of **relative positional encoding**.

Although we mainly talk about the two major categories of absolute and relative positional encoding, each category can actually be derived into various variants. For this, researchers have taken great pains and racked their brains. In addition, there are some unconventional positional encodings. In this article, let's appreciate the "Eight Immortals crossing the sea, each showing their unique prowess" style of encoding schemes that researchers have constructed to better express positional information.

## Absolute Positional Encoding

Formally, absolute positional encoding is a relatively simple solution, but even so, it doesn't stop researchers from coming up with fantastic ideas and many variations. Generally, absolute positional encoding is added to the input: the position vector $p_k$ is added to the $k$-th input vector $x_k$ to become $x_k + p_k$, where $p_k$ only depends on the position index $k$.

### Trainable

Obviously, the most straightforward approach for absolute positional encoding is not to design anything specific, but to directly **treat the positional encoding as a trainable parameter**. For example, if the maximum length is 512 and the encoding dimension is 768, then a $512 \times 768$ matrix is initialized as the position vectors and updated during the training process. This is the positional encoding used by current models like BERT and GPT. In fact, it can be traced back even earlier, for instance, it was used in Facebook's 2017 paper "[Convolutional Sequence to Sequence Learning](https://papers.cool/arxiv/1705.03122)".

For this type of trainable absolute positional encoding, it is generally believed that its disadvantage is the lack of extrapolation capability. That is, if the pre-training maximum length is 512, it can only process sentences up to length 512 and no longer. Of course, one can also randomly initialize the position vectors for positions beyond 512 and then continue fine-tuning. However, my recent research shows that through hierarchical decomposition, absolute positional encoding can be extrapolated to a sufficiently long range while maintaining decent performance. For details, please refer to my previous blog post "[Hierarchical Decomposition Positional Encoding, Allowing BERT to Process Ultra-Long Texts](https://kexue.fm/archives/7947)". Therefore, extrapolation is not necessarily a significant drawback of absolute positional encoding.

### Trigonometric

Trigonometric positional encoding, also commonly known as **Sinusoidal positional encoding**, is an explicit solution proposed in Google's paper "[Attention is All You Need](https://papers.cool/arxiv/1706.03762)":
$$
\begin{equation}\left\{\begin{aligned}&\boldsymbol{p}_{k,2i}=\sin\Big(k/10000^{2i/d}\Big)\\
&\boldsymbol{p}_{k, 2i+1}=\cos\Big(k/10000^{2i/d}\Big)
\end{aligned}\right.\end{equation}
$$
Here, $p_{k,2i}$ and $p_{k, 2i+1}$ are the $2i$-th and $(2i+1)$-th components of the encoding vector for position $k$, respectively, and $d$ is the dimension of the position vector.

Clearly, the characteristic of trigonometric positional encoding is that it has an explicit generation rule, so it can be expected to have a certain degree of extrapolation capability. Another reason for using it is that since $\sin(\alpha+\beta)=\sin\alpha\cos\beta+\cos\alpha\sin\beta$ and $\cos(\alpha+\beta)=\cos\alpha\cos\beta−\sin\alpha\sin\beta$, this indicates that the vector for position $\alpha+\beta$ can be expressed as a linear combination of the vectors for positions $\alpha$ and $\beta$, which provides the possibility of expressing relative position information. But strangely, we rarely see works that directly use this form of absolute positional encoding now, for unknown reasons.

### Recursive

In principle, an RNN model does not require positional encoding, as its structure inherently allows for learning positional information (because **recursion means we can train a "counting" model**). Therefore, if an RNN layer is placed after the input and before the Transformer, theoretically, positional encoding is not needed. Similarly, we can also use an RNN model to learn a kind of absolute positional encoding, for example, starting from a vector $\boldsymbol{p}_0$ and obtaining the encoding vectors for each position through the recursive formula $p_{k+1}=f(p_k)$.

The ICML 2020 paper "[Learning to Encode Position for Transformer with Continuous Dynamical Model](https://papers.cool/arxiv/2003.09229)" pushed this idea to the extreme. It proposed modeling positional encoding using a differential equation (ODE) $d\boldsymbol{p}_t/dt=\boldsymbol{h}(\boldsymbol{p}_t,t)$, a scheme called FLOATER. Obviously, FLOATER is also a recursive model, and the function $h(p_t,t)$ can be modeled by a neural network, so this type of differential equation is also called a Neural ODE. Work on this has been gradually increasing recently.

Theoretically, positional encoding based on recursive models also has good extrapolation capability, and it is more flexible than trigonometric positional encoding (for example, it can be easily proven that trigonometric positional encoding is a special solution of FLOATER). However, it is clear that recursive positional encoding sacrifices a certain degree of parallelism and may create a speed bottleneck.

### Multiplicative

We just mentioned that the combination of input $\boldsymbol{x}_k$ and absolute positional encoding $p_k$ is generally $x_k + p_k$. Are there any "unconventional" combination methods? For example, $x_k \otimes p_k$ (element-wise multiplication)? When we build models, we have various ways to fuse two vectors, such as **addition, multiplication, and even concatenation**. Why does everyone default to considering only addition when doing absolute positional encoding?

I'm sorry, I don't know the answer either. Perhaps everyone defaults to addition because the addition of vectors has a clearer geometric meaning, but for deep learning models, this geometric meaning has little practical value. A recent experiment I saw seems to show that replacing "addition" with "multiplication," i.e., the $\boldsymbol{x}_k \otimes \boldsymbol{p}_k$ method, seems to achieve better results than $x_k + p_k$. I haven't done a full comparison of the effects myself, but I am just providing this as a possibility. For the source of the experiment, you can refer to "[Chinese Language Model Research: (1) Multiplicative Positional Encoding](https://zhuanlan.zhihu.com/p/183234823)".

## Relative Positional Encoding

Relative position does not fully model the position information of each input. Instead, it considers the relative distance between the current position and the attended position when calculating Attention. Since natural language generally relies more on relative positions, relative positional encoding usually performs excellently. For relative positional encoding, its flexibility is greater, further reflecting the "unrestrained and imaginative" nature of researchers.

### Classic Style

Relative positional encoding originated from Google's paper "[Self-Attention with Relative Position Representations](https://papers.cool/arxiv/1803.02155)". The NEZHA model open-sourced by Huawei also uses this type of positional encoding, and subsequent variations of relative positional encoding are mostly simple modifications that follow this pattern.

It is generally believed that relative positional encoding was inspired by absolute positional encoding. Consider a general Attention with absolute positional encoding:
$$
\begin{equation}\left\{\begin{aligned}
\boldsymbol{q}_i =&\, (\boldsymbol{x}_i + \boldsymbol{p}_i)\boldsymbol{W}_Q \\
\boldsymbol{k}_j =&\, (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_K \\
\boldsymbol{v}_j =&\, (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_V \\
a_{i,j} =&\, \text{softmax}\left(\boldsymbol{q}_i \boldsymbol{k}_j^{\top}\right)\\
\boldsymbol{o}_i =&\, \sum_j a_{i,j}\boldsymbol{v}_j
\end{aligned}\right.\end{equation}
$$Here, `softmax` is normalized over the $j$ dimension, and all vectors are row vectors. Let's initially expand $q_i k_j^\top$:$$
\text{(3) } q_i k_j^\top = (x_i+p_i)W_Q W_K^\top(x_j+p_j)^\top = (x_i W_Q + p_i W_Q)(W_K^\top x_j^\top + W_K^\top p_j^\top)
$$To introduce relative position information, Google removed the first position term and changed the second term $p_j W_K$ to a two-argument position vector $R_{i,j}^K$, becoming:$$
\text{(4) } a_{i,j} = \text{softmax}(x_i W_Q (x_j W_K + R_{i,j}^K)^\top)
$$And in $o_i = \sum_j a_{i,j} v_j = \sum_j a_{i,j} (x_j W_V + p_j W_V)$, $p_j W_V$ is replaced with $R_{i,j}^V$:$$
\text{(5) } o_i = \sum_j a_{i,j} (x_j W_V + R_{i,j}^V)
$$The so-called relative position changes the vectors $R_{i,j}^K, R_{i,j}^V$, which originally depended on the two-argument coordinates $(i,j)$, to depend only on the relative distance $i-j$. Furthermore, it is usually truncated to adapt to arbitrary distances:$$
\text{(6) } R_{i,j}^K = p^K[\text{clip}(i-j, p_{\text{min}}, p_{\text{max}})], \quad R_{i,j}^V = p^V[\text{clip}(i-j, p_{\text{min}}, p_{\text{max}})]
$$
In this way, **with only a finite number of positional encodings, relative positions of any length can be expressed (due to truncation)**. Regardless of whether $p^K, p^V$ are chosen to be trainable or trigonometric, the requirement of processing arbitrary length text can be met.

### XLNet Style

XLNet-style positional encoding actually originates from the Transformer-XL paper "[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://papers.cool/arxiv/1901.02860)". However, it was only after the [XLNET](https://papers.cool/arxiv/1906.08237) model, which uses the Transformer-XL architecture, surpassed BERT to some extent that Transformer-XL became widely known. Therefore, this positional encoding is often referred to by the name XLNet.

XLNet-style positional encoding comes from the complete expansion of the aforementioned $\boldsymbol{q}_i \boldsymbol{k}_j^{\top}$:
$$\text{(7) } q_i k_j^\top = x_i W_Q W_K^\top x_j^\top + x_i W_Q W_K^\top p_j^\top + p_i W_Q W_K^\top x_j^\top + p_i W_Q W_K^\top p_j^\top$$
Transformer-XL's approach is simple: it directly replaces $p_j$ with the relative position vector $R_{i-j}$. As for the two $p_i$ terms, they are replaced by two trainable vectors $u, v$:
$$\text{(8) } x_i W_Q W_K^\top x_j^\top + x_i W_Q W_K^\top R_{i-j}^\top + u W_Q W_K^\top x_j^\top + v W_Q W_K^\top R_{i-j}^\top$$
The $R_{i-j}$ in this encoding method is not truncated as in equation $(6)$, but instead directly uses the Sinusoidal generation scheme. Since the encoding space of $R_{i-j}$ is not necessarily the same as $x_j$, the $W_K^\top$ in front of $R_{i-j}$ is replaced with another independent matrix $W_{K,R}^\top$. Also, $u W_Q$ and $v W_Q$ can be directly merged into single vectors $u$ and $v$, so the final formula used is:
$$\text{(9) } x_i W_Q W_K^\top x_j^\top + x_i W_Q W_{K,R}^\top R_{i-j}^\top + u W_K^\top x_j^\top + v W_{K,R}^\top R_{i-j}^\top$$
Furthermore, the positional bias on $v_j$ is directly removed, i.e., $o_i = \sum_j a_{i,j} x_j W_V$. **It seems that starting from this work, subsequent relative positional encodings are only added to the Attention matrix and not to $v_j$.**

### T5 Style

The T5 model comes from the article "[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://papers.cool/arxiv/1910.10683)", which uses a simpler relative positional encoding. The idea still stems from the expansion in $\eqref{eq:qk-exp}$. If we must analyze the meaning of each term, they can be understood as a combination of "input-input", "input-position", "position-input", and "position-position" attentions. If we believe that input information and position information should be independent (decoupled), then they should not have excessive interaction. Therefore, the "input-position" and "position-input" attention terms can be removed. The term $p_i W_Q W_K^\top p_j^\top$ is actually just a scalar that depends only on $(i,j)$, so we can directly train it as a parameter, simplifying to:
$$\text{(10) } x_i W_Q W_K^\top x_j^\top + \beta_{i,j}$$
To put it bluntly, it is merely **adding a trainable bias term** to the Attention matrix. And just like the XLNet style, the positional bias on $v_j$ is directly removed. A similar idea is also present in the TUPE positional encoding proposed by Microsoft in their ICLR 2021 paper "[Rethinking Positional Encoding in Language Pre-training](https://papers.cool/arxiv/2006.15595)".

What's rather "unique" is that, unlike conventional positional encodings that treat $\boldsymbol{\beta}_{i,j}$ as a function of $i-j$ and apply truncation, T5 performs a "**bucketing**" process for relative positions. That is, a position with a relative distance of $i-j$ actually corresponds to position $f(i-j)$, with the mapping as follows:
$$
\begin{array}{c|cccccccccccccccc}
\hline
i-j & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 & 14 & 15 \\
\hline
f(i-j) & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 8 & 8 & 8 & 9 & 9 & 9 & 9 \\
\hline
\hline
i-j & 16 & 17 & 18 & 19 & 20 & 21 & 22 & 23 & 24 & 25 & 26 & 27 & 28 & 29 & 30 & \cdots \\
\hline
f(i-j) & 10 & 10 & 10 & 10 & 10 & 10 & 10 & 10 & 11 & 11 & 11 & 11 & 11 & 11 & 11 & \cdots \\
\hline
\end{array}
$$
Readers can look at the source code for the specific mapping implementation. The idea behind this design is quite intuitive: for nearby positions (0-7), we need finer discrimination, so they are each assigned an independent positional encoding. For slightly more distant positions (e.g., 8-11), we don't need to distinguish them as clearly, so they can share a positional encoding. The farther the distance, the larger the shared range can be, until a specified range is reached, after which it is clipped.

### DeBERTa Style

DeBERTa was also developed by Microsoft and was released last June in the paper "[DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://papers.cool/arxiv/2006.03654)". It has recently become popular again, firstly because it was officially accepted at ICLR 2021, and secondly because it topped the [SuperGLUE](https://super.gluebenchmark.com/) leaderboard, slightly outperforming T5.

In fact, DeBERTa's main improvement is also in its positional encoding. It also starts from the expansion in $\eqref{eq:qk-exp}$. While T5 simply removed the 2nd and 3rd terms, keeping only the 4th term and replacing it with relative positional encoding, DeBERTa does the opposite. It discards the 4th term, keeps the 2nd and 3rd terms, and replaces them with relative positional encodings (Indeed, **research is about enumerating all permutations to see which is optimal**):
$$\text{(11) } q_i k_j^\top = x_i W_Q W_K^\top x_j^\top + x_i W_Q W_K^\top R_{i,j}^\top + R_{j,i} W_Q W_K^\top x_j^\top$$
The design of $R_{i,j}$ is also truncated like in equation $(6)$, with nothing particularly special.

However, an interesting aspect of DeBERTa is that it provides a **new perspective on using relative and absolute positional encodings**. It points out that most NLP tasks may only need relative position information, but there are indeed scenarios where absolute position information is more helpful. Thus, it divides the entire model into two parts for understanding. Taking the Base version of the MLM pre-training model as an example, it has 13 layers. The first 11 layers only use relative positional encoding, and this part is called the Encoder. The last 2 layers incorporate absolute position information, which it calls the Decoder, even coining the acronym EMD (Enhanced Mask Decoder). For fine-tuning downstream tasks, it uses the first 11 layers of the Encoder plus 1 layer of the Decoder.

The performance on SuperGLUE affirms the value of DeBERTa, but **the various names in its paper are extremely uncomfortable**. For instance, what it calls "Encoder" and "Decoder" can easily mislead people into thinking it's a Seq2Seq model. The acronym EMD also shares a name with Earth Mover's Distance. While name collisions are sometimes unavoidable, the names it collides with are well-known concepts in the ML community, which can easily cause misunderstanding. It's hard to know what the authors were thinking...

## Other Positional Encodings

Although absolute and relative positional encodings come in many varieties, they are still within the classic scope. From the descriptions above, we can still sense a strong sense of established patterns. Besides these, there are some that don't follow the conventional playbook, yet they also express positional information.

### CNN Style

Although the classic work using CNNs for NLP, "[Convolutional Sequence to Sequence Learning](https://papers.cool/arxiv/1705.03122)", incorporated positional encoding, we know that general CNN models, especially in computer vision, do not have additional positional encoding. So how exactly do CNN models capture position information?

If I were to answer, the answer might be that the anisotropy of the convolution kernel allows it to distinguish relative positions in different directions. However, the ICLR 2020 paper "[How Much Position Information Do Convolutional Neural Networks Encode?](https://papers.cool/arxiv/2001.08248)" gives a possibly surprising answer: **The position information in CNN models is leaked by Zero Padding!**

We know that to keep the feature map size constant during the convolution encoding process, we usually pad the input with a certain number of zeros. This paper shows that this operation enables the model to recognize positional information. That is to say, while the anisotropy of the convolution kernel is important, the most fundamental thing is the existence of zero padding. It is then conceivable that what is actually extracted is the relative distance from the current position to the padding boundary.

However, this ability depends on the locality of CNNs. A global, prior-less structure like Attention is not suitable. For readers who are only concerned with Transformer positional encoding schemes, this can be considered as broadening your horizons.

### Complex Number Style

Complex number positional encoding is perhaps the **most unconventional** positional encoding scheme. It comes from the ICLR 2020 paper "[Encoding word order in complex embeddings](https://papers.cool/arxiv/1912.12333)". The main idea of the paper is to combine the properties of complex numbers with some basic principles to derive its positional encoding form (Complex Order) as:
$$\begin{equation}\left[r_{j, 1} e^{\text{i}\left(\omega_{j, 1} k+\theta_{j, 1}\right)}, \ldots, r_{j, 2} e^{\text{i}\left(\omega_{j, 2} k+\theta_{j, 2}\right)}, \cdots, r_{j, d} e^{\text{i}\left(\omega_{j, d} k+\theta_{j, d}\right)}\right]\label{eq:complex}\end{equation}$$
Here, $i$ is the imaginary unit, $j$ represents a certain word, $k$ represents the position of that word, and
$$\text{(13) } \boldsymbol{r}_j = [r_{j,1}, r_{j,2}, \cdots, r_{j,d}], \quad \boldsymbol{\omega}_j = [\omega_{j,1}, \omega_{j,2}, \cdots, \omega_{j,d}], \quad \boldsymbol{\theta}_j = [\theta_{j,1}, \theta_{j,2}, \cdots, \theta_{j,d}]$$
represent three sets of word vectors for word $j$. You read that right, it does assume that each word has three sets of position-independent word vectors (of course, they can share parameters in some way to degenerate into two or even one set), and then the position-dependent word vector for position $k$ is calculated according to the formula above.

Do you think introducing multiple sets of word vectors is its most unconventional part? Not at all! We see that equation $\eqref{eq:complex}$ is still in complex form. Guess what it does next? Convert it to a real number? No, it uses it directly in a **complex model**! That is to say, it follows a complex model route, where not only the input Embedding layer is complex, but every Transformer layer inside is also complex. It even implements and compares complex versions of FastText, LSTM, CNN, and other models! The first author of this article is Benyou Wang, and a search reveals that his related work is basically centered around complex models, making him a die-hard fan of complex models.

### Fused Style

Coincidentally, using the form of complex numbers, I have also conceived a rather clever positional encoding that can **fuse absolute and relative positional encodings into one**. I'm sharing it here, and interested readers are welcome to discuss and research it together.

For simplicity, let's first assume that $\boldsymbol{q}_m, \boldsymbol{k}_n$ are two-dimensional row vectors at positions $m, n$ respectively. Since they are two-dimensional, we can treat them as complex numbers for computation. We know that the key to Attention is the dot product of vectors, which can be expressed in complex numbers as:
$$\text{(14) } \langle \boldsymbol{q}_m, \boldsymbol{k}_n \rangle = \text{Re}[\boldsymbol{q}_m \boldsymbol{k}_n^*]$$
Here, $*$ is the complex conjugate, the multiplication on the right is ordinary complex multiplication, and $\text{Re}[]$ means taking the real part of the result. The above formula says:

> The dot product of two 2D vectors is equal to the real part of the product of one complex number and the conjugate of the other, when treating them as complex numbers.

If we multiply $\boldsymbol{q}_m, \boldsymbol{k}_n$ by $e^{im\theta}, e^{in\theta}$ respectively to get $q_m e^{im\theta}, k_n e^{in\theta}$, it's equivalent to giving them absolute positional encoding (because it explicitly depends on absolute positions $m, n$). Then, putting it into the dot product, we have:
$$\text{(15) } \langle q_m e^{im\theta}, k_n e^{in\theta} \rangle = \text{Re}[(q_m e^{im\theta})(k_n e^{in\theta})^*] = \text{Re}[\boldsymbol{q}_m \boldsymbol{k}_n^* e^{i(m-n)\theta}]$$
Quite interestingly, the dot product only depends on the relative position $m-n$! This cleverly fuses absolute and relative positions together.

Note that we are not as "crazy" as Complex Order; the above operations are essentially still within the real number domain, we just used complex numbers to complete some derivations. From the above result, for a 2D real vector $[x,y]$ at position $n$, if we treat it as a complex number and multiply by $e^{in\theta}$, we get the identity:
$$\text{(16) } (x+yi)e^{in\theta} = (x\cos n\theta - y\sin n\theta) + i(x\sin n\theta + y\cos n\theta)$$
This means that by using the transformation:
$$\text{(17) } \begin{pmatrix} x \\ y \end{pmatrix} \rightarrow \begin{pmatrix} x\cos n\theta - y\sin n\theta \\ x\sin n\theta + y\cos n\theta \end{pmatrix} = \begin{pmatrix} x \\ y \end{pmatrix} \cos n\theta + \begin{pmatrix} -y \\ x \end{pmatrix} \sin n\theta$$
to imbue $[x,y]$ with absolute position information, it becomes equivalent to relative positional encoding during the Attention operation. If the vector has more than two dimensions, we can consider performing the same operation on every two dimensions as a group, and the $\theta$ for each group can be different.

In this way, we obtain a positional encoding scheme that integrates absolute and relative positions. Formally, it looks a bit like multiplicative absolute positional encoding. By applying this encoding to $\boldsymbol{q}$ and $\boldsymbol{k}$, the effect is equivalent to relative positional encoding. And if explicit absolute position information is still needed, this encoding can also be applied to $\boldsymbol{v}$ simultaneously. In summary, through an absolute position operation, we can achieve the effect of absolute position and also the effect of relative position. Preliminary experiments show that it can work, but it has not been fully validated. Everyone is welcome to try it and discuss.

## Article Summary

This article summarizes some works on positional encoding, broadly divided into absolute, relative, and unconventional types, from which we can see various magical operations. Finally, the author shares a fused absolute and relative encoding scheme of his own design for interested readers to reference.

***
***Please include the address of this article when reprinting:** [https://kexue.fm/archives/8130](https://kexue.fm/archives/8130 "The Transformer Positional Encodings That Have Researchers Racking Their Brains")*

***For more detailed reprinting matters, please refer to:*** "[Scientific Spaces FAQ](https://kexue.fm/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)"

**If you have any doubts or suggestions, you are welcome to continue the discussion in the comments section below.**

**If you find this article useful, feel free to [Share](https://kexue.fm/archives/#share) / [Donate](https://kexue.fm/archives/#pay) to this article. The purpose of donations is not to gain income, but to know how many readers' sincere attention Scientific Spaces has received. Of course, ignoring it will not affect your reading. Welcome and thank you again!**

**If you need to cite this article, please refer to:**

Su, Jianlin. (Feb. 03, 2021). "The Transformer Positional Encodings That Have Researchers Racking Their Brains" [Blog post]. Retrieved from [https://kexue.fm/archives/8130](https://kexue.fm/archives/8130)

```bibtex
@online{kexuefm-8130,
  title={让研究人员绞尽脑汁的Transformer位置编码},
  author={苏剑林},
  year={2021},
  month={Feb},
  url={\url{https://kexue.fm/archives/8130}},
}
```
< [A Theoretical Analysis Attempt on the Repetitive Decoding Phenomenon in Seq2Seq](https://kexue.fm/archives/8128 "A Theoretical Analysis Attempt on the Repetitive Decoding Phenomenon in Seq2Seq") | [How a Binarized Word Vector Model Got Connected to Fruit Flies?](https://kexue.fm/archives/8159 "How a Binarized Word Vector Model Got Connected to Fruit Flies?") >

---

Original:

---
title: "让研究人员绞尽脑汁的Transformer位置编码 - 科学空间|Scientific Spaces"
source: "https://kexue.fm/archives/8130"
author:
published:
created: 2025-07-31
description: "不同于RNN、CNN等模型，对于Transformer模型来说，位置编码的加入是必不可少的，因为纯粹的Attention模块是无法捕捉输入顺序的，即无法区分不同位置的Token。为此我们大体有两..."
tags:
  - "clippings"
---
3 Feb

不同于RNN、CNN等模型，对于Transformer模型来说，位置编码的加入是必不可少的，因为纯粹的Attention模块是无法捕捉输入顺序的，即无法区分不同位置的Token。为此我们大体有两个选择：1、想办法将位置信息融入到输入中，这构成了 绝对位置编码 的一般做法；2、想办法微调一下Attention结构，使得它有能力分辨不同位置的Token，这构成了 相对位置编码 的一般做法。

虽然说起来主要就是绝对位置编码和相对位置编码两大类，但每一类其实又能衍生出各种各样的变种，为此研究人员可算是煞费苦心、绞尽脑汁了，此外还有一些不按套路出牌的位置编码。本文就让我们来欣赏一下研究人员为了更好地表达位置信息所构建出来的“八仙过海，各显神通”般的编码方案。

## 绝对位置编码

形式上来看，绝对位置编码是相对简单的一种方案，但即便如此，也不妨碍各路研究人员的奇思妙想，也有不少的变种。一般来说，绝对位置编码会加到输入中：在输入的第 $k$ 个向量 $xk$ 中加入位置向量 $pk$ 变为 $xk+pk$ ，其中 $pk$ 只依赖于位置编号 $k$ 。

### 训练式

很显然，绝对位置编码的一个最朴素方案是不特意去设计什么，而是直接 将位置编码当作可训练参数 ，比如最大长度为512，编码维度为768，那么就初始化一个 $512\times 768$ 的矩阵作为位置向量，让它随着训练过程更新。现在的BERT、GPT等模型所用的就是这种位置编码，事实上它还可以追溯得更早，比如2017年Facebook的 [《Convolutional Sequence to Sequence Learning》](https://papers.cool/arxiv/1705.03122) 就已经用到了它。

对于这种训练式的绝对位置编码，一般的认为它的缺点是没有外推性，即如果预训练最大长度为512的话，那么最多就只能处理长度为512的句子，再长就处理不了了。当然，也可以将超过512的位置向量随机初始化，然后继续微调。但笔者最近的研究表明，通过层次分解的方式，可以使得绝对位置编码能外推到足够长的范围，同时保持还不错的效果，细节请参考笔者之前的博文 [《层次分解位置编码，让BERT可以处理超长文本》](https://kexue.fm/archives/7947) 。因此，其实外推性也不是绝对位置编码的明显缺点。

### 三角式

三角函数式位置编码，一般也称为 Sinusoidal位置编码 ，是Google的论文 [《Attention is All You Need》](https://papers.cool/arxiv/1706.03762) 所提出来的一个显式解：  
$$
\begin{equation}\left\{\begin{aligned}&\boldsymbol{p}_{k,2i}=\sin\Big(k/10000^{2i/d}\Big)\\ 
&\boldsymbol{p}_{k, 2i+1}=\cos\Big(k/10000^{2i/d}\Big) 
\end{aligned}\right.\end{equation}
$$
  
其中 $pk,2i,pk,2i+1$ 分别是位置 $k$ 的编码向量的第 $2i,2i+1$ 个分量， $d$ 是位置向量的维度。

很明显，三角函数式位置编码的特点是有显式的生成规律，因此可以期望于它有一定的外推性。另外一个使用它的理由是：由于 $\sin(\alpha+\beta)=\sin\alpha\cos\beta+\cos\alpha\sin\beta$ 以及 $cos⁡(α+β)=cos⁡αcos⁡β−sin⁡αsin⁡β$ ，这表明位置 $α+β$ 的向量可以表示成位置 $α$ 和位置 $β$ 的向量组合，这提供了表达相对位置信息的可能性。但很奇怪的是，现在我们很少能看到直接使用这种形式的绝对位置编码的工作，原因不详。

### 递归式

原则上来说，RNN模型不需要位置编码，它在结构上就自带了学习到位置信息的可能性（因为 递归就意味着我们可以训练一个“数数”模型 ），因此，如果在输入后面先接一层RNN，然后再接Transformer，那么理论上就不需要加位置编码了。同理，我们也可以用RNN模型来学习一种绝对位置编码，比如从一个向量 $\boldsymbol{p}_0$ 出发，通过递归格式 $pk+1=f(pk)$ 来得到各个位置的编码向量。

ICML 2020的论文 [《Learning to Encode Position for Transformer with Continuous Dynamical Model》](https://papers.cool/arxiv/2003.09229) 把这个思想推到了极致，它提出了用微分方程（ODE） $d\boldsymbol{p}_t/dt=\boldsymbol{h}(\boldsymbol{p}_t,t)$ 的方式来建模位置编码，该方案称之为FLOATER。显然，FLOATER也属于递归模型，函数 $h(pt,t)$ 可以通过神经网络来建模，因此这种微分方程也称为神经微分方程，关于它的工作最近也逐渐多了起来。

理论上来说，基于递归模型的位置编码也具有比较好的外推性，同时它也比三角函数式的位置编码有更好的灵活性（比如容易证明三角函数式的位置编码就是FLOATER的某个特解）。但是很明显，递归形式的位置编码牺牲了一定的并行性，可能会带速度瓶颈。

### 相乘式

刚才我们说到，输入 $\boldsymbol{x}_k$ 与绝对位置编码 $pk$ 的组合方式一般是 $xk+pk$ ，那有没有“不一般”的组合方式呢？比如 $xk⊗pk$ （逐位相乘）？我们平时在搭建模型的时候，对于融合两个向量有多种方式， 相加、相乘甚至拼接都是可以考虑的 ，怎么大家在做绝对位置编码的时候，都默认只考虑相加了？

很抱歉，笔者也不知道答案。可能大家默认选择相加是因为向量的相加具有比较鲜明的几何意义，但是对于深度学习模型来说，这种几何意义其实没有什么实际的价值。最近笔者看到的一个实验显示，似乎将“加”换成“乘”，也就是 $\boldsymbol{x}_k \otimes \boldsymbol{p}_k$ 的方式，似乎比 $xk+pk$ 能取得更好的结果。具体效果笔者也没有完整对比过，只是提供这么一种可能性。关于实验来源，可以参考 [《中文语言模型研究：(1) 乘性位置编码》](https://zhuanlan.zhihu.com/p/183234823) 。

## 相对位置编码

相对位置并没有完整建模每个输入的位置信息，而是在算Attention的时候考虑当前位置与被Attention的位置的相对距离，由于自然语言一般更依赖于相对位置，所以相对位置编码通常也有着优秀的表现。对于相对位置编码来说，它的灵活性更大，更加体现出了研究人员的“天马行空”。

### 经典式

相对位置编码起源于Google的论文 [《Self-Attention with Relative Position Representations》](https://papers.cool/arxiv/1803.02155) ，华为开源的NEZHA模型也用到了这种位置编码，后面各种相对位置编码变体基本也是依葫芦画瓢的简单修改。

一般认为，相对位置编码是由绝对位置编码启发而来，考虑一般的带绝对位置编码的Attention：  
$$
\begin{equation}\left\{\begin{aligned} 
\boldsymbol{q}_i =&\, (\boldsymbol{x}_i + \boldsymbol{p}_i)\boldsymbol{W}_Q \\ 
\boldsymbol{k}_j =&\, (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_K \\ 
\boldsymbol{v}_j =&\, (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_V \\ 
a_{i,j} =&\, softmax\left(\boldsymbol{q}_i \boldsymbol{k}_j^{\top}\right)\\ 
\boldsymbol{o}_i =&\, \sum_j a_{i,j}\boldsymbol{v}_j 
\end{aligned}\right.\end{equation}
$$
  
其中 $softmax$ 对 $j$ 那一维归一化，这里的向量都是指行向量。我们初步展开 $qikj⊤$ ：  
$$
(3)qikj⊤=(xi+pi)WQWK⊤(xj+pj)⊤=(xiWQ+piWQ)(WK⊤xj⊤+WK⊤pj⊤)
$$
  
为了引入相对位置信息，Google把第一项位置去掉，第二项 $pjWK$ 改为二元位置向量 $Ri,jK$ ，变成  
$$
(4)ai,j=softmax(xiWQ(xjWK+Ri,jK)⊤)
$$
  
以及 $oi=∑jai,jvj=∑jai,j(xjWV+pjWV)$ 中的 $pjWV$ 换成 $Ri,jV$ ：  
$$
(5)oi=∑jai,j(xjWV+Ri,jV)
$$
  
所谓相对位置，是将本来依赖于二元坐标 $(i,j)$ 的向量 $Ri,jK,Ri,jV$ ，改为只依赖于相对距离 $i−j$ ，并且通常来说会进行截断，以适应不同任意的距离  
$$
(6)Ri,jK=pK[clip(i−j,pmin,pmax)]Ri,jV=pV[clip(i−j,pmin,pmax)]
$$
  
这样一来， 只需要有限个位置编码，就可以表达出任意长度的相对位置（因为进行了截断） ，不管 $pK,pV$ 是选择可训练式的还是三角函数式的，都可以达到处理任意长度文本的需求。

### XLNET式

XLNET式位置编码其实源自Transformer-XL的论文 [《Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context》](https://papers.cool/arxiv/1901.02860) ，只不过因为使用了Transformer-XL架构的 [XLNET](https://papers.cool/arxiv/1906.08237) 模型并在一定程度上超过了BERT后，Transformer-XL才算广为人知，因此这种位置编码通常也被冠以XLNET之名。

XLNET式位置编码源于对上述 $\boldsymbol{q}_i \boldsymbol{k}_j^{\top}$ 的完全展开：  
$$
(7)qikj⊤=xiWQWK⊤xj⊤+xiWQWK⊤pj⊤+piWQWK⊤xj⊤+piWQWK⊤pj⊤
$$
  
Transformer-XL的做法很简单，直接将 $pj$ 替换为相对位置向量 $Ri−j$ ，至于两个 $pi$ ，则干脆替换为两个可训练的向量 $u,v$ ：  
$$
(8)xiWQWK⊤xj⊤+xiWQWK⊤Ri−j⊤+uWQWK⊤xj⊤+vWQWK⊤Ri−j⊤
$$
  
该编码方式中的 $Ri−j$ 没有像式 $(6)$ 那样进行截断，而是直接用了Sinusoidal式的生成方案，由于 $Ri−j$ 的编码空间与 $xj$ 不一定相同，所以 $Ri−j$ 前面的 $WK⊤$ 换了另一个独立的矩阵 $WK,R⊤$ ，还有 $uWQ$ 、 $vWQ$ 可以直接合并为单个 $u$ 、 $v$ ，所以最终使用的式子是  
$$
(9)xiWQWK⊤xj⊤+xiWQWK,R⊤Ri−j⊤+uWK⊤xj⊤+vWK,R⊤Ri−j⊤
$$
  
此外， $vj$ 上的位置偏置就直接去掉了，即直接令 $oi=∑jai,jxjWV$ 。 似乎从这个工作开始，后面的相对位置编码都只加到Attention矩阵上去，而不加到 $vj$ 上去了。

### T5式

T5模型出自文章 [《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》](https://papers.cool/arxiv/1910.10683) ，里边用到了一种更简单的相对位置编码。思路依然源自展开式 $\eqref{eq:qk-exp}$ ，如果非要分析每一项的含义，那么可以分别理解为“ 输入-输入 ”、“ 输入-位置 ”、“ 位置-输入 ”、“ 位置-位置 ”四项注意力的组合。如果我们认为输入信息与位置信息应该是独立（解耦）的，那么它们就不应该有过多的交互，所以“输入-位置”、“位置-输入”两项Attention可以删掉，而 $piWQWK⊤pj⊤$ 实际上只是一个只依赖于 $(i,j)$ 的标量，我们可以直接将它作为参数训练出来，即简化为  
$$
(10)xiWQWK⊤xj⊤+βi,j
$$
  
说白了，它仅仅是在Attention矩阵的基础上 加一个可训练的偏置项 而已，而跟XLNET式一样，在 $vj$ 上的位置偏置则直接被去掉了。包含同样的思想的还有微软在ICLR 2021的论文 [《Rethinking Positional Encoding in Language Pre-training》](https://papers.cool/arxiv/2006.15595) 中提出的TUPE位置编码。

比较“别致”的是，不同于常规位置编码对将 $\boldsymbol{\beta}_{i,j}$ 视为 $i−j$ 的函数并进行截断的做法，T5对相对位置进行了一个“ 分桶 ”处理，即相对位置是 $i−j$ 的位置实际上对应的是 $f(i−j)$ 位置，映射关系如下：  
$$
i−j0123456789101112131415f(i−j)0123456788889999i−j161718192021222324252627282930⋯f(i−j)101010101010101111111111111111⋯
$$
  
具体的映射代码，读者自行看源码就好。这个设计的思路其实也很直观，就是比较邻近的位置（0～7），我们需要比较得精细一些，所以给它们都分配一个独立的位置编码，至于稍远的位置（比如8～11），我们不用区分得太清楚，所以它们可以共用一个位置编码，距离越远，共用的范围就可以越大，直到达到指定范围再clip。

### DeBERTa式

DeBERTa也是微软搞的，去年6月就发出来了，论文为 [《DeBERTa: Decoding-enhanced BERT with Disentangled Attention》](https://papers.cool/arxiv/2006.03654) ，最近又小小地火了一把，一是因为它正式中了ICLR 2021，二则是它登上 [SuperGLUE](https://super.gluebenchmark.com/) 的榜首，成绩稍微超过了T5。

其实DeBERTa的主要改进也是在位置编码上，同样还是从展开式 $\eqref{eq:qk-exp}$ 出发，T5是干脆去掉了第2、3项，只保留第4项并替换为相对位置编码，而DeBERTa则刚刚相反，它扔掉了第4项，保留第2、3项并且替换为相对位置编码（果然， 科研就是枚举所有的排列组合看哪个最优 ）：  
$$
(11)qikj⊤=xiWQWK⊤xj⊤+xiWQWK⊤Ri,j⊤+Rj,iWQWK⊤xj⊤
$$
  
至于 $Ri,j$ 的设计也是像式 $(6)$ 那样进行截断的，没有特别的地方。

不过，DeBERTa比较有意思的地方，是提供了 使用相对位置和绝对位置编码的一个新视角 ，它指出NLP的大多数任务可能都只需要相对位置信息，但确实有些场景下绝对位置信息更有帮助，于是它将整个模型分为两部分来理解。以Base版的MLM预训练模型为例，它一共有13层，前11层只是用相对位置编码，这部分称为Encoder，后面2层加入绝对位置信息，这部分它称之为Decoder，还弄了个简称EMD（Enhanced Mask Decoder）；至于下游任务的微调截断，则是使用前11层的Encoder加上1层的Decoder来进行。

SuperGLUE上的成绩肯定了DeBERTa的价值，但是 它论文的各种命名真的是让人觉得极度不适 ，比如它自称的“Encoder”、“Decoder”就很容易让人误解这是一个Seq2Seq模型，比如EMD这个简称也跟Earth Mover's Distance重名。虽然有时候重名是不可避免的，但它重的名都是ML界大家都比较熟悉的对象，相当容易引起误解，真不知道作者是怎么想的...

## 其他位置编码

绝对位置编码和相对位置编码虽然花样百出，但仍然算是经典范围内，从上述介绍中我们依然可以体会到满满的套路感。除此之外，还有一些并不按照常规套路出牌，它们同样也表达了位置编码。

### CNN式

尽管经典的将CNN用于NLP的工作 [《Convolutional Sequence to Sequence Learning》](https://papers.cool/arxiv/1705.03122) 往里边加入了位置编码，但我们知道一般的CNN模型尤其是图像中的CNN模型，都是没有另外加位置编码的，那CNN模型究竟是怎么捕捉位置信息的呢？

如果让笔者来回答，那么答案可能是卷积核的各项异性导致了它能分辨出不同方向的相对位置。不过ICLR 2020的论文 [《How Much Position Information Do Convolutional Neural Networks Encode?》](https://papers.cool/arxiv/2001.08248) 给出了一个可能让人比较意外的答案： CNN模型的位置信息，是Zero Padding泄漏的！

我们知道，为了使得卷积编码过程中的feature保持一定的大小，我们通常会对输入padding一定的0，而这篇论文显示该操作导致模型有能力识别位置信息。也就是说，卷积核的各向异性固然重要，但是最根本的是zero padding的存在，那么可以想象，实际上提取的是当前位置与padding的边界的相对距离。

不过，这个能力依赖于CNN的局部性，像Attention这种全局的无先验结构并不适用，如果只关心Transformer位置编码方案的读者，这就权当是扩展一下视野吧。

### 复数式

复数式位置编码可谓是 最特立独行 的一种位置编码方案了，它来自ICLR 2020的论文 [《Encoding word order in complex embeddings》](https://papers.cool/arxiv/1912.12333) 。论文的主要思想是结合复数的性质以及一些基本原理，推导出了它的位置编码形式（Complex Order）为：  
$$
\begin{equation}\left[r_{j, 1} e^{\text{i}\left(\omega_{j, 1} k+\theta_{j, 1}\right)}, \ldots, r_{j, 2} e^{\text{i}\left(\omega_{j, 2} k+\theta_{j, 2}\right)}, \cdots, r_{j, d} e^{\text{i}\left(\omega_{j, d} k+\theta_{j, d}\right)}\right]\label{eq:complex}\end{equation}
$$
  
这里的 $i$ 是虚数单位， $j$ 代表某个词， $k$ 代表该词所在的位置，而  
$$
(13)rj=[rj,1,rj,2,⋯,rj,d]ωj=[ωj,1,ωj,2,⋯,ωj,d]θj=[θj,1,θj,2,⋯,θj,d]
$$
  
代表词 $j$ 的三组词向量。你没看错，它确实假设每个词有三组跟位置无关的词向量了（当然可以按照某种形式进行参数共享，使得它退化为两组甚至一组），然后跟位置 $k$ 相关的词向量就按照上述公式运算。

你以为引入多组词向量就是它最特立独行的地方了？并不是！我们看到式 $\eqref{eq:complex}$ 还是复数形式，你猜它接下来怎么着？将它实数化？非也，它是将它直接用于 复数模型 ！也就是说，它走的是一条复数模型路线，不仅仅输入的Embedding层是复数的，里边的每一层Transformer都是复数的，它还实现和对比了复数版的Fasttext、LSTM、CNN等模型！这篇文章的一作是Benyou Wang，可以搜到他的相关工作基本上都是围绕着复数模型展开的，可谓复数模型的铁杆粉了～

### 融合式

无偶独有，利用复数的形式，笔者其实也构思了一种比较巧的位置编码，它可以 将绝对位置编码与相对位置编码融于一体 ，分享在此，有兴趣的读者欢迎一起交流研究。

简单起见，我们先假设 $\boldsymbol{q}_m,\boldsymbol{k}_n$ 是所在位置分别为 $m,n$ 的二维行向量，既然是二维，那么我们可以将它当作复数来运算。我们知道，Attention关键之处在于向量的内积，用复数表示为  
$$
(14)⟨qm,kn⟩=Re[qmkn∗]
$$
  
其中 $∗$ 是共轭复数，右端的乘法是普通的复数乘法， $Re[]$ 表示取结果的实部。上式也就是说：

> 两个二维向量的内积，等于把它们当复数看时，一个复数与另一个复数的共轭的乘积实部。

如果我们将 $\boldsymbol{q}_m,\boldsymbol{k}_n$ 分别乘以 $eimθ,einθ$ 变成 $qmeimθ,kneinθ$ ，那么就相当于给它们配上了绝对位置编码了（因为显式地依赖绝对位置 $m,n$ ），然后放到内积里边，我们有  
$$
(15)⟨qmeimθ,kneinθ⟩=Re[(qmeimθ)(kneinθ)∗]=Re[qmkn∗ei(m−n)θ]
$$
  
相当有意思的是，内积只依赖于相对位置 $m−n$ ！这就巧妙地将绝对位置与相对位置融合在一起了。

注意，我们没有像Complex Order那么“疯狂”，上述运算本质上还是在实数范畴内的，只不过我们是借助复数来完成了某些推导而已。由上述结果可知，对于位置为 $n$ 的二维实数向量 $[x,y]$ ，我们当它复数来运算，乘以 $einθ$ ，得到恒等式  
$$
(16)(x+yi)einθ=(xcos⁡nθ−ysin⁡nθ)+i(xsin⁡nθ+ycos⁡nθ)
$$
  
这也就是意味着，通过  
$$
(17)(xy)→(xcos⁡nθ−ysin⁡nθxsin⁡nθ+ycos⁡nθ)=(xy)cos⁡nθ+(−yx)sin⁡nθ
$$
  
来赋予 $[x,y]$ 绝对位置信息，那么在Attention运算的时候也等价于相对位置编码。如果是多于二维的向量，可以考虑每两维为一组进行同样的运算，每一组的 $θ$ 可以不一样。

这样一来，我们得到了一种融绝对位置与相对位置于一体的位置编码方案，从形式上看它有点像乘性的绝对位置编码，通过在 $\boldsymbol{q},\boldsymbol{k}$ 中施行该位置编码，那么效果就等价于相对位置编码，而如果还需要显式的绝对位置信息，则可以同时在 $v$ 上也施行这种位置编码。总的来说，我们通过绝对位置的操作，可以达到绝对位置的效果，也能达到相对位置的效果，初步实验显示它是可以work的，但还没有充分验证，欢迎大家尝试交流。

## 文章内容小结

本文汇总了一些位置编码的工作，大体分为绝对式、相对式、非套路式三种，从中我们可以看到各种神奇的操作。最后，笔者分享了自己构思的一种融合绝对位置与相对位置的编码方案，供有兴趣的读者参考。

***转载到请包括本文地址：** [https://kexue.fm/archives/8130](https://kexue.fm/archives/8130 "让研究人员绞尽脑汁的Transformer位置编码")*

***更详细的转载事宜请参考：*** [《科学空间FAQ》](https://kexue.fm/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8 "《科学空间FAQ》")

**如果您还有什么疑惑或建议，欢迎在下方评论区继续讨论。**

**如果您觉得本文还不错，欢迎 [分享](https://kexue.fm/archives/#share) / [打赏](https://kexue.fm/archives/#pay) 本文。打赏并非要从中获得收益，而是希望知道科学空间获得了多少读者的真心关注。当然，如果你无视它，也不会影响你的阅读。再次表示欢迎和感谢！**

**如果您需要引用本文，请参考：**

苏剑林. (Feb. 03, 2021). 《让研究人员绞尽脑汁的Transformer位置编码 》\[Blog post\]. Retrieved from [https://kexue.fm/archives/8130](https://kexue.fm/archives/8130)

@online{kexuefm-8130,  
title={让研究人员绞尽脑汁的Transformer位置编码},  
author={苏剑林},  
year={2021},  
month={Feb},  
url={\\url{https://kexue.fm/archives/8130}},  
}

< [Seq2Seq重复解码现象的理论分析尝试](https://kexue.fm/archives/8128 "Seq2Seq重复解码现象的理论分析尝试") | [一个二值化词向量模型，是怎么跟果蝇搭上关系的？](https://kexue.fm/archives/8159 "一个二值化词向量模型，是怎么跟果蝇搭上关系的？") \>