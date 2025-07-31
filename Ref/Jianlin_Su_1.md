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