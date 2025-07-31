# The Path to Upgrading Transformers: 2. Rotary Position Embedding, a Combination of Strengths - Scientific Spaces

## Author: Su Jianlin
## Published: 2021-03-23
## Summary: This article introduces Rotary Position Embedding (RoPE), a novel method for encoding positional information in Transformers. RoPE uses a rotational transformation on query and key vectors to elegantly achieve the benefits of relative position encoding through an absolute position mechanism. This approach improves performance on long-text tasks and is uniquely compatible with linear attention architectures, offering a more robust and interpretable alternative to traditional sinusoidal embeddings.

---

Title: The Path to Upgrading Transformers: 2. Rotary Position Embedding, a Combination of Strengths - Scientific Spaces
Source: https://kexue.fm/archives/8265
Author: Su Jianlin
Published: 2021-03-23
Created: 2025-07-30
Description: This article introduces Rotary Position Embedding (RoPE), a novel method for encoding positional information in Transformers. RoPE uses a rotational transformation on query and key vectors to elegantly achieve the benefits of relative position encoding through an absolute position mechanism. This approach improves performance on long-text tasks and is uniquely compatible with linear attention architectures, offering a more robust and interpretable alternative to traditional sinusoidal embeddings.

Here is the translation of the provided article:

---
**Title:** The Path to Upgrading Transformers: 2. Rotary Position Embedding, a Combination of Strengths - Scientific Spaces

**Source:** [https://kexue.fm/archives/8265](https://kexue.fm/archives/8265)

**Author:**

**Published:**

**Created:** 2025-07-30

**Description:** In the previous article, we conducted a relatively detailed derivation and analysis of the original Sinusoidal position encoding. The general feeling is that Sinusoidal position encoding is an "absolute position encoding that wants to become a relative position encoding." Generally speaking, absolute position encoding has the advantages of simple implementation and fast computation, while relative position encoding directly reflects the relative position signal, which is consistent with our intuitive understanding and often performs better in practice. Thus, if relative position encoding can be achieved through the method of absolute position encoding, it would be a case of "gathering the strengths of all" and "having one's cake and eating it too." Sinusoidal position encoding vaguely achieves this, but not well enough.

**Tags:**
  - clippings

---
**23 Mar**

In the previous article, we conducted a relatively detailed derivation and analysis of the original Sinusoidal position encoding. The general feeling is that Sinusoidal position encoding is an "absolute position encoding that wants to become a relative position encoding." Generally speaking, absolute position encoding has the advantages of simple implementation and fast computation, while relative position encoding directly reflects the relative position signal, which is consistent with our intuitive understanding and often performs better in practice. Thus, if relative position encoding can be achieved through the method of absolute position encoding, it would be a case of "gathering the strengths of all" and "having one's cake and eating it too." Sinusoidal position encoding vaguely achieves this, but not well enough.

This article will introduce our self-developed Rotary Transformer (RoFormer) model. Its main modification is the application of the "Rotary Position Embedding (RoPE)" conceived by the author, which is a design that, in conjunction with the Attention mechanism, can achieve "relative position encoding through the method of absolute position encoding." And precisely because of this design, it is currently the only relative position encoding that can be used for linear Attention.

> **RoFormer： [https://github.com/ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer)**

## Basic Idea

In the previous article [《The Position Encoding of Transformers that Racks Researchers' Brains》](https://kexue.fm/archives/8130), we briefly introduced RoPE, at that time calling it "fusion style." This article will introduce its origin and properties in more detail. In RoPE, our starting point is to "achieve relative position encoding through the method of absolute position encoding." Doing so has both theoretical elegance and practical utility, for example, its ability to be extended to linear Attention is mainly due to this point.

To achieve this goal, we assume that the following operations are used to add absolute position information to $\boldsymbol{q},\boldsymbol{k}$:
$$(1)q~m=f(q,m),k~n=f(k,n)$$
That is to say, we design operations $f(⋅,m),f(⋅,n)$ for $q,k$ respectively, so that after this operation, $q~m,k~n$ carry the absolute position information of positions $m,n$. The core operation of Attention is the inner product, so we hope the result of the inner product carries relative position information. Therefore, we assume the existence of an identity relation:
$$(2)⟨f(q,m),f(k,n)⟩=g(q,k,m−n)$$
So we need to find a (preferably simple) solution to this identity. The solution process also requires some initial conditions. Obviously, we can reasonably set $f(q,0)=q$ and $f(k,0)=k$.

## Solution Process

Following the same line of thought as the previous article, we first consider the two-dimensional case and use complex numbers to solve it. In complex numbers, we have $\langle\boldsymbol{q},\boldsymbol{k}\rangle=\text{Re}[\boldsymbol{q}\boldsymbol{k}^*]$, where $Re[]$ represents the real part of a complex number, so we have
$$(3)Re[f(q,m)f∗(k,n)]=g(q,k,m−n)$$
For simplicity, we assume the existence of a complex number $g(q,k,m−n)$ such that $f(q,m)f∗(k,n)=g(q,k,m−n)$. Then we use the exponential form of complex numbers, setting
$$(4)f(q,m)=Rf(q,m)eiΘf(q,m)f(k,n)=Rf(k,n)eiΘf(k,n)g(q,k,m−n)=Rg(q,k,m−n)eiΘg(q,k,m−n)$$
Substituting into the equation, we get the system of equations
$$(5)Rf(q,m)Rf(k,n)=Rg(q,k,m−n)Θf(q,m)−Θf(k,n)=Θg(q,k,m−n)$$
For the first equation, substituting $m=n$ gives
$$(6)Rf(q,m)Rf(k,m)=Rg(q,k,0)=Rf(q,0)Rf(k,0)=‖q‖‖k‖$$
The last equal sign comes from the initial conditions $f(q,0)=q$ and $f(k,0)=k$. So now we can simply set $Rf(q,m)=‖q‖,Rf(k,m)=‖k‖$, meaning it does not depend on $m$. As for the second equation, also substituting $m=n$ gives
$$(7)Θf(q,m)−Θf(k,m)=Θg(q,k,0)=Θf(q,0)−Θf(k,0)=Θ(q)−Θ(k)$$
Here $Θ(q),Θ(k)$ are the arguments of $q,k$ themselves, and the last equal sign also comes from the initial conditions. From the above equation, we get $Θf(q,m)−Θ(q)=Θf(k,m)−Θ(k)$, so $Θf(q,m)−Θ(q)$ should be a function that only depends on $m$ and is independent of $q$, denoted as $φ(m)$, i.e., $Θf(q,m)=Θ(q)+φ(m)$. Next, substituting $n=m−1$ and rearranging, we get
$$(8)φ(m)−φ(m−1)=Θg(q,k,1)+Θ(k)−Θ(q)$$
That is, ${φ(m)}$ is an arithmetic sequence. Let the right side be $θ$, then we solve for $φ(m)=mθ$.

## Encoding Form

In summary, we obtain RoPE in two dimensions represented by complex numbers:
$$
\begin{equation} 
\boldsymbol{f}(\boldsymbol{q}, m) = R_f (\boldsymbol{q}, m)e^{\text{i}\Theta_f(\boldsymbol{q}, m)} 
= \Vert q\Vert e^{\text{i}(\Theta(\boldsymbol{q}) + m\theta)} = \boldsymbol{q} e^{\text{i}m\theta}\end{equation}
$$According to the geometric meaning of complex number multiplication, this transformation actually corresponds to the rotation of a vector, so we call it "Rotary Position Embedding." It can also be written in matrix form:$$
(10)f(q,m)=\begin{pmatrix} \cos{m\theta} & -\sin{m\theta} \\ \sin{m\theta} & \cos{m\theta} \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}
$$Since the inner product satisfies linear superposition, any even-dimensional RoPE can be expressed as a concatenation of two-dimensional cases, i.e.,$$
(11)\underbrace{\begin{pmatrix} \cos{m\theta_0} & -\sin{m\theta_0} & 0 & 0 & \cdots & 0 & 0 \\ \sin{m\theta_0} & \cos{m\theta_0} & 0 & 0 & \cdots & 0 & 0 \\ 0 & 0 & \cos{m\theta_1} & -\sin{m\theta_1} & \cdots & 0 & 0 \\ 0 & 0 & \sin{m\theta_1} & \cos{m\theta_1} & \cdots & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \cdots & \cos{m\theta_{d/2-1}} & -\sin{m\theta_{d/2-1}} \\ 0 & 0 & 0 & 0 & \cdots & \sin{m\theta_{d/2-1}} & \cos{m\theta_{d/2-1}} \end{pmatrix}}_{R_m} \begin{pmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} \end{pmatrix}
$$That is, by multiplying the vector $q$ at position $m$ by the matrix $R_m$ and the vector $k$ at position $n$ by the matrix $R_n$, and then performing Attention on the transformed $Q,K$ sequences, the Attention will automatically contain relative position information, because the following identity holds:$$
(12)(R_m\boldsymbol{q})^\top(R_n\boldsymbol{k}) = \boldsymbol{q}^\top R_m^\top R_n \boldsymbol{k} = \boldsymbol{q}^\top R_{n-m}\boldsymbol{k}
$$
It is worth noting that $R_m$ is an orthogonal matrix, which does not change the norm of the vector, so generally speaking, it will not change the stability of the original model.

Due to the sparsity of $\boldsymbol{\mathcal{R}}_m$, implementing it directly with matrix multiplication would be a waste of computational resources. It is recommended to implement RoPE in the following way:
$$(13)\begin{pmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} \end{pmatrix} \otimes \begin{pmatrix} \cos{m\theta_0} \\ \cos{m\theta_0} \\ \cos{m\theta_1} \\ \cos{m\theta_1} \\ \vdots \\ \cos{m\theta_{d/2-1}} \\ \cos{m\theta_{d/2-1}} \end{pmatrix} + \begin{pmatrix} -q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d-1} \\ q_{d-2} \end{pmatrix} \otimes \begin{pmatrix} \sin{m\theta_0} \\ \sin{m\theta_0} \\ \sin{m\theta_1} \\ \sin{m\theta_1} \\ \vdots \\ \sin{m\theta_{d/2-1}} \\ \sin{m\theta_{d/2-1}} \end{pmatrix}$$
where $⊗$ is element-wise multiplication, i.e., the $*$ operation in frameworks like NumPy and TensorFlow. From this implementation, it can also be seen that RoPE can be regarded as a variant of multiplicative position encoding.

## Remote Attenuation

It can be seen that the form of RoPE is somewhat similar to Sinusoidal position encoding, except that Sinusoidal position encoding is additive, while RoPE can be considered multiplicative. In the choice of $\theta_i$, we have also followed the scheme of Sinusoidal position encoding, i.e., $θi=10000−2i/d$, which can bring a certain degree of remote attenuation.

The specific proof is as follows: After pairing up $\boldsymbol{q},\boldsymbol{k}$, their inner product after adding RoPE can be expressed by complex multiplication as
$$(14)(R_m\boldsymbol{q})^\top(R_n\boldsymbol{k}) = \text{Re}\left[\sum_{i=0}^{d/2-1} \boldsymbol{q}[2i:2i+1]\boldsymbol{k}[2i:2i+1]^* e^{i(m-n)\theta_i}\right]$$
Let $h_i = \boldsymbol{q}[2i:2i+1]\boldsymbol{k}[2i:2i+1]^*, S_j = \sum_{i=0}^{j-1} e^{i(m-n)\theta_i}$, and we agree that $h_{d/2} = 0, S_0 = 0$. Then by [Abel's transformation (summation by parts)](https://en.wikipedia.org/wiki/Summation_by_parts), we can get:
$$(15)\sum_{i=0}^{d/2-1} \boldsymbol{q}[2i:2i+1]\boldsymbol{k}[2i:2i+1]^* e^{i(m-n)\theta_i} = \sum_{i=0}^{d/2-1} h_i(S_{i+1}-S_i) = -\sum_{i=0}^{d/2-1} S_{i+1}(h_{i+1}-h_i)$$
So,
$$(16)\left|\sum_{i=0}^{d/2-1} \boldsymbol{q}[2i:2i+1]\boldsymbol{k}[2i:2i+1]^* e^{i(m-n)\theta_i}\right| = \left|\sum_{i=0}^{d/2-1} S_{i+1}(h_{i+1}-h_i)\right| \le \sum_{i=0}^{d/2-1} |S_{i+1}||h_{i+1}-h_i| \le (\max_i|h_{i+1}-h_i|)\sum_{i=0}^{d/2-1}|S_{i+1}|$$
Therefore, we can examine the change of $\frac{1}{d/2}\sum_{i=1}^{d/2}|S_i|$ with the relative distance as a manifestation of the attenuation property. The Mathematica code is as follows:

```
d = 128;
\[Theta][t_] = 10000^(-2*t/d);
f[m_] = Sum[
    Norm[Sum[Exp[I*m*\[Theta][i]], {i, 0, j}]], {j, 0, d/2 - 1}]/(d/2);
Plot[f[m], {m, 0, 256}, AxesLabel -> {"Relative Distance", "Relative Magnitude"}]
```

The result is as follows:

[![Remote Attenuation of RoPE (d=128)](https://kexue.fm/usr/uploads/2021/03/1347893165.png)](https://kexue.fm/usr/uploads/2021/03/1347893165.png "Click to view original image")

From the figure, we can see that as the relative distance increases, the inner product result shows a tendency to decay. Therefore, choosing $\theta_i = 10000^{-2i/d}$ indeed brings a certain degree of remote attenuation. Of course, as mentioned in the previous article, this is not the only choice that can bring remote attenuation; almost any smooth and monotonic function will do. Here we just follow the existing choice. The author also tried to initialize with $θi=10000−2i/d$ and treat $θi$ as a trainable parameter, but after training for a period of time, it was found that $θi$ was not significantly updated, so it was simply fixed at $θi=10000−2i/d$.

## Linear Scenarios

Finally, we point out that RoPE is currently the only relative position encoding that can be used for linear Attention. This is because other relative position encodings operate directly on the Attention matrix, but linear Attention does not calculate the Attention matrix in advance, so there is no way to operate on the Attention matrix. Therefore, other schemes cannot be applied to linear Attention. For RoPE, it achieves relative position encoding through the method of absolute position encoding and does not need to operate on the Attention matrix, thus having the possibility of being applied to linear Attention.

Regarding the introduction to linear Attention, we will not repeat it here. Readers in need can refer to [《Exploration of Linear Attention: Must Attention Have a Softmax?》](https://kexue.fm/archives/7546). The common form of linear Attention is:
$$\begin{equation}Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_i = \frac{\sum\limits_{j=1}^n \text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)\boldsymbol{v}_j}{\sum\limits_{j=1}^n \text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)} = \frac{\sum\limits_{j=1}^n \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)\boldsymbol{v}_j}{\sum\limits_{j=1}^n \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)}\end{equation}$$
where $ϕ,φ$ are activation functions with a non-negative range. As you can see, linear Attention is also based on the inner product, so a natural idea is to insert RoPE into the inner product:
$$(18)\frac{\sum_{j=1}^n [R_i\phi(q_i)]^\top[R_j\phi(k_j)]v_j}{\sum_{j=1}^n [R_i\phi(q_i)]^\top[R_j\phi(k_j)]}$$
However, the problem with this is that the inner product $[R_i\phi(q_i)]^\top[R_j\phi(k_j)]$ may be negative, so it is no longer a conventional probability attention, and there is a risk of the denominator being 0, which may bring optimization instability. Considering that $R_i,R_j$ are orthogonal matrices and do not change the norm of the vector, we can abandon the conventional probability normalization requirement and use the following operation as a new linear Attention:
$$(19)\frac{\sum_{j=1}^n [R_i\phi(q_i)]^\top[R_j\phi(k_j)]v_j}{\sum_{j=1}^n \phi(q_i)^\top\phi(k_j)}$$
That is, RoPE is only inserted into the numerator, while the denominator remains unchanged. Such attention is no longer probability-based (the attention matrix no longer satisfies non-negative normalization), but in a sense, it is also a normalization scheme, and there is no evidence that non-probabilistic attention is not good (for example, [Nyströmformer](https://kexue.fm/archives/8180) can also be considered to construct attention without strictly following a probability distribution). Therefore, we will use it as one of the candidate schemes for experiments, and our preliminary experimental results show that such linear Attention is also effective.

In addition, the author in [《Exploration of Linear Attention: Must Attention Have a Softmax?》](https://kexue.fm/archives/7546) also proposed another linear Attention scheme: $\text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j) = 1 + \left( \frac{\boldsymbol{q}_i}{\Vert \boldsymbol{q}_i\Vert}\right)^{\top}\left(\frac{\boldsymbol{k}_j}{\Vert \boldsymbol{k}_j\Vert}\right)$. It does not depend on the non-negativity of the value range, and RoPE does not change the norm, so RoPE can be directly applied to this type of linear Attention without changing its probabilistic meaning.

## Model Open Sourced

We have completed the training of the first version of the RoFormer model and have open-sourced it on Github:

> **RoFormer： [https://github.com/ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer)**

Simply put, RoFormer is a [WoBERT](https://github.com/ZhuiyiTechnology/WoBERT) model with absolute position encoding replaced by RoPE. The comparison of its structure with other models is as follows:
$$
\begin{array}{c|cccc} 
\hline 
& \text{BERT} & \text{WoBERT} & \text{NEZHA} & \text{RoFormer} \\ 
\hline 
\text{token unit} & \text{character} & \text{word} & \text{character} & \text{word} & \\ 
\text{position encoding} & \text{absolute position} & \text{absolute position} & \text{classic relative position} & \text{RoPE}\\ 
\hline 
\end{array}
$$In pre-training, we based it on WoBERT Plus and used a method of alternating training with multiple lengths and batch sizes to allow the model to adapt to different training scenarios in advance:$$
\begin{array}{cccccc}
\text{maxlen} & \text{batch size} & \text{training steps} & \text{final loss} & \text{final acc} \\
15 & 12256 & 200,000 & 1.73 & 65.0\% \\
2 & 1536 & 256 & 12,500 & 1.61 & 66.8\% \\
3 & 256 & 256 & 120,000 & 1.75 & 64.6\% \\
4 & 128 & 512 & 80,000 & 1.83 & 63.4\% \\
5 & 1536 & 256 & 10,000 & 1.58 & 67.4\% \\
6 & 512 & 512 & 30,000 & 1.66 & 66.2\%
\end{array}
$$From the table, it can also be seen that increasing the sequence length, the pre-training accuracy has actually improved, which reflects the effectiveness of RoFormer in processing long text semantics and also reflects that RoPE has good extrapolation ability. In short text tasks, the performance of RoFormer is similar to that of WoBERT. The main feature of RoFormer is that it can directly handle text of any length. Below are our experimental results on the [CAIL2019-SCM](https://papers.cool/arxiv/1911.08962) task:$$
\begin{array}{ccc}
& \text{Validation Set} & \text{Test Set} \\
\text{BERT-512} & 64.13\% & 67.77\% \\
\text{WoBERT-512} & 64.07\% & 68.10\% \\
\text{RoFormer-512} & 64.13\% & 68.29\% \\
\text{RoFormer-1024} & 66.07\% & 69.79\%
\end{array}
$$
Here, the parameter after - is the maxlen truncated during fine-tuning. It can be seen that RoFormer can indeed handle long text semantics well. As for the hardware requirements, on a card with 24G of VRAM, running maxlen=1024, the batch_size can be more than 8. Currently, in Chinese tasks, the author has only found this task to be suitable for testing long text capabilities, so only this task has been tested for long text. Readers are welcome to test or recommend other evaluation tasks.

Of course, although in theory RoFormer can handle sequences of any length, RoFormer still has a quadratic complexity. We are also training a RoFormer model based on linear Attention, and the experiment will be open-sourced after completion. Please look forward to it.

(Note: RoPE and RoFormer have been compiled into a paper [《RoFormer: Enhanced Transformer with Rotary Position Embedding》](https://papers.cool/arxiv/2104.09864) and submitted to Arxiv. Welcome to use and cite it haha~)

## Article Summary

This article introduces our self-developed Rotary Position Embedding (RoPE) and the corresponding pre-trained model RoFormer. From a theoretical point of view, RoPE has some similarities with Sinusoidal position encoding, but RoPE does not rely on Taylor expansion, making it more rigorous and interpretable. From the results of the pre-trained model RoFormer, RoPE has good extrapolation ability and shows better ability to handle long texts when applied to Transformers. In addition, RoPE is currently the only relative position encoding that can be used for linear Attention.

***To reprint, please include the address of this article:** [https://kexue.fm/archives/8265](https://kexue.fm/archives/8265 "The Path to Upgrading Transformers: 2. Rotary Position Embedding, a Combination of Strengths")*

***For more detailed reprinting matters, please refer to:*** [《Scientific Spaces FAQ》](https://kexue.fm/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8 "《Scientific Spaces FAQ》")

**If you have any doubts or suggestions, please continue the discussion in the comments section below.**

**If you think this article is not bad, welcome to [share](https://kexue.fm/archives/#share) / [donate](https://kexue.fm/archives/#pay) to this article. The donation is not to make a profit from it, but to know how many readers' sincere attention Scientific Spaces has received. Of course, if you ignore it, it will not affect your reading. Welcome and thank you again!**

**If you need to cite this article, please refer to:**

Su, Jianlin. (Mar. 23, 2021). 《The Path to Upgrading Transformers: 2. Rotary Position Embedding, a Combination of Strengths》\[Blog post\]. Retrieved from [https://kexue.fm/archives/8265](https://kexue.fm/archives/8265)

@online{kexuefm-8265,  
title={The Path to Upgrading Transformers: 2. Rotary Position Embedding, a Combination of Strengths},  
author={Su, Jianlin},  
year={2021},  
month={Mar},  
url={\url{https://kexue.fm/archives/8265}},  
}

< [The success of WGAN may have little to do with the Wasserstein distance](https://kexue.fm/archives/8244 "The success of WGAN may have little to do with the Wasserstein distance") | [P-tuning: Automatically constructing templates to unleash the potential of language models](https://kexue.fm/archives/8295 "P-tuning: Automatically constructing templates to unleash the potential of language models") \>

---

Original:

---
title: "Transformer升级之路：2、博采众长的旋转式位置编码 - 科学空间|Scientific Spaces"
source: "https://kexue.fm/archives/8265"
author:
published:
created: 2025-07-31
description: "上一篇文章中，我们对原始的Sinusoidal位置编码做了较为详细的推导和理解，总的感觉是Sinusoidal位置编码是一种“想要成为相对位置编码的绝对位置编码”。一般来说，绝对位置编码具有实现..."
tags:
  - "clippings"
---
23 Mar

上一篇文章中，我们对原始的Sinusoidal位置编码做了较为详细的推导和理解，总的感觉是Sinusoidal位置编码是一种“想要成为相对位置编码的绝对位置编码”。一般来说，绝对位置编码具有实现简单、计算速度快等优点，而相对位置编码则直接地体现了相对位置信号，跟我们的直观理解吻合，实际性能往往也更好。由此可见，如果可以通过绝对位置编码的方式实现相对位置编码，那么就是“集各家之所长”、“鱼与熊掌兼得”了。Sinusoidal位置编码隐约做到了这一点，但并不够好。

本文将会介绍我们自研的 Rotary Transformer（RoFormer） 模型，它的主要改动是应用了笔者构思的“ 旋转式位置编码（Rotary Position Embedding，RoPE） ”，这是一种配合Attention机制能达到“绝对位置编码的方式实现相对位置编码”的设计。而也正因为这种设计，它还是目前唯一一种可用于线性Attention的相对位置编码。

> **RoFormer： [https://github.com/ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer)**

## 基本思路

在之前的文章 [《让研究人员绞尽脑汁的Transformer位置编码》](https://kexue.fm/archives/8130) 中我们就简要介绍过RoPE，当时称之为“融合式”，本文则更加详细地介绍它的来源与性质。在RoPE中，我们的出发点就是“通过绝对位置编码的方式实现相对位置编码”，这样做既有理论上的优雅之处，也有实践上的实用之处，比如它可以拓展到线性Attention中就是主要因为这一点。

为了达到这个目的，我们假设通过下述运算来给 $\boldsymbol{q},\boldsymbol{k}$ 添加绝对位置信息：  
$$
(1)q~m=f(q,m),k~n=f(k,n)
$$
  
也就是说，我们分别为 $q,k$ 设计操作 $f(⋅,m),f(⋅,n)$ ，使得经过该操作后， $q~m,k~n$ 就带有了位置 $m,n$ 的绝对位置信息。Attention的核心运算是内积，所以我们希望的内积的结果带有相对位置信息，因此假设存在恒等关系：  
$$
(2)⟨f(q,m),f(k,n)⟩=g(q,k,m−n)
$$
  
所以我们要求出该恒等式的一个（尽可能简单的）解。求解过程还需要一些初始条件，显然我们可以合理地设 $f(q,0)=q$ 和 $f(k,0)=k$ 。

## 求解过程

同上一篇思路一样，我们先考虑二维情形，然后借助复数来求解。在复数中有 $\langle\boldsymbol{q},\boldsymbol{k}\rangle=\text{Re}[\boldsymbol{q}\boldsymbol{k}^*]$ ， $Re[]$ 代表复数的实部，所以我们有  
$$
(3)Re[f(q,m)f∗(k,n)]=g(q,k,m−n)
$$
  
简单起见，我们假设存在复数 $g(q,k,m−n)$ ，使得 $f(q,m)f∗(k,n)=g(q,k,m−n)$ ，然后我们用复数的指数形式，设  
$$
(4)f(q,m)=Rf(q,m)eiΘf(q,m)f(k,n)=Rf(k,n)eiΘf(k,n)g(q,k,m−n)=Rg(q,k,m−n)eiΘg(q,k,m−n)
$$
  
那么代入方程后就得到方程组  
$$
(5)Rf(q,m)Rf(k,n)=Rg(q,k,m−n)Θf(q,m)−Θf(k,n)=Θg(q,k,m−n)
$$
  
对于第一个方程，代入 $m=n$ 得到  
$$
(6)Rf(q,m)Rf(k,m)=Rg(q,k,0)=Rf(q,0)Rf(k,0)=‖q‖‖k‖
$$
  
最后一个等号源于初始条件 $f(q,0)=q$ 和 $f(k,0)=k$ 。所以现在我们可以很简单地设 $Rf(q,m)=‖q‖,Rf(k,m)=‖k‖$ ，即它不依赖于 $m$ 。至于第二个方程，同样代入 $m=n$ 得到  
$$
(7)Θf(q,m)−Θf(k,m)=Θg(q,k,0)=Θf(q,0)−Θf(k,0)=Θ(q)−Θ(k)
$$
  
这里的 $Θ(q),Θ(k)$ 是 $q,k$ 本身的幅角，最后一个等号同样源于初始条件。根据上式得到 $Θf(q,m)−Θ(q)=Θf(k,m)−Θ(k)$ ，所以 $Θf(q,m)−Θ(q)$ 应该是一个只与 $m$ 相关、跟 $q$ 无关的函数，记为 $φ(m)$ ，即 $Θf(q,m)=Θ(q)+φ(m)$ 。接着代入 $n=m−1$ ，整理得到  
$$
(8)φ(m)−φ(m−1)=Θg(q,k,1)+Θ(k)−Θ(q)
$$
  
即 ${φ(m)}$ 是等差数列，设右端为 $θ$ ，那么就解得 $φ(m)=mθ$ 。

## 编码形式

综上，我们得到二维情况下用复数表示的RoPE：  
$$
\begin{equation} 
\boldsymbol{f}(\boldsymbol{q}, m) = R_f (\boldsymbol{q}, m)e^{\text{i}\Theta_f(\boldsymbol{q}, m)} 
= \Vert q\Vert e^{\text{i}(\Theta(\boldsymbol{q}) + m\theta)} = \boldsymbol{q} e^{\text{i}m\theta}\end{equation}
$$
  
根据复数乘法的几何意义，该变换实际上对应着向量的旋转，所以我们称之为“旋转式位置编码”，它还可以写成矩阵形式：  
$$
(10)f(q,m)=(cos⁡mθ−sin⁡mθsin⁡mθcos⁡mθ)(q0q1)
$$
  
由于内积满足线性叠加性，因此任意偶数维的RoPE，我们都可以表示为二维情形的拼接，即  
$$
(11)(cos⁡mθ0−sin⁡mθ000⋯00sin⁡mθ0cos⁡mθ000⋯0000cos⁡mθ1−sin⁡mθ1⋯0000sin⁡mθ1cos⁡mθ1⋯00⋮⋮⋮⋮⋱⋮⋮0000⋯cos⁡mθd/2−1−sin⁡mθd/2−10000⋯sin⁡mθd/2−1cos⁡mθd/2−1)⏟Rm(q0q1q2q3⋮qd−2qd−1)
$$
  
也就是说，给位置为 $m$ 的向量 $q$ 乘上矩阵 $Rm$ 、位置为 $n$ 的向量 $k$ 乘上矩阵 $Rn$ ，用变换后的 $Q,K$ 序列做Attention，那么Attention就自动包含相对位置信息了，因为成立恒等式：  
$$
(12)(Rmq)⊤(Rnk)=q⊤Rm⊤Rnk=q⊤Rn−mk
$$
  
值得指出的是， $Rm$ 是一个正交矩阵，它不会改变向量的模长，因此通常来说它不会改变原模型的稳定性。

由于 $\boldsymbol{\mathcal{R}}_m$ 的稀疏性，所以直接用矩阵乘法来实现会很浪费算力，推荐通过下述方式来实现RoPE：  
$$
(13)(q0q1q2q3⋮qd−2qd−1)⊗(cos⁡mθ0cos⁡mθ0cos⁡mθ1cos⁡mθ1⋮cos⁡mθd/2−1cos⁡mθd/2−1)+(−q1q0−q3q2⋮−qd−1qd−2)⊗(sin⁡mθ0sin⁡mθ0sin⁡mθ1sin⁡mθ1⋮sin⁡mθd/2−1sin⁡mθd/2−1)
$$
  
其中 $⊗$ 是逐位对应相乘，即Numpy、Tensorflow等计算框架中的 $∗$ 运算。从这个实现也可以看到，RoPE可以视为是乘性位置编码的变体。

## 远程衰减

可以看到，RoPE形式上和Sinusoidal位置编码有点相似，只不过Sinusoidal位置编码是加性的，而RoPE可以视为乘性的。在 $\theta_i$ 的选择上，我们同样沿用了Sinusoidal位置编码的方案，即 $θi=10000−2i/d$ ，它可以带来一定的远程衰减性。

具体证明如下：将 $\boldsymbol{q},\boldsymbol{k}$ 两两分组后，它们加上RoPE后的内积可以用复数乘法表示为  
$$
(14)(Rmq)⊤(Rnk)=Re[∑i=0d/2−1q[2i:2i+1]k[2i:2i+1]∗ei(m−n)θi]
$$
  
记 $hi=q[2i:2i+1]k[2i:2i+1]∗,Sj=∑i=0j−1ei(m−n)θi$ ，并约定 $hd/2=0,S0=0$ ，那么由 [Abel变换（分部求和法）](https://zh.wikipedia.org/wiki/%E5%88%86%E9%83%A8%E6%B1%82%E5%92%8C%E6%B3%95) 可以得到：  
$$
(15)∑i=0d/2−1q[2i:2i+1]k[2i:2i+1]∗ei(m−n)θi=∑i=0d/2−1hi(Si+1−Si)=−∑i=0d/2−1Si+1(hi+1−hi)
$$
  
所以  
$$
(16)|∑i=0d/2−1q[2i:2i+1]k[2i:2i+1]∗ei(m−n)θi|=|∑i=0d/2−1Si+1(hi+1−hi)|≤∑i=0d/2−1|Si+1||hi+1−hi|≤(maxi|hi+1−hi|)∑i=0d/2−1|Si+1|
$$
  
因此我们可以考察 $1d/2∑i=1d/2|Si|$ 随着相对距离的变化情况来作为衰减性的体现，Mathematica代码如下：

```
d = 128;
\[Theta][t_] = 10000^(-2*t/d);
f[m_] = Sum[
    Norm[Sum[Exp[I*m*\[Theta][i]], {i, 0, j}]], {j, 0, d/2 - 1}]/(d/2);
Plot[f[m], {m, 0, 256}, AxesLabel -> {相对距离, 相对大小}]
```

结果如下图：  

[![RoPE的远程衰减性（d=128）](https://kexue.fm/usr/uploads/2021/03/1347893165.png)](https://kexue.fm/usr/uploads/2021/03/1347893165.png "点击查看原图")

RoPE的远程衰减性（d=128）

从图中我们可以可以看到随着相对距离的变大，内积结果有衰减趋势的出现。因此，选择 $\theta_i = 10000^{-2i/d}$ ，确实能带来一定的远程衰减性。当然，同上一篇文章说的一样，能带来远程衰减性的不止这个选择，几乎任意的光滑单调函数都可以，这里只是沿用了已有的选择而已。笔者还试过以 $θi=10000−2i/d$ 为初始化，将 $θi$ 视为可训练参数，然后训练一段时间后发现 $θi$ 并没有显著更新，因此干脆就直接固定 $θi=10000−2i/d$ 了。

## 线性场景

最后，我们指出，RoPE是目前唯一一种可以用于线性Attention的相对位置编码。这是因为其他的相对位置编码，都是直接基于Attention矩阵进行操作的，但是线性Attention并没有事先算出Attention矩阵，因此也就不存在操作Attention矩阵的做法，所以其他的方案无法应用到线性Attention中。而对于RoPE来说，它是用绝对位置编码的方式来实现相对位置编码，不需要操作Attention矩阵，因此有了应用到线性Attention的可能性。

关于线性Attention的介绍，这里不再重复，有需要的读者请参考 [《线性Attention的探索：Attention必须有个Softmax吗？》](https://kexue.fm/archives/7546) 。线性Attention的常见形式是：  
$$
\begin{equation}Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_i = \frac{\sum\limits_{j=1}^n \text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)\boldsymbol{v}_j}{\sum\limits_{j=1}^n \text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)} = \frac{\sum\limits_{j=1}^n \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)\boldsymbol{v}_j}{\sum\limits_{j=1}^n \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)}\end{equation}
$$
  
其中 $ϕ,φ$ 是值域非负的激活函数。可以看到，线性Attention也是基于内积的，所以很自然的想法是可以将RoPE插入到内积中：  
$$
(18)∑j=1n[Riϕ(qi)]⊤[Rjφ(kj)]vj∑j=1n[Riϕ(qi)]⊤[Rjφ(kj)]
$$
  
但这样存在的问题是，内积 $[Riϕ(qi)]⊤[Rjφ(kj)]$ 可能为负数，因此它不再是常规的概率注意力，而且分母有为0的风险，可能会带来优化上的不稳定。考虑到 $Ri,Rj$ 都是正交矩阵，它不改变向量的模长，因此我们可以抛弃常规的概率归一化要求，使用如下运算作为一种新的线性Attention：  
$$
(19)∑j=1n[Riϕ(qi)]⊤[Rjφ(kj)]vj∑j=1nϕ(qi)⊤φ(kj)
$$
  
也就是说，RoPE只插入分子中，而分母则不改变，这样的注意力不再是基于概率的（注意力矩阵不再满足非负归一性），但它某种意义上来说也是一个归一化方案，而且也没有证据表明非概率式的注意力就不好（比如 [Nyströmformer](https://kexue.fm/archives/8180) 也算是没有严格依据概率分布的方式构建注意力），所以我们将它作为候选方案之一进行实验，而我们初步的实验结果显示这样的线性Attention也是有效的。

此外，笔者在 [《线性Attention的探索：Attention必须有个Softmax吗？》](https://kexue.fm/archives/7546) 中还提出过另外一种线性Attention方案： $\text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j) = 1 + \left( \frac{\boldsymbol{q}_i}{\Vert \boldsymbol{q}_i\Vert}\right)^{\top}\left(\frac{\boldsymbol{k}_j}{\Vert \boldsymbol{k}_j\Vert}\right)$ ，它不依赖于值域的非负性，而RoPE也不改变模长，因此RoPE可以直接应用于此类线性Attention，并且不改变它的概率意义。

## 模型开源

RoFormer的第一版模型，我们已经完成训练并开源到了Github中：

> **RoFormer： [https://github.com/ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer)**

简单来说，RoFormer是一个绝对位置编码替换为RoPE的 [WoBERT](https://github.com/ZhuiyiTechnology/WoBERT) 模型，它跟其他模型的结构对比如下：  
$$
\begin{array}{c|cccc} 
\hline 
& \text{BERT} & \text{WoBERT} & \text{NEZHA} & \text{RoFormer} \\ 
\hline 
\text{token单位} & \text{字} & \text{词} & \text{字} & \text{词} & \\ 
\text{位置编码} & \text{绝对位置} & \text{绝对位置} & \text{经典式相对位置} & \text{RoPE}\\ 
\hline 
\end{array}
$$
  
在预训练上，我们以WoBERT Plus为基础，采用了多个长度和batch size交替训练的方式，让模型能提前适应不同的训练场景：  
$$
maxlenbatch size训练步数最终loss最终acc151225620万1.7365.0%215362561.25万1.6166.8%325625612万1.7564.6%41285128万1.8363.4%515362561万1.5867.4%65125123万1.6666.2%
$$
  
从表格还可以看到，增大序列长度，预训练的准确率反而有所提升，这侧面体现了RoFormer长文本语义的处理效果，也体现了RoPE具有良好的外推能力。在短文本任务上，RoFormer与WoBERT的表现类似，RoFormer的主要特点是可以直接处理任意长的文本。下面是我们在 [CAIL2019-SCM](https://papers.cool/arxiv/1911.08962) 任务上的实验结果：  
$$
验证集测试集BERT-51264.13%67.77%WoBERT-51264.07%68.10%RoFormer-51264.13%68.29%RoFormer-102466.07%69.79%
$$
  
其中 $-$ 后面的参数是微调时截断的maxlen，可以看到RoFormer确实能较好地处理长文本语义，至于设备要求，在24G显存的卡上跑maxlen=1024，batch\_size可以跑到8以上。目前中文任务中笔者也就找到这个任务比较适合作为长文本能力的测试，所以长文本方面只测了这个任务，欢迎读者进行测试或推荐其他评测任务。

当然，尽管理论上RoFormer能处理任意长度的序列，但目前RoFormer还是具有平方复杂度的，我们也正在训练基于线性Attention的RoFormer模型，实验完成后也会开源放出，请大家期待。

（注：RoPE和RoFormer已经整理成文 [《RoFormer: Enhanced Transformer with Rotary Position Embedding》](https://papers.cool/arxiv/2104.09864) 提交到了Arxiv，欢迎使用和引用哈哈～）

## 文章小结

本文介绍了我们自研的旋转式位置编码RoPE以及对应的预训练模型RoFormer。从理论上来看，RoPE与Sinusoidal位置编码有些相通之处，但RoPE不依赖于泰勒展开，更具严谨性与可解释性；从预训练模型RoFormer的结果来看，RoPE具有良好的外推性，应用到Transformer中体现出较好的处理长文本的能力。此外，RoPE还是目前唯一一种可用于线性Attention的相对位置编码。

***转载到请包括本文地址：** [https://kexue.fm/archives/8265](https://kexue.fm/archives/8265 "Transformer升级之路：2、博采众长的旋转式位置编码")*

***更详细的转载事宜请参考：*** [《科学空间FAQ》](https://kexue.fm/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8 "《科学空间FAQ》")

**如果您还有什么疑惑或建议，欢迎在下方评论区继续讨论。**

**如果您觉得本文还不错，欢迎 [分享](https://kexue.fm/archives/#share) / [打赏](https://kexue.fm/archives/#pay) 本文。打赏并非要从中获得收益，而是希望知道科学空间获得了多少读者的真心关注。当然，如果你无视它，也不会影响你的阅读。再次表示欢迎和感谢！**

**如果您需要引用本文，请参考：**

苏剑林. (Mar. 23, 2021). 《Transformer升级之路：2、博采众长的旋转式位置编码 》\[Blog post\]. Retrieved from [https://kexue.fm/archives/8265](https://kexue.fm/archives/8265)

@online{kexuefm-8265,  
title={Transformer升级之路：2、博采众长的旋转式位置编码},  
author={苏剑林},  
year={2021},  
month={Mar},  
url={\\url{https://kexue.fm/archives/8265}},  
}

< [WGAN的成功，可能跟Wasserstein距离没啥关系](https://kexue.fm/archives/8244 "WGAN的成功，可能跟Wasserstein距离没啥关系") | [P-tuning：自动构建模版，释放语言模型潜能](https://kexue.fm/archives/8295 "P-tuning：自动构建模版，释放语言模型潜能") \>