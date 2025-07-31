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