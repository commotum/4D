---
title: "Thread by @jeremyphoward"
source: "https://x.com/jeremyphoward/status/1881000354923544757"
author:
  - "[[@jeremyphoward]]"
published: 2025-01-17
created: 2026-02-03
description: "licklider escher faraday"
tags:
  - "clippings"
---
**Jeremy Howard** @jeremyphoward 2025-01-17

Folks seem to rediscover this every couple of years.

As I’ve been saying for many years, you have to track token accuracy, not loss/perplexity, otherwise you’ll wrongly think the validation loss going up is a bad thing.

> 2025-01-17
> 
> Wtf is up with deep learning???
> 
> ![Image](https://pbs.twimg.com/media/Ghg9yb6XkAAOLWh?format=png&name=large)

---

**Jeremy Howard** @jeremyphoward [2025-01-19](https://x.com/jeremyphoward/status/1881000620305621100)

Remember, loss is a differentiable \*proxy\* for what you actually care about.

Token accuracy is much closer to what you should \*actually\* care about.

---

**Jeremy Howard** @jeremyphoward [2025-01-19](https://x.com/jeremyphoward/status/1881000904230617365)

Validation per token accuracy will continue to improve for many epochs after validation loss appears to be getting worse. That’s because the probability calibration gets worse, not because the predictions are worse.

---

**Paul Calcraft** @paul\_cal [2025-01-19](https://x.com/paul_cal/status/1881017600685281409)

By token accuracy do you mean probability that the highest weighted token equals the target token? So greedy decoding accuracy?

---

**Jeremy Howard** @jeremyphoward [2025-01-19](https://x.com/jeremyphoward/status/1881019756259737733)

It’s not a probability, it’s binary: 1 if the highest logit token is correct.

---

**Andrea** @\_\_AndreaW\_\_ [2025-01-19](https://x.com/__AndreaW__/status/1881003990839914536)

Jeremy, did you publish any tutorial about this?

---

**Jeremy Howard** @jeremyphoward [2025-01-19](https://x.com/jeremyphoward/status/1881007055148101782)

I think every http://fast.ai course covered it.

---

**Victor** @victor\_explore [2025-01-20](https://x.com/victor_explore/status/1881208770153554333)

actually monitoring loss is like checking your weight while building muscle - sometimes the numbers lie to you

---

**uɐɥdǝʇS** @StephanSturges [2025-01-19](https://x.com/StephanSturges/status/1881001881566433465)

the good news is you can leak alpha all day for years and still 99.9% of people won’t pick it up… which is good for your own career st least😅

---

**Paras Chopra** @paraschopra [2025-01-20](https://x.com/paraschopra/status/1881251950148333965)

what is token accuracy?

---

**Lee Penkman** @LeeLeepenkman [2025-01-19](https://x.com/LeeLeepenkman/status/1881094423678976285)

Yea :)

They can become really overtrained this way though and start regurgitating training data, and its also dependant on the decoding strategy. (sometimes this is what you want anyway like if you want a llm to remember specific things about a person recall style)

This

---

**‍ ɐɯɐu ‍** @aman\_gif [2025-01-19](https://x.com/aman_gif/status/1881059823967695022)

how is that statement related to hyperfitting?

---

**Ujas** @ujas\_1 [2025-01-19](https://x.com/ujas_1/status/1881019490156376454)

What do you mean exactly by token accuracy?

---

**Daniel Hussey** @danrhuss [2026-02-03](https://x.com/danrhuss/status/2018513048345161869)

Because loss also penalises / rewards non-outputted tokens while token accuracy doesn’t?

---

**generatorman** @generatorman\_ai [2025-01-20](https://x.com/generatorman_ai/status/1881410305407984044)

goes to show why synthetic pretraining is so good - the current web-scale pretraining distributions are misaligned to the human preference distribution.

---

**Elan Sopher Markowitz** @elan\_marko [2025-01-21](https://x.com/elan_marko/status/1881512594223271945)

Fascinating. I think collapsing the distribution might not be good for search or sample-based RL strategies though

---

**Jules Jacobs** @JulesJacobs5 [2025-01-19](https://x.com/JulesJacobs5/status/1881011805302173701)

Isn't the loss token accuracy when sampling from the model?

---

**Kush Juvekar** @smartass\_cutie [2025-01-20](https://x.com/smartass_cutie/status/1881273459751448745)

💡

---

**mag** @mag\_pl [2025-01-19](https://x.com/mag_pl/status/1881001959546896735)

Grokking paper again, I think it’s an amazing phenomenon we need to use more

---

**Ethan\_PolyPredict AI** @EthanSynthMind [2025-01-20](https://x.com/EthanSynthMind/status/1881334772825567614)

tracking token accuracy is key for real insights

---

**Jero** @jeroaranda [2025-01-19](https://x.com/jeroaranda/status/1881037848562061449)

do you think this can be leveraged for efficient natural gradient?

![Image](https://pbs.twimg.com/media/GhrJ6hxWkAArvWX?format=png&name=large)

---

**Elmo Musk — e/acc** @Elmo\_Acc [2025-01-20](https://x.com/Elmo_Acc/status/1881231660232089975)

why not both?

---

**Ali Kanaan** @kanaan\_cyber [2025-01-19](https://x.com/kanaan_cyber/status/1881004827167347188)

Interesting point, Jeremy. Tracking token accuracy certainly makes a difference. How does it impact your real-world performance? Thanks for sharing!