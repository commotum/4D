**Dwarkesh Patel (00:00:21):**

First question. My audience and I are familiar with the LLM way of thinking about AI. Conceptually, what are we missing in terms of thinking about AI from the RL perspective?

**Richard Sutton (00:00:33):**

It’s really quite a different point of view. It can easily get separated and lose the ability to talk to each other. Large language models have become such a big thing, generative AI in general a big thing. Our field is subject to bandwagons and fashions, so we lose track of the basic things. I consider reinforcement learning to be basic AI.

What is intelligence? The problem is to understand your world. Reinforcement learning is about understanding your world, whereas large language models are about mimicking people, doing what people say you should do. They’re not about figuring out what to do.

**Dwarkesh Patel (00:01:19):**

You would think that to emulate the trillions of tokens in the corpus of Internet text, you would have to build a world model. In fact, these models do seem to have very robust world models. They’re the best world models we’ve made to date in AI, right? What do you think is missing?

**Richard Sutton (00:01:38):**

I would disagree with most of the things you just said. To mimic what people say is not really to build a model of the world at all. You’re mimicking things that have a model of the world: people. I don’t want to approach the question in an adversarial way, but I would question the idea that they have a world model. A world model would enable you to predict what would happen. They have the ability to predict what a person would say. They don’t have the ability to predict what will happen.

What we want, to quote Alan Turing, is a machine that can learn from experience, where experience is the things that actually happen in your life. You do things, you see what happens, and that’s what you learn from. The large language models learn from something else. They learn from “here’s a situation, and here’s what a person did”. Implicitly, the suggestion is you should do what the person did.

**Richard Sutton (00:03:12):**

No. I agree that it’s the large language model perspective. I don’t think it’s a good perspective. To be a prior for something, there has to be a real thing. A prior bit of knowledge should be the basis for actual knowledge. What is actual knowledge? There’s no definition of actual knowledge in that large-language framework. What makes an action a good action to take?

You recognize the need for continual learning. If you need to learn continually, continually means learning during the normal interaction with the world. There must be some way during the normal interaction to tell what’s right. Is there any way to tell in the large language model setup what’s the right thing to say? You will say something and you will not get feedback about what the right thing to say is, because there’s no definition of what the right thing to say is. There’s no goal. If there’s no goal, then there’s one thing to say, another thing to say. There’s no right thing to say.

There’s no ground truth. You can’t have prior knowledge if you don’t have ground truth, because the prior knowledge is supposed to be a hint or an initial belief about what the truth is. There isn’t any truth. There’s no right thing to say. In reinforcement learning, there is a right thing to say, a right thing to do, because the right thing to do is the thing that gets you reward.

We have a definition of what’s the right thing to do, so we can have prior knowledge or knowledge provided by people about what the right thing to do is. Then we can check it to see, because we have a definition of what the actual right thing to do is.

An even simpler case is when you’re trying to make a model of the world. When you predict what will happen, you predict and then you see what happens. There’s ground truth. There’s no ground truth in large language models because you don’t have a prediction about what will happen next. If you say something in your conversation, the large language models have no prediction about what the person will say in response to that or what the response will be.

**Dwarkesh Patel (00:05:29):**

I think they do. You can literally ask them, “What would you anticipate a user might say in response?” They’ll have a prediction.

**Richard Sutton (00:05:37):**

No, they will respond to that question right. But they have no prediction in the substantive sense that they won’t be surprised by what happens. If something happens that isn’t what you might say they predicted, they will not change because an unexpected thing has happened. To learn that, they’d have to make an adjustment.

**Dwarkesh Patel (00:05:56):**

I think a capability like this does exist in context. It’s interesting to watch a model do chain of thought. Suppose it’s trying to solve a math problem. It’ll say, “Okay, I’m going to approach this problem using this approach first.” It’ll write this out and be like, “Oh wait, I just realized this is the wrong conceptual way to approach the problem. I’m going to restart with another approach.”

That flexibility does exist in context, right? Do you have something else in mind or do you just think that you need to extend this capability across longer horizons?

**Richard Sutton (00:06:28):**

I’m just saying they don’t have in any meaningful sense a prediction of what will happen next. They will not be surprised by what happens next. They’ll not make any changes if something happens, based on what happens.

**Dwarkesh Patel (00:06:41):**

Isn’t that literally what next token prediction is? Prediction about what’s next and then updating on the surprise?

**Richard Sutton (00:06:47):**

The next token is what they should say, what the actions should be. It’s not what the world will give them in response to what they do.

Let’s go back to their lack of a goal. For me, having a goal is the essence of intelligence. Something is intelligent if it can achieve goals. I like John McCarthy’s definition that intelligence is the computational part of the ability to achieve goals. You have to have goals or you’re just a behaving system. You’re not anything special, you’re not intelligent. You agree that large language models don’t have goals?

**Dwarkesh Patel (00:07:25):**

No, they have a goal.

**Richard Sutton (00:07:26):**

What’s the goal?

**Dwarkesh Patel (00:07:27):**

Next token prediction.

**Richard Sutton (00:07:29):**

That’s not a goal. It doesn’t change the world. Tokens come at you, and if you predict them, you don’t influence them.

**Dwarkesh Patel (00:07:39):**

Oh yeah. It’s not a goal about the external world.

**Richard Sutton (00:07:43):**

It’s not a goal. It’s not a substantive goal. You can’t look at a system and say it has a goal if it’s just sitting there predicting and being happy with itself that it’s predicting accurately.