# Fourteen Declarative Principles of Experience-Oriented Intelligence

## Rich Sutton
### April 24, 2008

1. All goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a single externally received number (reward). "The reward hypothesis" thus life is a sequential decision-making problem, also known as a Markov decision process. "Learning is adaptive optimal control"

2. A major thing that the mind does is learn a state representation and a process for updating it on a moment-by-moment basis. The input to the update process is the current sensation, action, and state (representation). "State is constructed"

3. All action is taken at the shortest possible time scale, by a reactive, moment-by-moment policy function mapping from state to action. Anything higher or at longer time scales is for thinking about action, not for taking it. "All behavior is reactive"

4. All efficient methods for solving sequential decision-making problems compute, as an intermediate step, an estimate for each state of the long-term cumulative reward that follows that state (a value function). Subgoals are high-value states. "Values are more important than rewards"

5. A major thing that the mind does is learn a predictive model of the world's dynamics at multiple time scales. This model is used to anticipate the outcome (consequences) of different ways of behavior, and then learn from them as if they had actually happened (planning).

6. Learning and planning are fundamentally the same process, operating in the one case on real experience, and in the other on simulated experience from a predictive model of the world. "Thought is learning from imagined experience"

7. All world knowledge can be well thought of as predictions of experience. "Knowledge is prediction" In particular, all knowledge can be thought of as predictions of the outcomes of temporally extended ways of behaving, that is, policies with termination conditions, also known as "options." These outcomes can be abstract state representations if those in turn are predictions of experience.

8. State representations, like all knowledge, should be tied to experience as much as possible. Thus, the Bayesian and POMDP conceptions of state estimation are mistaken.

9. Temporal-difference learning is not just for rewards, but for learning about everything, for all world knowledge. Any moment-by-moment signal (e.g., a sensation or a state variable) can substitute for the reward in a temporal-difference error. "TD learning is not just for rewards"

10. Learning is continual, with the same processes operating at every moment, with only the content changing at different times and different levels of abstraction. "The one learning algorithm"

11. Evidence adds and subtracts to get an overall prediction or action tendency. Thus policy and prediction functions can be primarily linear in the state representation, with learning restricted to the linear parameters. This is possible because the state representation contains many state variables other than predictions and that are linearly independent of each other. These include immediate non-linear functions of the other state variables as well as variables with their own dynamics (e.g., to create internal "micro-stimuli").

12. A major thing that the mind does is to sculpt and manage its state representation. It discovers a) options and option models that induce useful abstract state variables and predictive world models, and b) useful non-linear, non-predictive state variables. It continually assesses all state variables for utility, relevance, and the extent to which they generalize. Researching the process of discovery is difficult outside of the context of a complete agent.

13. Learning itself is intrinsically rewarding. The tradeoff between exploration and exploitation always comes down to "learning feels good."

14. Options are not data structures, and are not executed. They may exist only as abstractions.

---

Some of these principles are stated in radical, absolutist, and reductionist terms. This is as it should be. In some cases, softer versions of the principles (for example, removing the word "all") are still interesting. Moreover, the words "is" and "are" in the principles are a shorthand and simplification. They should be interpreted in the sense of Marr's "levels of explanation of a complex information-processing system." That is, "is" can be read as "is well thought of as" or "insight can be gained by thinking of it as."

---

## A Complete Agent

A complete agent can be obtained from just two processes:
- A moment-by-moment state-update process, and
- A moment-by-moment action selection policy.

Everything else has an effect only by changing these two. A lot can be done purely by learning processes (operating uniformly as in principle 10), before introducing planning. This can be done in the following stages:

1. A policy and value function can be learned by conventional model-free reinforcement learning using the current state variables
2. State variables with a predictive interpretation can learn to become more accurate predictors
3. Discovery processes can operate to find more useful predictive and non-predictive state variables
4. Prediction of outcomes, together with fast learning, can produce a simple form of foresight and behavior controlled by anticipated consequences

Much of the learning above constitutes learning a predictive world model, but it is not yet planning. Planning requires learning from anticipated experience at states other than the current one. The agent must disassociate himself from the current state and imagine absent others.