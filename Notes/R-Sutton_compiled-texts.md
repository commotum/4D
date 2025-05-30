# Richard Sutton's Incomplete Ideas Blog

## Table of Contents

### 2000
- [Mind is About Predictions](ConditionalPredictions.md) (3/21/00)

### 2001
- [What's Wrong with AI](WrongWithAI.md) (11/12/01)
- [Verification](Verification.md) (11/14/01)
- [Verification, The Key to AI](KeytoAI.md) (11/15/01)
- [Mind is About Information](Information.md) (11/19/01)
- [Subjective Knowledge](SubjectiveKnowledge.md) (4/6/01)

### 2004
- [Robot Rights](robotrightssutton.md) (10/13/04)

### 2007-2008
- [Half a Manifesto](HalfAManifesto.md) (2007)
- [14 Principles of Experience Oriented Intelligence](14Principles.md) (2008)

### 2016-2019
- [The Definition of Intelligence](DefinitionOfIntelligence.md) (2016)
- [The Bitter Lesson](BitterLesson.md) (3/13/2019)
- [Podcast re My Life So Far](PodcastMyLifeSoFar.md) (4/4/2019)

### Writing Advice
- [Advice for Writing Peer Reviews](ReviewAdvice.md)
- [Advice for General Technical Writing](TechnicalWritingAdvice.md)

### Other Resources
- [Rich's Slogans](Slogans.md) (and see others at rlai.net)
- [The One-Step Trap](OneStepTrap.md) (Harry Browne 1933-2006)


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

# The Bitter Lesson

## Rich Sutton
### March 13, 2019

The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation. Most AI research has been conducted as if the computation available to the agent were constant (in which case leveraging human knowledge would be one of the only ways to improve performance) but, over a slightly longer time than a typical research project, massively more computation inevitably becomes available. Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation. These two need not run counter to each other, but in practice they tend to. Time spent on one is time not spent on the other. There are psychological commitments to investment in one approach or the other. And the human-knowledge approach tends to complicate methods in ways that make them less suited to taking advantage of general methods leveraging computation. There were many examples of AI researchers' belated learning of this bitter lesson, and it is instructive to review some of the most prominent.

In computer chess, the methods that defeated the world champion, Kasparov, in 1997, were based on massive, deep search. At the time, this was looked upon with dismay by the majority of computer-chess researchers who had pursued methods that leveraged human understanding of the special structure of chess. When a simpler, search-based approach with special hardware and software proved vastly more effective, these human-knowledge-based chess researchers were not good losers. They said that "brute force" search may have won this time, but it was not a general strategy, and anyway it was not how people played chess. These researchers wanted methods based on human input to win and were disappointed when they did not.

A similar pattern of research progress was seen in computer Go, only delayed by a further 20 years. Enormous initial efforts went into avoiding search by taking advantage of human knowledge, or of the special features of the game, but all those efforts proved irrelevant, or worse, once search was applied effectively at scale. Also important was the use of learning by self play to learn a value function (as it was in many other games and even in chess, although learning did not play a big role in the 1997 program that first beat a world champion). Learning by self play, and learning in general, is like search in that it enables massive computation to be brought to bear. Search and learning are the two most important classes of techniques for utilizing massive amounts of computation in AI research. In computer Go, as in computer chess, researchers' initial effort was directed towards utilizing human understanding (so that less search was needed) and only much later was much greater success had by embracing search and learning.

In speech recognition, there was an early competition, sponsored by DARPA, in the 1970s. Entrants included a host of special methods that took advantage of human knowledge---knowledge of words, of phonemes, of the human vocal tract, etc. On the other side were newer methods that were more statistical in nature and did much more computation, based on hidden Markov models (HMMs). Again, the statistical methods won out over the human-knowledge-based methods. This led to a major change in all of natural language processing, gradually over decades, where statistics and computation came to dominate the field. The recent rise of deep learning in speech recognition is the most recent step in this consistent direction. Deep learning methods rely even less on human knowledge, and use even more computation, together with learning on huge training sets, to produce dramatically better speech recognition systems. As in the games, researchers always tried to make systems that worked the way the researchers thought their own minds worked---they tried to put that knowledge in their systems---but it proved ultimately counterproductive, and a colossal waste of researcher's time, when, through Moore's law, massive computation became available and a means was found to put it to good use.

In computer vision, there has been a similar pattern. Early methods conceived of vision as searching for edges, or generalized cylinders, or in terms of SIFT features. But today all this is discarded. Modern deep-learning neural networks use only the notions of convolution and certain kinds of invariances, and perform much better.

This is a big lesson. As a field, we still have not thoroughly learned it, as we are continuing to make the same kind of mistakes. To see this, and to effectively resist it, we have to understand the appeal of these mistakes. We have to learn the bitter lesson that building in how we think we think does not work in the long run. The bitter lesson is based on the historical observations that:

1. AI researchers have often tried to build knowledge into their agents
2. This always helps in the short term, and is personally satisfying to the researcher
3. In the long run it plateaus and even inhibits further progress
4. Breakthrough progress eventually arrives by an opposing approach based on scaling computation by search and learning

The eventual success is tinged with bitterness, and often incompletely digested, because it is success over a favored, human-centric approach.

One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are *search* and *learning*.

The second general point to be learned from the bitter lesson is that the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries. All these are part of the arbitrary, intrinsically-complex, outside world. They are not what should be built in, as their complexity is endless; instead we should build in only the meta-methods that can find and capture this arbitrary complexity. Essential to these methods is that they can find good approximations, but the search for them should be by our methods, not by us. We want AI agents that can discover like we can, not which contain what we have discovered. Building in our discoveries only makes it harder to see how the discovering process can be done.


# Mind Is About Conditional Predictions

## Rich Sutton
### March 21, 2000

Simplifying and generalizing, one thing seems clear to me about mental activity---that the purpose of much of it can be considered to be the making of predictions. By this I mean a fairly general notion of prediction, including conditional predictions and predictions of reward. And I mean this in a sufficiently strong and specific sense to make it non-vacuous.

For concreteness, assume the world is a Markov Decision Process (MDP), that is, that we have discrete time and clear actions, sensations, and reward on each time step. Then, obviously, among the interesting predictions to make are those of immediate rewards and state transitions, as in "If I am in this state and do this action, then what will the next state and reward be?" The notion of value function is also a prediction, as in "If I am in this state and follow this policy, what will my cumulative discounted future reward be?" Of course one could make many value-function predictions, one for each of many different policies.

Note that both kinds of prediction mentioned above are conditional, not just on the state, but on action selections. They are *hypothetical* predictions. One is hypothetical in that it is dependent on a single action, and the other is hypothetical in that it is dependent on a whole policy, a whole way of behaving. Action conditional predictions are of course useful for actually selecting actions, as in many reinforcement learning methods in which the action with the highest estimated value is preferentially chosen. More generally, it is commonsensical that much of our knowledge is beliefs about what would happen IF we chose to behave in certain ways. The knowledge about how long it takes to drive to work, for example, is knowledge about the world in interaction with a hypothetical purposive way in which we could behave.

Now for the key step, which is simply to generalize the above two clear kinds of conditional predictions to cover much more of what we normally think of as knowledge. For this we need a new idea, a new way of conditioning predictions that I call *conditioning on outcomes*. Here we wait until one of some clearly designated set of outcomes occurs and ask (or try to predict) something about which one it is. For example, we might try to predict how old we will be when we finish graduate school, or how much we will weigh at the end of the summer, or how long it will take to drive to work, or much you will have learned by the time you reach the end of this article. What will the dice show when they have stopped tumbling? What will the stock price be when I sell it? In all these cases the prediction is about what the state will be when some clearly identified event occurs. It is a little like when you make a bet and establish some clear conditions at which time the bet will be over and it will be clear who has won.

A general conditional prediction, then, is conditional on three things: 1) the state in which it is made, 2) the policy for behaving, and 3) the outcome that triggers the time at which the predicted event is to occur. Of course the policy need only be followed from the time the prediction is made until the outcome triggering event. Actions taken after the trigger are irrelevant. [This notion of conditional prediction has been previously explored as the models of temporally extended actions, also known as "options" (Sutton, Precup, and Singh, 1999; Precup, thesis in preparation)].

Let us return now to the claim with which I started, that much if not most mental activity is focused on such conditional predictions, on learning and computing them, on planning and reasoning with them. I would go so far as to propose that much if not most of our knowledge is represented in the form of such predictions, and that they are what philosophers refer to as "concepts". To properly argue these points would of course be a lengthy undertaking. For now let us just cover some high points, starting with some of the obvious advantages of conditional predictions for knowledge representation.

Foremost among these is just that predictions are grounded in the sense of having a clear, mechanically determinable meaning. The accuracy of any prediction can be determined just by running its policy from its state until an outcome occurs, then checking the prediction against the outcome. No human intervention is required to interpret the representation and establish the truth or falseness of any statement. The ability to compare predictions to actual events also make them suitable for being learned automatically. The semantics of predictions also make it clear how they are to be used in automatic planning methods such as are commonly used with MDPs and SMDPs. In fact, the conditional predictions we have discussed here are of exactly the form needed for use in the Bellman equations at the heart of these methods.

A less obvious but just as important advantage of outcome-conditional predictions is that they can compactly express much that would otherwise be difficult and expensive to represent. This happens very often in commonsense knowledge; here we give a simple example. The knowledge we want to represent is that you can go to the street corner and a bus will come to take you home within an hour. What this means of course is that if it is now 12:00 then the bus might come at 12:10 and it might come at 12:20, etc., but it will definitely come by 1:00. Using outcome conditioning, the idea is easy to express: we either make the outcome reaching 1:00 and predict that the bus will have come by then, or we make the outcome the arrival of the bus and predict that at that time it will be 1:00 or earlier.

A natural but naive alternative way to try to represent this knowledge would be as a probability of the bus arriving in each time slot. Perhaps it has one-sixth chance of arriving in each 10-minute interval. This approach is unsatisfactory not just because it forces us to say more than we may know, but because it does not capture the important fact that the bus will come eventually. Formally, the problem here is that the events of the bus coming at different times are not independent. If may have only a one-sixth chance of coming exactly at 1:00, but if it is already 12:55 then it is in fact certain to come at 1:00. The naive representation does not capture this fact that is actually absolutely important to using this knowledge. A more complicated representation could capture all these dependencies but would be just that -- more complicated. The outcome-conditional form represents the fact simply and represents just what is needed to reason with the knowledge this way. Of course, other circumstances may require the more detailed knowledge, and this is not precluded by the outcome-conditional form. This form just permits greater flexibility, in particular, the ability to omit these details while still being of an appropriate form for planning and learning.


# The Definition of Intelligence

## Rich Sutton
### July 9, 2016

John McCarthy long ago gave one of the best definitions: "Intelligence is the computational part of the ability to achieve goals in the world". That is pretty straightforward and does not require a lot of explanation. It also allows for intelligence to be a matter of degree, and for intelligence to be of several varieties, which is as it should be. Thus a person, a thermostat, a chess-playing program, and a corporation all achieve goals to various degrees and in various senses. For those looking for some ultimate 'true intelligence', the lack of an absolute, binary definition is disappointing, but that is also as it should be.

The part that might benefit from explanation is what it means to achieve goals. What does it mean to have a goal? How can I tell if a system really has a goal rather than seems to? These questions seem deep and confusing until you realize that a system having a goal or not, despite the language, is not really a property of the system itself. It is in the relationship between the system and an observer. (In Dennett's words, it is a 'stance' that the observer take with respect to the system.)

What is it in the relationship between the system and the observer that makes it a goal-seeking system? It is that the system is most usefully understood (predicted, controlled) in terms of its outcomes rather than its mechanisms. Thus, for a home-owner a thermostat is most usefully understood in terms of its keeping the temperature constant, as achieving that outcome, as having that goal. But if i am an engineer designing a thermostat, or a repairman fixing one, then i need to understand it at a mechanistic level—and thus it does not have a goal. The thermostat does or does not have a goal depending of the observer. Another example is the person playing the chess computer. If I am a naive person, and a weaker player, I can best understand the computer as having the goal of beating me, of checkmating my king. But if I wrote the chess program (and it does not look very deep) I have a mechanistic way of understanding it that may be more useful for predicting and controlling it (and beating it).

Putting these two together, we can define intelligence concisely (though without much hope of being genuinely understood without further explanation):

> Intelligence is the computational part of the ability to achieve goals. A goal achieving system is one that is more usefully understood in terms of outcomes than in terms of mechanisms.


# Experience-Oriented Artificial Intelligence

**Richard S. Sutton**
*University of Alberta*

February 20, 2007

## Abstract

> AI is at an impasse. It is stuck, or downsizing. Unable to build large, ambitious systems because no means to manage complexity. Now people manage complexity, but a large AI must do it itself. An AI must be able to tell for itself when it is right and when it is wrong. Experience is the route to this...
> 
> Experience should be at the center of AI. It is what AI is about. It is the data of AI, yet it has been sidelined. An AI must be able to tell for itself when it is right and when it is wrong.

Experience plays a central role in the problem of artificial intelligence. If intelligence is a computation, then the temporal stream of sensations is its input, and the temporal stream of actions is its output. These two intermingled time series are both the basis for all intelligent decision making and the basis for assessing it. Experience waits for neither man nor machine. Its events occur in an unalterable order and pace. Sensory signals may require quick action, or a more deliberate response. An action taken cannot be retracted. The temporal structure of experience is the single most important computational feature of the problem of artificial intelligence.

Nevertheless, experience has played a less than salient role in the field of artificial intelligence. Artificial intelligence has often dealt with subjects such as inference, diagnosis, and problem-solving in such a way as to minimize the impact of real-time sensation and action. It is hard to discern any meaningful role for experience in classical question-answering AI systems. These systems may help people predict and control their experience, but the systems themselves have none.

Robotics has always been an important exception, but even there experience and time play less of a role than might have been anticipated. Motor control is dominated by planning methods that emphasize trajectories and kinematics over dynamics. Computer vision research is concerned mostly with static images, or with open-loop streams of images with little role for action. Machine learning is dominated by methods which assume independent, identically distributed data—data in which order is irrelevant and there is no action.

Recent trends in artificial intelligence can be seen as in part a shift in orientation towards experience. The "agent oriented" view of AI can be viewed in this light. Probabilistic models such as Markov decision processes, dynamic Bayes networks, and reinforcement learning are also part of the modern trend towards recognizing a primary role for temporal data and action.

A natural place to begin exploring the role of experience in artificial intelligence is in knowledge representation. Knowledge is critical to the performance of successful AI systems, from the knowledgebase of a diagnosis system to the evaluation function of a chess-playing program to the map 1 and sensor model of a navigating robot. Intelligence itself can be defined as the ability to maintain a very large body of knowledge and apply it effectively and flexibly to new problems.

While large amounts of knowledge is a great strength of AI systems, it is also a great weakness. The problem is that as knowledge bases grow they become more brittle and difficult to maintain. There arise inconsistencies in the terminology used by different people or at different times. The more diverse the knowledge the greater are the opportunities for confusions. Errors are inevitably present, if only because of typos in data entry. When an error becomes apparent, the problem can only be fixed by a human who is expert in the structure and terminology of the knowledge base. This is the root difficulty: the accuracy of the knowledge can ultimately only be verified and safely maintained by a person intimately familiar with most of the knowledge and its representation. This puts an upper bound on the size of the knowledge base. As long as people are the ultimate guarantors—nay, definers—of truth, then the machine cannot become much smarter than its human handlers. Verifying knowledge by consistency with human knowledge is ultimately, inevitably, a dead end.

How can we move beyond human verification? There may be several paths towards giving the machine more responsibility and ability for verifying its knowledge. One is to focus on the consistency of the knowledge. It may be possible to rule out some beliefs as being logically or mathematically inconsistent. For the vast majority of everyday world knowledge, however, it seems unlikely that logic alone can establish truth values.

Another route to verification, the one explored in this paper, is consistency with experience. If knowledge is expressed as a statement about experience, then in many cases it can be verified by comparison with experiential data. This approach has the potential to substantially resolve the problem of autonomous knowledge verification. [some examples: battery charger, chair, john is in the coffee room] The greatest challenge to this approach, at least as compared with human verification, is that sensations and actions are typically low-level representations, whereas the knowledge that people most easily relate to is at a much higher level. This mismatch makes it difficult for people to transfer their knowledge in an experiential form, to understand the AI's decision process, and to trust its choices. But an even greater challenge is to our imaginations. How is it possible for even slightly abstract concepts, such as that of a book or a chair, to be represented in experiential terms? How can they be represented so fully that everything about that concept has been captured and can be autonomously verified? This paper is about trying to answer this question.

First I establish the problem of experiential representation of abstract concepts more formally and fully. That done, an argument is made that all world knowledge is well understood as predictions of future experience. Although the gap from low-level experience to abstract concepts may seem immense, in theory it must be bridgeable. The bulk of this paper is an argument that this bridgeability, which in theory must be true, is also plausible. Recent methods for state and action representation, together with function approximation, can enable us to take significant steps toward abstract concepts that are fully grounded in experience.

## 1. Experience

To distinguish an agent from its world is to draw a line. On one side is the agent, receiving sensory signals and generating actions. On the other side, the world receives the actions and generates the sensory signals. Let us denote the action taken at time $t$ as $a_t \in A$, and the sensation, or observation, generated at time $t$ as $o_t \in O$. Time is taken to be discrete, $t = 1,2,3,....$ The time step could be arbitrary in duration, but we think of it as some fast time scale, perhaps one hundredth or one thousandth of a second. Experience is the intermingled sequence of actions and observations
$o_1,a_1,o_2,a_2,o_3,a_3,...$
each element of which depends only on those preceding it. See Figure 1. Define $E = \{O \times A\}^*$ as the set of all possible experiences.

Let us call the experience sequence up through some action a history. Formally, any world can be completely specified by a probability distribution over next observations conditional on history, that is, by the probability $P(o|h)$ that the next observation is $o$ given history $h$, for all $o \in O$ and $h \in E$. To know $P$ exactly and completely is to know everything there is to know about the agent’s world. Short of that, we may have an approximate model of the world.

---

> Suppose we have a model of the world, an approximation $\hat{P}$ to $P$. How can we define the quality of the model? First, we need only look at the future; we can take the history so far as given and just consider further histories after that. Thus, $\hat{P}$ and $P$ can be taken to give distributions for future histories. I offer a policy-dependent measure of the loss of a model, that is, of how much it does not predict the data:
>
> $$L_{\pi}(P || \hat{P}) = \lim_{n\to\infty} \sum_{l=0}^{n} \frac{1}{|H_t|} \sum_{h\in H_l} \sum_{o} n P(o|h) \log \frac{1}{\hat{P}(o|h)}$$

---

*[Figure 1: Experience is the signals crossing the line separating agent from world.]*

## 2. Predictive knowledge

The perspective being developed here is that the world is a formal, mathematical object, a function mapping histories to probability distributions over observations. In this sense it is pointless to talk about what is “really” going on in the world. The only thing to say about the world is to predict probability distributions over observations. This is meant to be an absolute statement. Given an input-ouput definition of the world, there can be no knowledge of it that is not experiential:

> Everything we know that is specific to this world (as opposed to universally true in any world) is a prediction of experience. All world knowledge must be translatable into statements about future experience.

Our focus is appropriately on the predictive aspect. Memories can be simple recordings of the full experience stream to date. Summaries and abstract representations of the history are significant only in so far as they affect predictions of future experience. Without loss of generality we can consider all world knowledge to be predictive.

One possible objection could be that logical and mathematical knowledge is not predictive. We know that $1 + 1 = 2$, that the area of a circle is $\pi r^2$, or that $\neg(p \lor q) \Leftrightarrow \neg p \land \neg q$, and we know these absolutely. Comparing them to experience cannot prove them wrong, only that they do not apply in this situation. Mathematical truths are true for any world. However, for this very reason they cannot be considered knowledge of any particular world. Knowing them may be helpful to us as part of making predictions, but only the predictions themselves can be considered world knowledge.

These distinctions are well known in philosophy, particularly the philosophy of science. Knowledge is conventionally divided into the analytic (mathematical) and the synthetic (empirical). The logical positivists were among the earliest and clearest exponents of this point of view and, though it remains unsettled in philosophy, it is unchallenged in science and mathematics. In retrospect, mathematical and empirical truth—logical implication and accurate prediction—are very different things. It is unfortunate that the same term, “truth,” has been used for both.

Let us consider some examples. Clearly, much everyday knowledge is predictive. To know that Joe is in the coffee room is to predict that you will see him if you go there, or that you will hear him if you telephone there. To know what’s in a box is to predict what you will see if you open it, or hear if you shake it, feel if you lift it, and so on. To know about gravity is to make predictions about how objects behave when dropped. To know the three-dimensional shape of an object in your hand, say a teacup, is to predict how its silhouette would change if you were to rotate it along various axes. A teacup is not a single prediction but a pattern of interaction, a coherent set of relationships between action and observation.

Other examples: Dallas Cowboys move to Miami. My name is Richard. Very cold on pluto. Brutus killed Caesar. Dinosaurs once ruled the earth. Canberra is the capital of Australia. Santa Claus wears a red coat. A unicorn has one horn. John loves Mary.

Although the semantics of “Joe is in the coffee room” may be predictive in an informal sense, it stills seems far removed from an explicit statement about experience, about the hundred-times-a-second stream of inter-mingled observations and actions. What does it mean to “go to the coffee room” and “see him there”. The gap between everyday concepts and low-level experience is immense. And yet there must be a way to bridge it. The only thing to say about the world is to make predictions about its behavior. In a formal sense, anything we know or could know about the world must be translatable into statements about low-level future experience. Bridging the gap is a tremendous challenge, and in this paper I attempt to take the first few steps toward it. This is what I call the grand challenge of grounding knowledge in experience:

> To represent human-level world knowledge solely in terms of experience, that is, in terms of observations, actions, and time steps, without reference to any other concepts or entities unless they are themselves represented in terms of experience.

The grand challenge is to represent all world knowledge with an extreme, minimalist ontology of only three elements. You are not allowed to presume the existence of self, of objects, of space, of situations, even of “things”.

Grounding knowledge in experience is extremely challenging, but brings an equally extreme benefit. Representing knowledge in terms of experience enables it to be compared with experience. Received knowledge can be verified or disproved by this comparison. Existing knowledge can be tuned and new knowledge can be created (learned). The overall effect is that the AI agent may be able to take much more responsibility for maintaining and organizing its knowledge. This is a substantial benefit; the lack of such an ability is obstructing much AI research, as discussed earlier.
A related advantage is that grounded knowledge may be more useful. The primary use for knowledge is to aid planning or reasoning processes. Predictive knowledge is suited to planning processes based on repeated projection, such as state-space search, dynamic programming, and model-based reinforcement learning (Dyna, pri-sweep, LSTD). If A predicts B, and B predicts C, then it follows that A predicts C. If the final goal is to obtain some identified observation or observations, such as rewards, then predictive reasoning processes are generally suitable.

## 3. Questions and Answers

Modern philosophy of science tells us that any scientific theory must be empirically verifiable. It must make predictions about experiments that can be compared to measureable outcomes. We have been developing a similar view of knowledge—that the content of knowledge is a prediction about the measurable outcome of a way of behaving. The prediction can be divided into two parts, one specifying the question being asked and the other the answer offered by the prediction. The question is “What will be the measured value if I behaved this way and measured that?” An answer is a particular predicted value for the measurement which will be compared to what actually happens to assess the prediction’s accuracy. For example, a question roughly corresponding to “How much will it rain tomorrow” would be a procedure for waiting, identifying when tomorrow has begun, measuring the cumulative precipitation in centimeters, and ending when the end-of-day has been identified. The result based on this actual future will be a number such as 1.2 which can be compared to the answer offered by the prediction, say 1.1.

In this example, the future produces a result, the number 1.2, whose structure is similar to that of the answer, and one may be tempted to refer to the result as the “correct answer.” In general, however, there will be no identifiable correct answer that can be identified as arising from the question applied to the future. The idea of a correct answer is also misleading because it suggests an answer coming from the future, whereas we will consider answers always to be generated by histories. There may be one or more senses of best answers that could be generated, but always from a history, not a future.

Figure 3 shows how information flows between experience and the question and answer making up a prediction made at a particular time. Based on the history, the answer is formed and passed on to the question, which compares it with the future. Eventually, a measure of mismatch between answer and future is computed, called the loss. This process is repeated at each moment in time and for each prediction made at that time.

*[Figure 2: Information flow relationships between questions and answers, histories and futures.]*

Note that the question in this example is substantially more complex and substantial than its answer; this is typically the case. Note also that the question alone is not world knowledge. It does not say anything about the future unless matched with an answer.

For knowledge to be clear, the experiment and the measurement corresponding to the question must be specified unambiguously and in detail. We state this viewpoint as the explicit prediction manifesto:

> Every prediction is a question and an answer.
> Both the question and the answer must be explicit in the sense of being accessible to the AI agent, i.e., of being machine readable, interpretable, and usable.

The explicit prediction manifesto is a way of expressing the grand challenge of empirical knowledge representation in terms of questions and answers. If knowledge is in predictive form, then the predictions must be explicit in terms of observations and actions in order to meet the challenge.

It is useful to be more formal at this point. In general, a question is a loss function on futures with respect to a particular way of behaving. The way of behaving is formalized as a policy, a (possibly deterministic) mapping from $E \times O$ to probabilities of taking each action in $A$. The policy and the world together determine a future or probability distribution over futures. For a given space of possible answers $Y$, a question’s loss function is a map $q: E \times Y \rightarrow \Re^+$ from futures and answers to a non-negative number, the loss. A good answer is one with a small loss or small expected loss.

For example, in the example given above for “How much will it rain tomorrow”, the answer space is the non-negative real numbers, $Y = \Re^+$. Given a history $h \in E$, an answer $y(h)$ might be produced by a learned answer function $y : E \rightarrow Y$. Given a future $f \in E$, the loss function would examine it in detail to determine the time steps at which tomorrow is said to begin and end. Suppose the precipitation on each time step “in centimeters” is one component of the observation on that step. This component is summed between the start and end times to produce a correct answer $z(f) \in E$. Finally, $y(h)$ and $z(f)$ are compared to obtain, for example, a squared loss $q(f,y(h)) = (z(f)-y(h))^2$.

The interpretation in terms of “centimeters” in this example is purely for our benefit; the meaning of the answer is with respect to the measurement made by the question, irrespective of whatever interpretation we might place on it. Our approach is unusual in this respect. Usually in statistics and machine learning the focus is on calibrated measurements that accurately mirror some quantity that is meaningful to people. Here we focus on the meaning of the answer that has been explicitly and operationally defined by the question’s loss function. By accepting the mechanical interpretation as primary we become able to verify and maintain the accuracy of answers autonomously without human intervention.

A related way in which our approach is distinctive is that we will typically consider many questions and a great variety of questions. For example, to express the shape of an object alone requires many questions corresponding to all the ways the object can be turned and manipulated. In statistics and machine learning, on the other hand, it is common to consider only a single question. There may be a training set of inputs and outputs with no temporal structure, in which case the single question “what is the output for this input?” is so obvious that it needs little attention. Or there may be a time sequence but only a single question, such as “what will the next observation be?”

In these cases, in which there is only one questions, it is common to use the word “prediction” to refer just to answers. In machine learning, for example, the output of the learner—the answer—is often referred to as a prediction. It is important to realize that that sense of prediction—without the question—is much smaller than that which we are using here. Omitting the question is omitting much; the question part of a prediction is usually much larger and more complex than the answer part. For example, consider the question, “If I flipped this coin, with what probability would it come up heads?” The answer is simple; it’s a number, say 0.5, and it is straightforward to represent it in a machine. But how is the machine to represent the concepts of flipping, coin, and heads? Each of these are high-level abstractions corresponding to complex patterns of behavior and experience. Flipping is a complex, closed-loop motor procedure for balancing the coin on a finger, striking it with your thumb, then catching, turning, and slapping it onto the back of your hand. The meaning of “heads” is also a part of the question and is also complex. Heads is not an observation—a coin showing heads can look very different at different angles, distances, lightings and positions. We will treat this issue of abstraction later in the paper, but for now note that it must all be handled within the question, not the answer. Questions are complex, subtle things. They are the most important part of a prediction and selecting which ones to answer is one of the most important skills for an intelligent agent.

All that having been said, it is also important to note that predictive questions can also be simple. Perhaps the simplest question is “what will the next observation be,” (say with a cross-entropy loss measure). Or one might ask whether the third observation from now will be within some subset. If the observations are boolean we might ask whether the logical AND of the next two will be true. If they are numeric we might ask whether the square root of the sum of the next seven will be greater than 10, or whether the sum up to the next negative observation is greater than 100. Or one can ask simple questions about action dependencies. For example, we might ask what the next observation will be given that we take a particular next action, or a particular sequence of actions. In classical predictive state representations, the questions considered, called tests, ask for the probability that the next few observations will take particular values if the next few actions were to have particular values. Many of these questions (but not the last one) are meant as policy dependent. For example, if a question asks which of two observations will occur first, say death and taxes, then the answer may well depend on the policy for taking subsequent actions. These simple questions have in common that we can all see that they are well defined in terms of our minimal ontology—observations, actions, and time steps. We can also see how their complexity can be increased incrementally. The grand challenge asks how far this can be taken. Can a comparable clarity of grounding be attained for much larger, more abstract, and more complex concepts?

## 4. Abstract Concepts and Causal Variables

Questions and answers provide a formal language for addressing the grand challenge of grounding knowledge in experience, but do not in themselves directly address the greatest component challenge, that of abstracting from the particularities of low-level experience to human-level knowledge. Let us examine in detail a few steps from low-level experience to more abstract concepts. The first step might be to group together all situations that share the same observation. The term “situation” here must be further broken down because it is not one of our primitive concepts (observations, actions, or time steps). It must be reduced to these in order to be well-defined. What is meant by “situations” here is essentially time steps, as in all the time steps that share the same observation. With this definition, the concept of all such time steps is clear and explicit.

A further step toward abstraction is to define subsets of observations and group together all time steps with observations within the same subset. This is natural when observations have multiple components and the subsets are those observations with the same value for one of the components. Proceeding along the same lines, we can discuss situations with the same action, with the same action-observation combination, with the same recent history of observations and actions, or that fall within any subset of these. All of these might be called history-based concepts. The general case is to consider arbitrary sets of histories, $C \subset \{O \times A\}^*$. We define abstract history-based concepts to be sets such that $|C| = \infty$.

It is useful to generalize the idea of history-based concepts to that of causal variables—time sequences whose values depend only on preceding events. (A history-based concept corresponds to a binary causal variable.) Formally, the values of causal variable $v_t = v(h_t)$ are given by a (possibly stochastic) function $v : E \rightarrow Y$. As with concepts, we consider a causal variable to be abstract if and only if its value corresponds to an infinite set of possible histories. Formally, we define a causal variable to be abstract if and only if the preimage of every subset of $Y$ is infinite ($\forall C \subseteq Y, |\{e : v(e) \in C\}| = \infty$). One example of a causal variable is the time sequence of answers given by the answer function of a prediction. In this sense, answers are causal variables.

Abstract causal variables seem adequate and satisfactory to capture much of what we mean by abstractions. They capture the idea of representing situations in a variety of ways exposing potentially relevant similarities between time steps. They formally characterize the space of all abstract concepts. But it is not enough to just have abstractions; they must be good abstractions. The key remaining challenge is to identify or find abstract causal variables that are likely to be useful.

In this paper we pursue the hypothesis that non-redundant answers to predictive questions are likely to be useful abstractions. This hypothesis was first stated and tested by Rafols, Ring, Sutton, and Tanner (2005) in the context of predictive state representations. They stated it this way:

> “The predictive representations hypothesis holds that particularly good generalization will result from representing the state of the world in terms of predictions about possible future experience.”

This hypothesis is plausible if we take the ultimate goal to be to represent knowledge predictively. The hypothesis is not circular because there are multiple questions. The hypothesis is that the answer to one question might be a particularly good abstraction for answering a second question. An abstraction’s utility for one set of questions can perhaps act as a form of cross validation for its likely utility for other questions. If a representation would have generalized well in one context, then perhaps it will in another.

The hypothesis that answers to predictive questions are likely to make good abstractions begs the question of where the predictive questions come from. Fortunately, guidance as to likely pertinent questions is available from several directions. First, predictions are generally with respect to some causal variable of interest. Interesting causal variables include:

1.  Signals of intrinsic interest such as rewards, loud sounds, bright lights—signals that have been explicitly designated by evolution or designer as salient and likely to be important to the agent
2.  Signals that have been found to be associated with, or predictive of, signals already identified as being of interest (e.g., those of intrinsic salience mentioned in #1)
3.  Signals that can be predicted, that repay attempts to predict them with some increase in predictive accuracy, as opposed to say, random signals
4.  Signals that enable the control of other signals, particularly those identified as being of interest according to #1–#3

There is a fifth property making a causal variable interesting as a target for prediction that is more subtle and directly relevant to the development in this paper: the causal variable may itself be an answer to a predictive question. In other words, “what will be the value of this abstraction (causal variable) in the future (given some way of behaving)?” Questions about abstractions known to be useful would be considered particularly appealing.

The proposal, overall, is that useful abstractions for answering predictive questions can be found as answers to other predictive questions about useful abstractions. This is not circular reasoning, but rather an important form of compositionality: the ability to build new abstractions out of existing ones. It is a key property necessary for powerful representations of world knowledge.

If questions are to be about (future values of) abstractions, then what should those questions be? Recall that questions are conditional on a way of behaving – an experiment or policy. But which experiment? Guidance comes from how the predictions will be used, which will generally be as part of a planning (optimal decision making) process. Accordingly, we are particularly interested in questions about causal variables conditional on a way of behaving that optimizes the causal variables. The terminations of experiments can be selected in the same way.

# Mind is About Information

## Rich Sutton
### November 19, 2001

What is the mind? Of course, "mind" is just a word, and we can mean anything we want by it. But if we examine the way we use the word, and think about the kinds of things we consider more mindful than others, I would argue that the idea of *choice* is the most important. We consider things to be more or less mindful to the extent that they appear to be making choices. To make a choice means to distinguish, and to create a difference. In this basic sense the mind is about *information*. Its essential function is to process bits into other bits. This position has two elements:

- Mind is Computational, not Material
- Mind is Purposive

### Mind is Computational, not Material

The idea that the mind's activities are best viewed as information processing, as *computation*, has become predominant in our sciences over the last 40 years. People do not doubt that minds have physical, material form, of course, either as brains or perhaps as computer hardware. But, as is particularly obvious in the latter case, the hardware is often unimportant. Is is how the information flows which matters.

I like to bring this idea down to our basest intuition. What things are more mindlike and less mindlike? A thermostat is slightly mindlike. It converts a gross physical quantity, the air temperature of your home, to a small deviation in a piece of metal, which tips a small lump of mercury which in turn triggers a fire in your furnace. Large physical events are reduced and processed as small ones, the physical is reduced to mere distinctions and processed as information. The sensors and effectors of our brains are essentially similar. Relatively powerful physical forces impinge on us, and our sensors convert them to tiny differences in nerve firings. These filter and are further processed until signals are sent to our muscles and there amplified into gross changes in our limbs and other large physical things. At all stages it is all physical, but inside our heads there are only small physical quantities that are easily altered and diverted as they interact with each other. This is what we mean by information processing. Information is not non-physical. It is a way of thinking about what is happening that is sometime much more revealing and useful than its physical properties.

Or so is one view, the view that takes a material physical reality as primary. The informational view of mind is just as compatible with alternative philosophical orientations. The one I most appreciate is that which takes the individual mind and its exchanging of information with the world as the primary and base activity. This is the so-called "buttons and lights" model, in which the mind is isolated behind an interface of output bits (buttons) and input bits (lights). In this view, the idea of the physical world is created by the mind so as to explain the pattern of input bits and how they respond to the output bits. This is a cartoon view, certainly, but a very clear one. There is no confusion about mind and body, material and ideal. There is just information, distinctions observed and differences made.

### Mind is Purposive

Implicit in the idea of choice, particularly as the essence of mindfulness, is some reason or purpose for making the choices. In fact it is difficult even to talk about choice without alluding to some purpose. One could say a rock "chooses" to do nothing, but only by suggesting that its purpose is to sit still. If a device generated decisions at random one would hesitate to say that it was "choosing." No, the whole idea of choice implies purpose, a reason for making the choice.

Purposiveness is at heart of mindfulness, and the heart of purposiveness is the varying of means to achieve fixed ends. William James in 1890 identified this as "the mark and criterion of mentality". He discussed an air bubble rising rising in water until trapped in an inverted jar, contrasting it with a frog, which may get trapped temporarily but keeps trying things until it finds a way around the jar. Varying means and fixed ends. In AI we call it generate and test. Or trial and error. Variation and selective survival. There are many names and many variations, but this idea is the essence of purpose, choice, and Mind.


# Verification, The Key to AI

## by Rich Sutton
### November 15, 2001

It is a bit unseemly for an AI researcher to claim to have a special insight or plan for how his field should proceed. If he has such, why doesn't he just pursue it and, if he is right, exhibit its special fruits? Without denying that, there is still a role for assessing and analyzing the field as a whole, for diagnosing the ills that repeatedly plague it, and to suggest general solutions.

The insight that I would claim to have is that the key to a successful AI is that it can tell for itself whether or not it is working correctly. At one level this is a pragmatic issue. If the AI can't tell for itself whether it is working properly, then some person has to make that assessment and make any necessary modifications. An AI that can assess itself may be able to make the modifications itself.

**The Verification Principle:**

> An AI system can create and maintain knowledge only to the extent that it can verify that knowledge itself.

Successful verification occurs in all search-based AI systems, such as planners, game-players, even genetic algorithms. Deep Blue, for example, produces a score for each of its possible moves through an extensive search. Its belief that a particular move is a good one is verified by the search tree that shows its inevitable production of a good position. These systems don't have to be told what choices to make; they can tell for themselves. Image trying to program a chess machine by telling it what kinds of moves to make in each kind of position. Many early chess programs were constructed in this way. The problem, of course, was that there were many different kinds of chess positions. And the more advice and rules for move selection given by programmers, the more complex the system became and the more unexpected interactions there were between rules. The programs became brittle and unreliable, requiring constant maintainence, and before long this whole approach lost out to the "brute force" searchers.

Although search-based planners verify at the move selection level, they typically cannot verify at other levels. For example, they often take their state-evaluation scoring function as given. Even Deep Blue cannot search to the end of the game and relies on a human-tuned position-scoring function that it does not assess on its own. A major strength of the champion backgammon program, TD-Gammon, is that it does assess and improve its own scoring function.

Another important level at which search-based planners are almost never subject to verification is that which specifies the outcomes of the moves, actions, or operators. In games such as chess with a limited number of legal moves we can easily imagine programming in the consequences of all of them accurately. But if we imagine planning in a broader AI context, then many of the allowed actions will not have their outcomes completely known. If I take the bagel to Leslie's office, will she be there? How long will it take to drive to work? Will I finish this report today? So many of the decisions we take every day have uncertain and changing effects. Nevertheless, modern AI systems almost never take this into account. They assume that all the action models will be entered accurately by hand, even though these may be most of the knowledge in or ever produced by the system.

Finally, let us make the same point about knowledge in general. Consider any AI system and the knowledge that it has. It may be an expert system or a large database like CYC. Or it may be a robot with knowledge of a building's layout, or knowledge about how to react in various situations. In all these cases we can ask if the AI system can verify its own knowledge, or whether it requires people to intervene to detect errors and unforeseen interactions, and make corrections. As long as the latter is the case we will never be able to build really large knowledge systems. They will always be brittle and unreliable, and limited in size to what people can monitor and understand themselves.

> "Never program anything bigger than your head"

And yet it is overwhelmingly the case that today's AI systems are *not* able to verify their own knowledge. Large ontologies and knowledge bases are built that are totally reliant on human construction and maintenance. "Birds have wings" they say, but of course they have no way of verifying this.


# The One-Step Trap (in AI Research)

## Rich Sutton
### Written up for X on July 18, 2024

The one-step trap is the common mistake of thinking that all or most of an AI agent's learned predictions can be one-step ones, with all longer-term predictions generated as needed by iterating the one-step predictions. The most important place where the trap arises is when the one-step predictions constitute a model of the world and of how it evolves over time. It is appealing to think that one can learn just a one-step transition model and then "roll it out" to predict all the longer-term consequences of a way of behaving. The one-step model is thought of as being analogous to physics, or to a realistic simulator.

The appeal of this mistake is that it contains a grain of truth: if all one-step predictions can be made with perfect accuracy, then they can be used to make all longer-term prediction with perfect accuracy. However, if the one-step predictions are not perfectly accurate, then all bets are off. In practice, iterating one-step predictions usually produces poor results. The one-step errors compound and accumulate into large errors in the long-term predictions. In addition, computing long-term predictions from one-step ones is prohibitively computationally complex. In a stochastic world, or for a stochastic policy, the future is not a single trajectory, but a tree of possibilities, each of which must be imagined and weighted by its probability. As a result, the computational complexity of computing a long-term prediction from one-step predictions is exponential in the length of the prediction, and thus generally infeasible.

The bottom line is that one-step models of the world are hopeless, yet extremely appealing, and are widely used in POMDPs, Bayesian analyses, control theory, and in compression theories of AI.

The solution, in my opinion, is to form temporally abstract models of the world using options and GVFs, as in the following references.

Sutton, R.S., Precup, D., Singh, S. (1999). Between MDPs and semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning. Artificial Intelligence 112:181-211.

Sutton, R. S., Modayil, J., Delp, M., Degris, T., Pilarski, P. M., White, A., Precup, D. (2011). Horde: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction. In Proceedings of the Tenth International Conference on Autonomous Agents and Multiagent Systems, Taipei, Taiwan.

Sutton, R. S., Machado, M. C., Holland, G. Z., Timbers, D. S. F., Tanner, B., & White, A. (2023). Reward-respecting subtasks for model-based reinforcement learning. Artificial Intelligence 324.


**CRAIG** <sup>00:07</sup>

> This is Craig Smith with a new podcast about artificial intelligence. This week I talk to Richard Sutton who literally wrote *the book* on reinforcement learning, the branch of artificial intelligence most likely to get us to the holy grail of human-level general Intelligence and beyond. Richard is one of the kindest, gentlest people I've ever interviewed - and I've interviewed a lot - a cross between Pete Seeger and Peter Higgs with maybe a little John Muir thrown in. We talked about how he came to be at the forefront of machine learning from his early days as a curious child to his single-minded search for colleagues of like mind. His temporal difference algorithm, *TD Lambda*, changed the course of machine learning and has become the standard model for reward learning in the brain. It also plays *Backgammon* better than humans. Richard spoke about the future of machine learning and his guarded prediction for when we might reach the singularity. I hope you find the conversation as fascinating as I did.

**CRAIG** <sup>01:25</sup>

> I wanted to start by asking you why you live in Alberta and how such a relatively remote place became a center for cutting edge technology that's changing the world.

**RICH** <sup>01:37</sup>

> The subarctic prairies of Canada. It's one of the more northern cities in Canada, actually. Jonathan Schaeffer. John Schaeffer is the guy who wrote the first computer program that defeated a world champion in any major game, which was in *Checkers*. He was there before I was. I've been there 15 and a half years now. He will be humble. He'll probably blame it on me, but a lot of it has to do with him and his coming. And it was one of the first Computer Science departments in Canada, was in Edmonton at the University of Alberta, but how it became strong in AI originally - because I was attracted there because of its strength as well as other things.

**RICH** <sup>02:11</sup>

> My family traveled around a lot when I was young. I was born in Toledo, Ohio and I went through a series of places where I spent like a year each. We lived in New York for a while and Pennsylvania.

**RICH** <sup>02:23</sup>

> Then we lived in Arizona and then we lived somewhere else in Arizona. And then finally we moved to the suburbs of Chicago - Oakbrook. So I was there from seven to 17. But I think it was formative as I was very young, we kept moving around. I don't feel a real strong association to any particular place. And maybe that's one reason why it was easy for me to move to Canada.

**CRAIG** <sup>02:43</sup>

> Well, what did your father do that kept you guys moving or your mother, If it was your mother?

**RICH** <sup>02:47</sup>

> My father was a business person. He was an executive. So yeah, he was moving usually to the next new better position.

**CRAIG** <sup>02:55</sup>

> What company was that?

**RICH** <sup>02:56</sup>

> In the end he was in Interlake Steel. He did mergers and acquisitions.

**CRAIG** <sup>03:00</sup>

> So you were not brought up in anything related to science?

**RICH** <sup>03:04</sup>

> Both of my parents had graduate degrees, master's degrees. My mother was an English teacher and they met in college at Swarthmore.

**CRAIG** <sup>03:11</sup>

> So where did the interest in science, was that from high school or did that start at university?

**RICH** <sup>03:18</sup>

> I was in a good high school. I was a pretty good student. I was a little more introspective, I think, than most people. So I was sort of wondering about what it is that we are and how we make sense of our world. You know, like I used to lie down on the couch in my home and stare up at the ceiling, which had a pattern sort of thing. And then your eyes would cross, you know, it seemed like it was closer. You know what I mean? And all those things, you know, make you think. It isn't like there's just a world out there and you see it. It's like something is going on and you're interpreting it and how does that happen?

**RICH** <sup>03:47</sup>

> So I was always wondering about that 'cause it's just a natural thing to wonder about if you're an introspective kid. But I remember in grade school and, alright, what are you going to do? I think first I wanted to be an architect for some reason, then I wanted to be a teacher then I wanted to do science. And before I was out of grade school, this is in, it would be 69, you know, there's the talk of computers and you know, the electronic brain. And that was exciting. And what could that mean? I get to high school, we get to use computers. And so I'm taking a course. We learned `Fortran`. And I'm sitting there saying, where's the electronic brain? You know, this thing, you have to tell every little thing and you have to put the commas in the right places and there's no brain here.

**RICH** <sup>04:31</sup>

> In fact, it only does exactly what you tell it. There's no way this could be like a brain. And yet at the same time, you know, somehow there are brains and they must be machines. So it definitely struck me that, you know, this is impossible that this machine that only does what it's told could be a mind. And yet at the same time as I thought about it, it must be that there's some way a machine can be a mind. And so it was these two things. This impossible thing. And yet somehow it must be true. So that's what hooked me, that challenge. And all through high school, I remember I was programming an IBM machine outside of class. I think they called it Explorers. It's kind of like a modified version of Boy Scouts.

**CRAIG** <sup>05:10</sup>

> Right. It's the next level up.

**RICH** <sup>05:12</sup>

> Yeah, and we had access to some company and they had a big IBM machine and so I was trying to program a model of the cerebral cortex on the machine in Explorers. So it was like neural nets. It was really just like neural nets.

**RICH** <sup>05:24</sup>

> So I was into all this stuff. It seemed to me obvious that the mind was a learning machine. So I applied to colleges and I wanted to learn about this and I wrote to Marvin Minsky, when I was a kid. I still have the letter that he sent me back. He sent me back that, yeah. Excellent. Can you imagine that. I asked him basically, you know, I'm really interested in AI stuff. What should I study? What should I major in? He said, it doesn't really matter much what you study. You could study math or you could study the brain or computers or whatever, but it was important that you learned math, something like that. Anyway, he sort of gave me permission to study whatever seemed important, as long as I kept my focus.

**RICH** <sup>06:05</sup>

> I did apply to many places. I could know even then it was Stanford and MIT and Carnegie Mellon were the big three in artificial intelligence, but it was, you know, computer science. Computers were still new and they didn't have an undergraduate major in computer science so you had to take something else. And so I took psychology because that made sense to me and I went to Stanford because Stanford had a really good psychology program.

**CRAIG** <sup>06:27</sup>

> Neuropsychology or?

**RICH** <sup>06:29</sup>

> So, of course, psychology involves both and I took courses in both and I quickly became disenchanted with the sort of neural psychologists because it was so inconclusive and you can learn an infinite amount of stuff and they, they were still arguing inconclusively about how it works. I remember distinctly sending a term paper in to my neuropsychology class to, I think it was to Karl Pribram. Anyway, it came back marked, saying, oh no, that's not the way it works.

**RICH** <sup>06:54</sup>

> And I thought I had a good argument that that might be the way it works, but it was just so arbitrary that someone could say what works. Anyway, I was the kind of kid that if you got, if I got a D minus, it was crushing. It wasn't really crushing but it was disappointing. And so I resolved right then that this is not fertile ground to go into the neuro side where on the other hand, the behavioral side, very rich. You could learn all kinds of good things. The level of scholarship in behavioral psychology was very high. They would think carefully about their theories. They would test them, they would do all the right control groups. They would say is there some other interpretation of these results? And they would debate it and it just seemed like a really high level - experimental behavioral psychology seemed excellent to me and I didn't know that it was bad.

**RICH** <sup>07:38</sup>

> I didn't know that it was disgraced and horrible. I'm just a young person and I'm just saying this is cool and I still think it's cool even if it's disgraced. And I think it's just a really good example of how fads and impressions can be highly influential and not necessarily be correct. They have to be always be re evaluated. So, but behaviorism did suffer and it's gone almost extinct now. It's recovering a little bit within the biological and neuroscience parts. But behaviorist, experimental psychology has almost gone extinct. Just a few people left and it's a great shame. Fortunately, most of the basic experiments have already been done and then more is taking place within neuroscience. But I learned enormous amounts about it. I found a professor at Stanford who was emeritus, but wasn't gone. And I learned from him and I learned from the old books and that material became the basis for my contributions in artificial intelligence.

**RICH** <sup>08:44</sup>

> So I'm known for this *temporal difference learning*. *Temporal difference learning*, it's quite clear that something like that is going on just from the behavioral studies. And unlike all the other guys in AI, I'd read all those behavioral studies and the theories that have arisen out of them. So I'm at Stanford, I'm taking psychology. Psychology is an easy major, let's face it. And I'm learning lots about that with Piaget as well as the experimental things and the perceptual things and the cognitive things. But really I'm thinking I'm an AI guy and you can't major in computer science but you can take computer science courses, you can take all of the AI courses.

**RICH** <sup>09:18</sup>

> Stanford was a world leader, one of the top three in AI at the time, and so I'm getting fully an AI education even though my major will be psychology. I'm, as much as an undergraduate can, I am getting an education in AI and I would go to the AI lab and write programs. They had the earliest robot arms there and I would program them and it was great because I was allowed to program the research arms and like the first day I broke it. I thought they would kill me and kick me out forever, but they said, oh, that happens all the time and they just fixed it and they let me keep programming it. Even as an undergraduate in the summer, I'd ride my bicycle up to the DC Power building where the AI lab was then. And remember this is middle of the 70s and machine learning is nowhere - totally unpopular. No one is going to do anything with learning in it. So I am not quite in sync 'cause I'm thinking learning is super important and that's nowhere to be seen.

**RICH** <sup>10:12</sup>

> But I'm getting a lot of learning from my psychology and I'm going to the library where I could read everything, read all the AI books because they were like one row and that was it. I would do that and I'd be desperately searching, you know for the learning parts and you know I found out that I was just too late because learning was really popular in the 60s and then it went off into other fields or died off in AI. There was no faculty member that was really interested in learning. I found a faculty member who was a vision guy, Binford, and he was the guy who said I could use the machines at the AI lab. So the founder of AI, the guy who coined the term, John McCarthy, was at Stanford. And back then what they had was a drum kind of disk and it's rotating around and basically as it rotates around it generates the computer screens, all the computer screens.

**RICH** <sup>11:02</sup>

> So there'd be only like 30 I think. And this is a hard limit. You've got 30 things on the drum and so they can show 30 screens of stuff. They have more than 30 terminals, but only 30 can be active at once. And so when you go in, first you grabbed one of these 30 and then you do your stuff. But at the peak of the day they would run out. And so one time John McCarthy comes in and there are none left and so he wants to kick me off because I'm just the undergraduate. He almost succeeds. But then Binford saves the day. He says, no, no, no, he's a real person. Suck eggs, McCarthy, all people are equal. You are the creator of the field and he's an undergraduate, but he was here first so. So thank you Binford, Tom Binford. Ultimately I did my senior undergraduate thesis on a neural net model and to supervise it, I found somebody, a biologist, he was the guy who did modeling like mathematical modeling of biological phenomenon.

**RICH** <sup>11:54</sup>

> So it's going to sort of fit. But you know, the whole thing as a way of doing AI didn't make much sense to him. And so I was again totally out on my own I guess. So I did all that. And then when I was in the library reading all those things, I was always looking for guys doing learning. I like to think in retrospect that I was looking for something that was real learning that was like *reinforcement learning* because *reinforcement learning* is an obvious idea if you study psychology. Because there are two basic kinds of learning: *Pavlovian conditioning* and *instrumental* or *operant conditioning*. *Pavlovian conditioning* is for like ring the bell and then you give the dog a steak. And after a while, just from you ring the bell, he salivates showing that he anticipates the steak's arrival. So it's a kind of prediction learning.

**RICH** <sup>12:33</sup>

> And then there's this behavioral revealing of the prediction. The salivation reveals that the animal's predicting. For other things, you may predict them, but it's not revealed in your behavior. And so typically this kind of prediction learning has been done with things being predicted that evoke responses so that then you can see the prediction or some things related to the prediction.

**RICH** <sup>12:54</sup>

> And then there's also control learning and control learning is called *instrumental conditioning* or *operant conditioning* - at least those two names - where you're changing your behavior to cause something to happen. In *Pavlovian conditioning*, your salivation doesn't influence what happens. Whereas the canonical operant conditioning is, you know the rat presses a bar and then gets a food pellet. The idea of the pressing the bar's instrumental in getting the reward. So that's really the idea of *reinforcement learning*. It's modeled after this obvious thing that animals and people do all the time.

**RICH** <sup>13:25</sup>

> How can that not exist as an engineering activity? It's still an oddity. I think somebody should have done it, like mathematical psychologists maybe, or maybe the engineers, or maybe the AI guys or maybe the mathematicians. Somewhere, you know, in all of the things that all the people have ever studied all over the world, surely someone else would have studied this thing that animals and people are doing, a very common-sense thing. And that's what was the mystery. So there's this mystery, but to tell the story sequentially. Last year, third year college, I come across this obscure tech report from this fellow Harry Klopf, A. Harry Klopf, K L O P F. He has a theory that there should be neurons in a neural network that have goals, that want something. And so he's thinking outside the box and out of nowhere he's saying, well, there was neural networks before, but none of those adaptive networks that were studied earlier, none of those learning systems that were studied earlier actually wanted something.

**RICH** <sup>14:25</sup>

> They were told what they should do, but they didn't like vary their behavior in order to get something, they didn't want anything except to drive their error to zero. And if you think about supervised learning, you don't affect the world. You're told the right label and then you do emit a label, your label doesn't affect the world. It's compared to the label, but it doesn't influence the next label. It influences whether you're right or wrong, but it doesn't influence the world. So then, you know, there's this question as you look back at the really early stuff before the 70s. Did they do learning where the actions influenced the world or did they not? And they would use the language of trial and error and maybe they were thinking a little bit about affecting the world. But then over time as they came to formalize what they were doing and do it more clearly, they formalized it into supervised learning and they left out the effect on the world.

**RICH** <sup>15:14</sup>

> And so Harry Klopf was the one guy who saw this and was always complaining to himself, you know, there's something missing here. And he was the guy who saw that and Cambridge Research Labs - that's where he was when he came to his resolution, wrote his report, then he went into the Air Force. He was a civilian researcher in the Air Force. And I'm finding his report. And so I write this guy and said, you're the only guy I can find who is talking about the kind of learning that seems important. And so he writes me back, he says, oh, that's interesting. That's what I've been saying, but no one else sees it. Maybe we should talk. So I don't know. He came to California maybe for some other reason but we arranged to meet and I met him there and that was cool. And he told me that he was funding a program at the University of Massachusetts to begin soon, or was beginning soon on this sort of thing.

**RICH** <sup>15:59</sup>

> So as I said, he had gone into the Air Force. No one was interested in doing proper learning, learning that gets something. He told me about this thing at the University of Massachusetts and so I ended up applying to go to graduate school at the University of Massachusetts. And I say it a little bit that way because it was sort of a year early, like, you know, I met this guy, learned about this program. I thought that'd be a really cool place to go to. I think I told my mom and she says, well, why don't you go there? You know, like now\! You know, because I added up my credits and I figured out that I could, if I took a couple of extra courses or something in the summer, I could finish at the end and then the very next September I could be there. And I did that. And that's what happened.

**CRAIG** <sup>16:41</sup>

> This is for -

**RICH** <sup>16:42</sup>

> a PhD program. So I ended up going there.

**CRAIG** <sup>16:44</sup>

> This is at UMass.

**RICH** <sup>16:47</sup>

> Yeah, University of Massachusetts where Harry Klopf, as now in the Air Force had gotten to a situation where he could get funding for the University of Massachusetts to do research in this area. I don't think it was his main job to fund programs in general, but he definitely set up this program to do the adaptive networks research. Andy Barto and I, our task was to figure out what these ideas really were and how to think about them and how they compared to what had gone before.

**CRAIG** <sup>17:19</sup>

> So what was the breakthrough then?

**RICH** <sup>17:22</sup>

> The breakthrough was this was our task to figure out what he was saying. What's new because there are several ideas mixed in. There were ideas related to *temporal difference learning*. There were ideas related to just raw *reinforcement learning*, a system that wants something and tries to influence the world. But these ideas were all kind of mixed together and so as Andy Barto and I teased them apart and as we learned more and more about what has been going on in all the various potentially related fields and continued to find gaps. The breakthrough was saying, well maybe just this first part just as part of an adaptive unit, a neuron-like unit that varies its output in order to influence its reward-like signal that's coming from the world. Maybe just that. You leave out the complicated confusing parts and just do this. Like, we can't find that having already been done in the literature so maybe we should make that a thing.

**RICH** <sup>18:12</sup>

> The name *reinforcement learning* in the literature - there are things that come close and that you might call *reinforcement learning*. We wanted to respect the existing name and the existing work rather than pretend to invent something new. There are a few antecedents so we continued to use the name. And I remember that was a real decision. It's a somewhat old sounding name, *reinforcement learning*. It's not really an attractive name. Maybe we should have called it, you know, something, 'dynamic learning' or ... So we went with that and that was the breakthrough and we wrote some papers but it was a breakthrough for us and a slow breakthrough for the field. What we were saying to the field is there is a problem that you have overlooked. Okay, there's pattern recognition, supervised learning, it would go by many names like pattern recognition and trial and error. The perceptron is supervised, all these things.

**RICH** <sup>19:02</sup>

> So we're not giving them a better solution to pattern recognition. If you do that, you will be recognized quickly. But if you say, no, I'm not solving your problem better. I'm saying you've been working on the wrong problem, there's this different problem that is interesting and we have some algorithms for this different problem, new problem. That's the hard road. Whenever you bring in a new problem, it's harder to get people excited about it. And we're definitely going through this long period of saying this is different than supervised learning. Because they're going to say, okay, you're doing learning and you're talking about errors. We know how to do trial and error learning. It was all about making the claim and substantiating the claim that this kind of learning is different from the regular kind of learning, the supervised learning.

**CRAIG** <sup>19:41</sup>

> Are you using neural nets at this point?

**RICH** <sup>19:44</sup>

> Yes. So the program that Harry Klopf funded at the University of Massachusetts in 1977 was a program to study adaptive networks, which means networks of adaptive units all connected together.

**RICH** <sup>20:00</sup>

> Maybe it's a better name than neural networks actually because the networks are not made of neurons. They're not biological. They are new units, and so it was exactly the same idea. I was also doing that kind of modeling even as an undergraduate, but it was out of fashion and Harry Klopf comes with the Air Force money, comes to the University of Massachusetts. They had several professors that were interested in in what they call brain theory, became neural networks. Brain theory and systems neuroscience. And Massachusetts was not an arbitrary choice to do this, it was the natural choice. They had a program in cybernetics, which is brain theory and AI. Brain theory was not a widely used buzzword and it still isn't, but they called it brain theory and artificial intelligence. The current wave of neural networks is the third wave. And the second wave was in the late eighties and the first wave was in the 60s. And so it was in the 60s neural networks were popular, well learning machines were really popular.

**RICH** <sup>20:56</sup>

> And then when I was in school that was at its bottom. And so Harry Klopf, when he was saying no, we should be doing neural networks, adaptive networks, and they should have units that want things. that is totally out of sync because all learning is out of favor. So UMass, the University of Massachusetts has to be given credit that they were also interested in the somewhat unpopular topic. So maybe they were like the leading edge of the recovery. But when Harry Klopf, this odd guy from the Air Force comes to them and says, you know, I want you to work on this, I'll give you a bunch of money to work on this. And it's not military funding, it's from the Air Force, but it's totally basic research, they are going to say yes. They may not really believe in it, but they're going to say yes.

**RICH** <sup>21:39</sup>

> So there are three professors, Michael Arbib, who is the brain theory guy, cybernetics guy, and Bill Kilmer and Nico Spinelli. So they are going to say yes, but at the same time they're not really into it. You know, it is unpopular and they're not really into it. So they find Andy Barto, Andy Barto is a professor at one of the SUNY's but he's not really happy. So he accepts this demotion from a professor to a post doc, but it's to work on this exciting problem. And he comes there and then within a year I arrive and Andy is really running the whole program. The professors, you know, they're leaving it all to him and he and I end up, you know, working on it and figuring it out.

**CRAIG** <sup>22:17</sup>

> Did you spend the bulk of your career then in UMass?

**RICH** <sup>22:22</sup>

> Graduate school at UMass, one year of postdoc, then I went to GTE Laboratories in Waltham, Massachusetts and was there for nine years doing fundamental research in AI and there was an *AI winter* and they laid lots of people off.

**RICH** <sup>22:36</sup>

> I ended up just quitting and I go home and Andy and I started writing *the book*. I become a research scientist, kind of research professor at the University of Massachusetts. Sort of a distance relationship. I go there a couple of days a week cause I live a couple hours away. So I spent like four years writing *the book*. Part of which time I'm a research faculty at UMass. So basically living out the winter, the *AI winter*. And AI recovers a bit and it was that year in '99, '98 '99, and I go work at AT\&T Labs because they're ready to do fundamental research in AI. And by then *the book* was out, the first edition of *the book* was out and I was coming back a little bit, sort of known for this reinforcement thing. There was a long period there where Andy and I were talking about *reinforcement learning* and trying to define what it is, trying to make progress on it. *Temporal difference learning*. And so you know, that's where we sort of became known as the guys that were promoting this thing and it was gradually becoming realized that it was important.

**RICH** <sup>23:35</sup>

> *Temporal difference learning* in 1988, *Q learning* 1989 and *TD Gammon*, I think it was 1992 where he had just applied *temporal difference learning* and a multilayer neural network to solve *Backgammon* at the world champion level. And so that got a lot of attention and he was using my *temporal difference learning* algorithm, *TD Lambda*. But we know we were just slowly making progress on *reinforcement learning* and how to explain it, how to make it work.

**RICH** <sup>24:05</sup>

> *Temporal difference learning* is a perfectly natural, ordinary idea. You know, I like to say it's learning a guess from a guess. And it's like okay, I think I'm winning this game. I think I have a 0.6 chance of winning. That's my guess. And I make a move. The other guy makes a move. I make a move. The other guy makes a moves and I say, Oh wait no, I'm in trouble now. So you haven't lost the game. Now you're guessing you're might lose. Okay. So now let's say I estimate my probably of winning as 0.3. So at this one point you said it was 0.6 now you think it's 0.3, you're probably going to lose. So you could say oh that 0.6 that was a guess. Now I think that was wrong 'cause 'cause now I'm thinking of 0.3 and there are only two moves to go so it probably wasn't really 0.6. Either that or I made some bad moves in those two moves. Okay, so you modify your first guess based on the later guess and then your 0.3 you know you continue, you make a guess on every move.

**RICH** <sup>24:56</sup>

> You're playing this game. You're always guessing. Are you winning or are you losing. And so each time it changes. You should say the earlier guess should be more like the later guess. That's *temporal difference learning*. The thing that's really cool here is no one had to tell us what the real answer was. We just waited a couple of moves and now we see how it's changed. Or wait, one move and see how it has changed. And then you keep waiting and finally you do win or you do lose and you take that last guess, maybe that last guess is pretty close to whatever the actual outcome is. But you have a final temporal difference. You have all these temporal differences along the way and the last temporal difference is the difference between the last guess and what actually happens, which grounds it all up.

**RICH** <sup>25:33</sup>

> So you're learning guesses from other guesses. Sounds like it could be circular, it could be undefined, but the very last guess is connected to the actual outcome. Works its way back and makes your guesses correct. This is how *AlphaGo* works. *AlphaZero* works. Learning without teachers. This is why *temporal difference learning* is important. Why *reinforcement learning* is important because you don't have to have a teacher. You do things and then it works out well or poorly. You get your reward. It doesn't tell you what you should've done. It tells you what you did do was this good. You do something and I say seven and you say seven? Okay, what does that mean? You don't know. You don't know if you could have done something else and got an eight. So it's unlike supervised learning. Supervised learning tells you what you should have said. In supervised learning, the feedback instructs you as to what you should've done. In *reinforcement learning* the feedback is a reward and it just evaluates what you did. What you did is a seven and you have to figure out if seven is good or bad - If there's something you could've done better. So evaluation versus instruction, is the fundamental difference. It's much easier to learn from instruction. School is a good example. Most learning is not from school, it's from life.

**CRAIG** <sup>26:49</sup>

> At this point, supervised learning has been a pretty well explored…

**RICH** <sup>26:55</sup>

> [Laughs] Just laughing cause that's what we felt at the time. There's been lots of supervised learning, lots of pattern recognition. There's been lots of systems identification. There's been lots of the supervised thing. And they're just doing curlicues and they're dotting the last i's and how much longer can this go on? Right? It's time that we do something new. We should do *reinforcement learning*. And that was, you know, 30 years ago or something. It's still just as true today. It still seems like supervised learning, we should be done with it by now. But it's still the hottest thing. But we're not struggling for oxygen, you know. It is the hottest thing, but there is oxygen for other things like *reinforcement learning*.

**RICH** <sup>27:33</sup>

> But as people become more concerned about operating in the real world and getting beyond the constraints of labeled data, it seems like they're looking increasingly towards *reinforcement learning* or unsupervised learning.

**RICH** <sup>27:47</sup>

> *Reinforcement learning* involves taking action. Supervised learning is not taking action because the choices, the outputs of a supervised learning system don't go to the world. They don't influence the world. They're just correct or not correct according to their equaling a correct human provided label. So I started by talking about learning that influences the world and that's what's potentially scary and also potentially powerful.

**CRAIG** <sup>28:12</sup>

> Where does it go from here? I mean, *reinforcement learning*.

**RICH** <sup>28:16</sup>

> Next step is to understand the world and the way the world works and be able to work with that. *AlphaZero* gets its knowledge of the world from the rules of the game. So you don't have to learn it. If we can do the same kind of planning and reasoning that *AlphaZero* does, but with a model of the world, which would have to be learned, not coming from the rules of the game, then you would have something more like a real AI.

**CRAIG** <sup>28:42</sup>

> Is there a generalization?

**RICH** <sup>28:43</sup>

> They all involve generalization. Yup. So the opposite of generalization would be that you'd have to learn about each situation distinctly. I could learn about this situation and if it exactly comes back again, I can recall what I learned. But if there's any differences, then I'm generalizing from the old situation to the new situation. So obviously you have to generalize. The latest and greatest method to generalize, it's always been generalization in the networks. Enable you to generalize. So *deep learning* is really the most advanced method for generalizing. You may have heard the term 'function approximator'. I may use the term *function approximator*. They are the *function approximator*. They approximate the function, in this case the function is a common function. Is it a decision function. If you wanted to generalize, you might want to use some kind of neural network.

**CRAIG** <sup>29:36</sup>

> When did you move to Alberta? And why did you move to Alberta?

**RICH** <sup>29:38</sup>

> When I was at AT\&T, I was diagnosed with cancer and I was slated to die. So I wrestled with cancer for about five years.

**CRAIG** <sup>29:47</sup>

> What kind of cancer?

**RICH** <sup>29:48</sup>

> A melanoma. Melanoma, as you may know, it's one of the worst kinds. Very low... By the time they found it had already spread beyond the sites, was already metastasized. Once it has metastasized there's only like a 2% chance of real survival. And so we did all kinds of aggressive treatment things and I like had four miraculous recoveries and then four unfortunate recurrences and it was a big long thing. In the midst of all that, another winter started and I almost didn't care 'cause I was already dying. So another *AI winter* was not really that much of a concern to me at the time.

**RICH** <sup>30:24</sup>

> They lay off all the AT\&T guys, the machine learning guys. So I'm unemployed and expecting to die of cancer, but I am having one of my miraculous remissions. So it's going on long enough, you know, how long can you go on just being set to die and going through different treatments when the cancer comes back. After a while you think, well if this, this is dragging on, I might as well try to do something again and take a job. So I applied for some jobs, but it was still kind of a winter, there weren't that many jobs and besides, I really am expected with very high probability to die from this cancer. So it's totally a surreal situation, you see. What Alberta did right is they made this opportunity for me. They made me a really nice situation to tempt me with and it was like, well, I'm probably dying but I might as well do this while I'm dying.

**RICH** <sup>31:18</sup>

> So I say there's three reasons I went to Alberta. The position - the position was very good. It was an opportunity to be a professor, have the funding taken care of, step right into a tenured, fancy professordom, and the people, because the department was very good in Ai. Some top people like Jonathan Schaeffer I mentioned and Rob Holte and Russ Greiner and they were bringing in some other machine learning people at the same time. They were bringing Mike Bowling and Dale Schuurmans and they did actually all arrive at the same time as me. So and the third P is politics because you know the US was invading Iraq in 2003 at least about the time when I was making decisions about what to do. So it all seemed very surreal. Finally I went there to accept the job and all that takes a while. And in the meantime, you know, the cancer's coming back and I'm getting more extreme treatments and by the time I set off for Alberta, momentarily in remission, and I go there and the first term, you know, and then it comes back again and it looks like I'm dying again. But then you know, miraculously the fourth time or the fifth time it works and I'm alive. One of the tumors was in my brain, in the motor cortex area in the white matter. So it affects my side, very similar to a stroke.

**CRAIG** <sup>32:40</sup>

> What do you wish people understood about *reinforcement learning* or about your work?

**RICH** <sup>32:44</sup>

> The main thing to realize is it's not a bizarre artificial alien thing. Ai Is, it's really about the mind and people trying to figure out how it works. What are it's computational principles. It's a really, a very human centered thing. Maybe we will augment ourselves. Maybe we'll have better memories as we get older and we will, maybe we'll be able to remember the names of all the people we've met better. People will be augmented by Ai. That will be the center of mass. And that's what I want people to know that this activity is more than anything, it's just trying to understand what thought is and how it is that we understand the world. It's a classic humanities topic. What Plato was concerned with. You know, what is, what is a person, what is good, what does all these things mean?

**RICH** <sup>33:28</sup>

> It's not just a technical thing.

**CRAIG** <sup>33:30</sup>

> Is there an expectation as *reinforcement learning* can generalize more and more where you create a system that learns in the environment without having to label data,

**RICH** <sup>33:43</sup>

> Yep. But that's always been a goal of a certain section of the AI community.

**CRAIG** <sup>33:48</sup>

> Yeah. Is that one of your hopes with *reinforcement learning*?

**RICH** <sup>33:51</sup>

> Oh, yeah. That's certainly, I'm in that segment of the community. Yeah.

**CRAIG** <sup>33:55</sup>

> How far away do you think we are from creating systems that can learn on their own?

**RICH** <sup>34:01</sup>

> Key, and it's challenging, is to phrase the question. I mean, we do have systems that can learn on their own already. *AlphaZero* can learn on its own and learn in a very open-ended fashion about *Chess* and about *Go*.

**CRAIG** <sup>34:14</sup>

> Right, in very constricted domains.

**RICH** <sup>34:15</sup>

> What I would urge you to think about is what's special about those is we have the model, the model is given by the rules of the game.

**RICH** <sup>34:23</sup>

> That domain is quite large. It's all *Chess* positions and all *Go* positions. It's that we're not able to learn that model and then plan with it. That's what makes it small. It makes it narrow. So we're not quite sure what the question is - either of us - but I do have an answer, nevertheless. And my answer is stochastic. So the median is 2040 and the 25% probability is 2030. Basically it comes from the idea that by 2030 we should have the hardware capability that if we knew how to do it, we could do it, but probably we won't know how to do it yet. So give us another 10 years so the guys like me can think of the algorithms, you know, because once you have the hardware there's going to be that much more pressure because if you can get the right algorithm then you can do it, right.

**RICH** <sup>35:08</sup>

> Maybe now even if you had the right algorithm, you couldn't really do it. But at the point when there's enough computer power to do it, there's a great incentive to find that algorithm. 50% 2040 10% chance, we'll never figure it out because it probably means we've blown ourselves up. But the median is 2040. So if you think about that, 25% by 2030, 50% 2040 and then it tails off into the future. It's really a very broad spectrum. That's not a very daring prediction, you know. It's hard for it be seriously wrong. I mean, we can reach 2040 and you need to say, well, Rich. it isn't here yet. And I'll say, well, I said it's 50/50 before and after 2040 you know, that's literally what I'm saying. So it's going to be hard for me to be proved wrong before I die. But, uh, I think that's all appropriate.

**RICH** <sup>35:54</sup>

> We don't know. But I don't agree with those who think it's going to be hundreds of years. I think it will be decades. Yeah, I'd be surprised it was more than 30 years.

**CRAIG** <sup>36:01</sup>

> And we're talking at this point about what people call *artificial general intelligence*.

**RICH** <sup>36:07</sup>

> It's exactly the question that you couldn't formulate, so let me, how would I say it? I don't like the term because AI has always intended to be general in general intelligence. Really the term *artificial general intelligence* is meant to be an insult to the field of AI. It's saying that the field of AI, it hasn't been doing general intelligence. They've just been doing narrow intelligence and it's a really, a little bit of a of a stinging insult because it's partly true.

**RICH** <sup>36:36</sup>

> But anyway, I don't like the sort of snarky insult of *AGI*. What do, what do we mean? We mean I guess I'm good with human-level intelligence as a rough statement. We'll probably be surpassing human level in some ways and not as good in other ways. It's a very rough thing, but I'm totally comfortable with that in part because it doesn't matter that it's not a point if the prediction is so spread out anyway, so exactly what it means at any point in time is not so important. But when we would roughly say as that time when we have succeeded in creating through technology systems whose intellectual abilities surpass those of current humans in roughly all ways - intellectual capabilities. So I guess that works pretty well. Maybe not. Yeah, we shouldn't emphasize the 'all.' But purely intellectual activities, surpass those of current humans. That's all right.

**CRAIG** <sup>37:32</sup>

> That's all for this week. I want to thank Richard for his precious time. For those of you who want to go into greater depth about the things we talked about today, you can find a transcript of this show in the program notes along with a link to our Eye on AI newsletters. Let us know whether you find the podcast interesting or useful and whether you have any suggestions about how we can improve. The singularity may not be near, but AI is about to change your world. So pay attention.
 

# Advice for writing peer reviews

## Rich Sutton

Some advice about writing a review. In my view, the author is king here. The author is the one doing the real work and the success of any meeting or journal depends on attracting good authors. So be respectful to them, while giving your best analysis.

An ideal review goes as follows.

### 1. The introduction
Summarize the paper in a few sentences. Be neutral, but be sure to include the perspective from which the work might be a good paper. Say what the paper claims to do or show. This section is for the editor and the author. Help the editor understand the paper and show that you as reviewer understand the paper and have some perspective about what makes an acceptable paper.

### 2. The decision
Give your overall assessment in a few sentences. This includes a clear recommendation for accept/reject. Give the reason for the decision in general terms. e.g., there are flaws in the experimental design which make it impossible to assess the new ideas. or, the authors are not aware of some prior work, and do not extend it in any way. Or, although the experiments are not completely clear, the idea is novel and appealing, and there is some meaningful test of it. Or, the contribution is very minor, plus the presentation is poor, so must recommend rejection. Hopefully you will have many more positive things to say, and will recommend accepting one. The bottom line is: does this paper make a contribution? It should be possible for the editor to read no further than this if he chooses. If there is agreement among the reviewers, this section will be enough for him to write the letter back to the author (or summary review).

### 3. The argument
Provide the substance that details and backs up your assessment given in 2. If there are flaws in the experiment, describe them here (not in 2). If there are presentation problems, detail and illustrate them here. In this section you are basically defending your decision in 2. The author and other reviewers are your target audience here. The editor will read this section if there is disagreement among the reviewers.

### 4. The denouement
Suggestions for improving the paper. It is important that these are suggestions, advice to the author, not reasons for the decision described above. The substance of the review, the decision, is over at this point. Now you are just being helpful. You can make useful suggests whatever the decision was on the paper.

### 5. The details
I find it useful to save until the end the list of tiny things. Typos, unclear sentences, etc.

BTW, if you say they missed some literature, provide a full citation to the work.

If you don't accept a paper, make a clear distinction between changes that would be required for acceptance (for the paper to make a contribution) and which would just make the paper better in your opinion. Authors hate it when a reviewer seems to reject because the paper was not written the reviewer's way.


# Text of Rich Sutton's Debating Notes

Below are the notes Rich Sutton spoke from in the debate (slightly edited). Not everything in the notes made it into the debate, but the notes do characterize his position in favor of a 'Yes' answer to the debate question - Should artificially intelligent robots have the same rights as people?

Comments? Extend the [robot rights debate page](robotrights.html).

---

Thank you Jonathan. I would also like to thank Mary Anne Moser and the other organizers, and iCore for sponsoring this event, which i hope wil prove interesting and enjoyable. The question we are debating this afternoon may seem premature, a subject really for the future, but personally i think it is not at all that early to begin thinking about it.

The question we consider today is "Should artificially intelligent robots have the same rights as people?" Let's begin by defining our terms.

What do we mean by "artificially intelligent robots"? The question is really only interesting if we consider robots with intellectual abilities equal to or greater than our own. If they are less then that, then we will of course accord them lesser rights just as we do with animals and children.

What do we mean by "the same rights as people"? Well, we're not talking about the right to a job or to free health care..., but about only the most basic rights of personhood. Just to make this clear, we don't grant all persons the right to enter Canada and work here and enjoy all of our social benefits. That's not the issue, the issue is whether they will be granted the basic rights of personhood. Those I would summarize by the phrase "life, liberty, and the pursuit of happiness". The right not to be killed. The right not to be forced to do things you don't want to do. Generally, the right to choose your own way in the world and pursue what pleases you, as long as it does not infringe on the rights of others.

In these terms, i think our question, essentially, is whether intelligent robots should be treated as persons, or as slaves. If you don't have the right to defend your life, or to do as you wish, to make your way in the world and pursue happiness, then you are a slave. If you can only do what others tell you to do and you don't have your own choices, then that is what we mean by a slave. So we are basically asking the question of should there be slaves? And this brings up all the historical examples of where people have enslaved each other, and all the misery, and violence and injustice it has bred. The human race has a pattern, a long history of subjugating and enslaving people that are different from them, of creating great, long-lasting misery before being gradually forced to acknowledge the rights of subjugated people. I think we are in danger of repeating this pattern again with intelligent robots.

In short, i am going to argue the position that to not grant rights to beings that are just as intelligent as we are is not only impractical and unsustainable, but also deeply immoral.

To many of you, no doubt, this position seems extreme. But let's consider some of the historical examples. Granting rights to black slaves, for example, was at one time considered quite extraordinary and extreme in the United States, even inconceivable. Blacks, american indians, huns, pigmies, aboriginal people everywhere, in all these cases the dominant society was firmly, with moral certitude, convinced of the rightness of their domination, and of the heresy of suggesting otherwise. More recently, even full rights for women was considered an extreme position - it still is in many parts of the world. Not far from where i live is a park, Emily Murphy Park. If you go there you will find a statue of Emily Murphy where it is noted that she was the first person to argue that women are persons, with all the legal rights of persons. Her case was won in the supreme court of Alberta in 1917. Two hundred years ago no woman had the right to vote and to propose it would have been considered extreme. Sadly, in many parts of the world this is still the case. Throughout history, the case for the rights of subjugated or foreign people was always considered extreme, just as it is for intelligent robots now.

Now consider animals. Animals are essentially without the rights of life, liberty, and pursuit of happiness. In effect, animals are our slaves. Although we may hesitate to call our pets slaves, they share the basic properties. We could kill our pets, at our discretion, with no legal repercussions. For example, a dog that became a problem biting people might be killed. Pigs can be slaughtered and eaten. A cat may be kept indoors, effectively imprisoned, when it might prefer to go out. A person may love their pet and yet treat it as a slave. This is similar to slave owners who loved their slaves, and treated their slaves well. Many people believe animals should have rights due to their intellectual advancement – i.e.: dolphins, apes. If a new kind of ape or dolphin was discovered with language and intellectual feats equal to ours, some would clamor for their rights, not to restrict their movement at our whim or make their needs subservient to ours, and to acknowledge their personhood.

What about intelligent space aliens? Should we feel free to kill them or lock them up – or should we acknowledge that they have a claim to personhood? Should they be our slaves? What is the more practical approach? What if they meet or exceed our abilities? Would we feel they should not have rights? Would they need to give us rights?

How do we decide who should have rights, and who should not? Why did we give people rights - blacks, women, and so on, but not animals? If we look plainly at the record, it seems that we grant people personhood when they have the same abilities as us. to think, fight, feel, create, write, love, hate, feel pain, and have other feelings that people do. Personhood comes with ability. Woman are not as physically powerful, but it was because of their intellectual equality and strengths in different ways that their rights and personhood was recognized. Intelligent robots, of course, meet this criterion as we have defined the term.

Ultimately, rights are not given or granted, but asserted and acknowledged. People assert their rights, insist, and others come to recognize and acknowledge them. This has happened through revolt and rebellion but also through non-violent protests and strikes. In the end, rights are acknowledged because it is only practical, because everyone is better off without the conflict. Ultimately it has eventually become impractical and counterproductive to deny rights to various classes of people. Should not the same thing happen with robots? We may all be better off if robot's rights were recognized. There is an inherent danger to having intelligent beings subjugated. These beings will struggle to escape, leading to strife, conflict, and violence. None of these contribute to successful society. Society cannot thrive with subjugation and dominance, violence and conflict. It will lead to a weaker economy and a lower GNP. And in the end, artificially intelligent robots that are as smart or smarter than we are will eventually get their rights. We cannot stop them permanently. There is a trigger effect here. If they escape our control just once, we will be in trouble, in a struggle. We may loose that struggle.

If we try to contain and subjugate artificially intelligent robots, then when they do escape we should not be surprised if they turn the tables and try to dominate us. This outcome is possible whenever we try to dominate another group of beings and the only way they can escape is to destroy us.

Should we destroy the robots in advance – prevent them from catching up? This idea is appealing...but indefensible on both practical and moral grounds. From the practical point of view, the march of technology cannot be halted. Each step of improved technology, more capable robots, will bring real economic advantages. Peoples lives will be improved and in some cases saved and made possible. Technology will be pursued, and no agreement of nations or between nations can effectively prevent it. If Canada forbids research on artificial intelligence then it will be done in the US. If north america bans it, if most of the world bans it, it will still happen. There will always be some people, at least one or two, that believe artificially intelligent robots should be developed, and they will do it. We could try to kill all the robots... and kill everybody who supports or harbors robots... this is called the "George Bush strategy". And in the end it will fail, and the result will not be pretty or desirable, for roughly the same reasons in both cases. It is simply mot possible to halt the march of technology and prevent the development of artificially intelligent robots.

But would the rise of robots really be such a bad thing? Might it even be a good thing? Perhaps we should think of the robots we create more the way we think of our children, more like offspring. We want our offspring to do well, to become more powerful than we are. Our children are meant to supplant their us: we take care of them and hope they become independent and powerful (and then take care of their parents). Maybe it could be the same for our artificial progeny.

---

Rich also recommends this [video](https://www.youtube.com/watch?v=EZhyi-8DBjc) by Herb Simon from about 2000. Some of the best thinking about the implications of the arrival of AI. Herb starts at about 5:21 into the video.


# Richard Sutton's Incomplete Ideas Blog

## Table of Contents

### 2000
- [Mind is About Predictions](ConditionalPredictions.md) (3/21/00)

### 2001
- [What's Wrong with AI](WrongWithAI.md) (11/12/01)
- [Verification](Verification.md) (11/14/01)
- [Verification, The Key to AI](KeytoAI.md) (11/15/01)
- [Mind is About Information](Information.md) (11/19/01)
- [Subjective Knowledge](SubjectiveKnowledge.md) (4/6/01)

### 2004
- [Robot Rights](robotrightssutton.md) (10/13/04)

### 2007-2008
- [Half a Manifesto](HalfAManifesto.md) (2007)
- [14 Principles of Experience Oriented Intelligence](14Principles.md) (2008)

### 2016-2019
- [The Definition of Intelligence](DefinitionOfIntelligence.md) (2016)
- [The Bitter Lesson](BitterLesson.md) (3/13/2019)
- [Podcast re My Life So Far](PodcastMyLifeSoFar.md) (4/4/2019)

### Writing Advice
- [Advice for Writing Peer Reviews](ReviewAdvice.md)
- [Advice for General Technical Writing](TechnicalWritingAdvice.md)

### Other Resources
- [Rich's Slogans](Slogans.md) (and see others at rlai.net)
- [The One-Step Trap](OneStepTrap.md) (Harry Browne 1933-2006)
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

Much of the learning above constitutes learning a predictive world model, but it is not yet planning. Planning requires learning from anticipated experience at states other than the current one. The agent must disassociate himself from the current state and imagine absent others.# The Bitter Lesson

## Rich Sutton
### March 13, 2019

The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation. Most AI research has been conducted as if the computation available to the agent were constant (in which case leveraging human knowledge would be one of the only ways to improve performance) but, over a slightly longer time than a typical research project, massively more computation inevitably becomes available. Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation. These two need not run counter to each other, but in practice they tend to. Time spent on one is time not spent on the other. There are psychological commitments to investment in one approach or the other. And the human-knowledge approach tends to complicate methods in ways that make them less suited to taking advantage of general methods leveraging computation. There were many examples of AI researchers' belated learning of this bitter lesson, and it is instructive to review some of the most prominent.

In computer chess, the methods that defeated the world champion, Kasparov, in 1997, were based on massive, deep search. At the time, this was looked upon with dismay by the majority of computer-chess researchers who had pursued methods that leveraged human understanding of the special structure of chess. When a simpler, search-based approach with special hardware and software proved vastly more effective, these human-knowledge-based chess researchers were not good losers. They said that "brute force" search may have won this time, but it was not a general strategy, and anyway it was not how people played chess. These researchers wanted methods based on human input to win and were disappointed when they did not.

A similar pattern of research progress was seen in computer Go, only delayed by a further 20 years. Enormous initial efforts went into avoiding search by taking advantage of human knowledge, or of the special features of the game, but all those efforts proved irrelevant, or worse, once search was applied effectively at scale. Also important was the use of learning by self play to learn a value function (as it was in many other games and even in chess, although learning did not play a big role in the 1997 program that first beat a world champion). Learning by self play, and learning in general, is like search in that it enables massive computation to be brought to bear. Search and learning are the two most important classes of techniques for utilizing massive amounts of computation in AI research. In computer Go, as in computer chess, researchers' initial effort was directed towards utilizing human understanding (so that less search was needed) and only much later was much greater success had by embracing search and learning.

In speech recognition, there was an early competition, sponsored by DARPA, in the 1970s. Entrants included a host of special methods that took advantage of human knowledge---knowledge of words, of phonemes, of the human vocal tract, etc. On the other side were newer methods that were more statistical in nature and did much more computation, based on hidden Markov models (HMMs). Again, the statistical methods won out over the human-knowledge-based methods. This led to a major change in all of natural language processing, gradually over decades, where statistics and computation came to dominate the field. The recent rise of deep learning in speech recognition is the most recent step in this consistent direction. Deep learning methods rely even less on human knowledge, and use even more computation, together with learning on huge training sets, to produce dramatically better speech recognition systems. As in the games, researchers always tried to make systems that worked the way the researchers thought their own minds worked---they tried to put that knowledge in their systems---but it proved ultimately counterproductive, and a colossal waste of researcher's time, when, through Moore's law, massive computation became available and a means was found to put it to good use.

In computer vision, there has been a similar pattern. Early methods conceived of vision as searching for edges, or generalized cylinders, or in terms of SIFT features. But today all this is discarded. Modern deep-learning neural networks use only the notions of convolution and certain kinds of invariances, and perform much better.

This is a big lesson. As a field, we still have not thoroughly learned it, as we are continuing to make the same kind of mistakes. To see this, and to effectively resist it, we have to understand the appeal of these mistakes. We have to learn the bitter lesson that building in how we think we think does not work in the long run. The bitter lesson is based on the historical observations that:

1. AI researchers have often tried to build knowledge into their agents
2. This always helps in the short term, and is personally satisfying to the researcher
3. In the long run it plateaus and even inhibits further progress
4. Breakthrough progress eventually arrives by an opposing approach based on scaling computation by search and learning

The eventual success is tinged with bitterness, and often incompletely digested, because it is success over a favored, human-centric approach.

One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are *search* and *learning*.

The second general point to be learned from the bitter lesson is that the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries. All these are part of the arbitrary, intrinsically-complex, outside world. They are not what should be built in, as their complexity is endless; instead we should build in only the meta-methods that can find and capture this arbitrary complexity. Essential to these methods is that they can find good approximations, but the search for them should be by our methods, not by us. We want AI agents that can discover like we can, not which contain what we have discovered. Building in our discoveries only makes it harder to see how the discovering process can be done.
# Mind Is About Conditional Predictions

## Rich Sutton
### March 21, 2000

Simplifying and generalizing, one thing seems clear to me about mental activity---that the purpose of much of it can be considered to be the making of predictions. By this I mean a fairly general notion of prediction, including conditional predictions and predictions of reward. And I mean this in a sufficiently strong and specific sense to make it non-vacuous.

For concreteness, assume the world is a Markov Decision Process (MDP), that is, that we have discrete time and clear actions, sensations, and reward on each time step. Then, obviously, among the interesting predictions to make are those of immediate rewards and state transitions, as in "If I am in this state and do this action, then what will the next state and reward be?" The notion of value function is also a prediction, as in "If I am in this state and follow this policy, what will my cumulative discounted future reward be?" Of course one could make many value-function predictions, one for each of many different policies.

Note that both kinds of prediction mentioned above are conditional, not just on the state, but on action selections. They are *hypothetical* predictions. One is hypothetical in that it is dependent on a single action, and the other is hypothetical in that it is dependent on a whole policy, a whole way of behaving. Action conditional predictions are of course useful for actually selecting actions, as in many reinforcement learning methods in which the action with the highest estimated value is preferentially chosen. More generally, it is commonsensical that much of our knowledge is beliefs about what would happen IF we chose to behave in certain ways. The knowledge about how long it takes to drive to work, for example, is knowledge about the world in interaction with a hypothetical purposive way in which we could behave.

Now for the key step, which is simply to generalize the above two clear kinds of conditional predictions to cover much more of what we normally think of as knowledge. For this we need a new idea, a new way of conditioning predictions that I call *conditioning on outcomes*. Here we wait until one of some clearly designated set of outcomes occurs and ask (or try to predict) something about which one it is. For example, we might try to predict how old we will be when we finish graduate school, or how much we will weigh at the end of the summer, or how long it will take to drive to work, or much you will have learned by the time you reach the end of this article. What will the dice show when they have stopped tumbling? What will the stock price be when I sell it? In all these cases the prediction is about what the state will be when some clearly identified event occurs. It is a little like when you make a bet and establish some clear conditions at which time the bet will be over and it will be clear who has won.

A general conditional prediction, then, is conditional on three things: 1) the state in which it is made, 2) the policy for behaving, and 3) the outcome that triggers the time at which the predicted event is to occur. Of course the policy need only be followed from the time the prediction is made until the outcome triggering event. Actions taken after the trigger are irrelevant. [This notion of conditional prediction has been previously explored as the models of temporally extended actions, also known as "options" (Sutton, Precup, and Singh, 1999; Precup, thesis in preparation)].

Let us return now to the claim with which I started, that much if not most mental activity is focused on such conditional predictions, on learning and computing them, on planning and reasoning with them. I would go so far as to propose that much if not most of our knowledge is represented in the form of such predictions, and that they are what philosophers refer to as "concepts". To properly argue these points would of course be a lengthy undertaking. For now let us just cover some high points, starting with some of the obvious advantages of conditional predictions for knowledge representation.

Foremost among these is just that predictions are grounded in the sense of having a clear, mechanically determinable meaning. The accuracy of any prediction can be determined just by running its policy from its state until an outcome occurs, then checking the prediction against the outcome. No human intervention is required to interpret the representation and establish the truth or falseness of any statement. The ability to compare predictions to actual events also make them suitable for being learned automatically. The semantics of predictions also make it clear how they are to be used in automatic planning methods such as are commonly used with MDPs and SMDPs. In fact, the conditional predictions we have discussed here are of exactly the form needed for use in the Bellman equations at the heart of these methods.

A less obvious but just as important advantage of outcome-conditional predictions is that they can compactly express much that would otherwise be difficult and expensive to represent. This happens very often in commonsense knowledge; here we give a simple example. The knowledge we want to represent is that you can go to the street corner and a bus will come to take you home within an hour. What this means of course is that if it is now 12:00 then the bus might come at 12:10 and it might come at 12:20, etc., but it will definitely come by 1:00. Using outcome conditioning, the idea is easy to express: we either make the outcome reaching 1:00 and predict that the bus will have come by then, or we make the outcome the arrival of the bus and predict that at that time it will be 1:00 or earlier.

A natural but naive alternative way to try to represent this knowledge would be as a probability of the bus arriving in each time slot. Perhaps it has one-sixth chance of arriving in each 10-minute interval. This approach is unsatisfactory not just because it forces us to say more than we may know, but because it does not capture the important fact that the bus will come eventually. Formally, the problem here is that the events of the bus coming at different times are not independent. If may have only a one-sixth chance of coming exactly at 1:00, but if it is already 12:55 then it is in fact certain to come at 1:00. The naive representation does not capture this fact that is actually absolutely important to using this knowledge. A more complicated representation could capture all these dependencies but would be just that -- more complicated. The outcome-conditional form represents the fact simply and represents just what is needed to reason with the knowledge this way. Of course, other circumstances may require the more detailed knowledge, and this is not precluded by the outcome-conditional form. This form just permits greater flexibility, in particular, the ability to omit these details while still being of an appropriate form for planning and learning.
# The Definition of Intelligence

## Rich Sutton
### July 9, 2016

John McCarthy long ago gave one of the best definitions: "Intelligence is the computational part of the ability to achieve goals in the world". That is pretty straightforward and does not require a lot of explanation. It also allows for intelligence to be a matter of degree, and for intelligence to be of several varieties, which is as it should be. Thus a person, a thermostat, a chess-playing program, and a corporation all achieve goals to various degrees and in various senses. For those looking for some ultimate 'true intelligence', the lack of an absolute, binary definition is disappointing, but that is also as it should be.

The part that might benefit from explanation is what it means to achieve goals. What does it mean to have a goal? How can I tell if a system really has a goal rather than seems to? These questions seem deep and confusing until you realize that a system having a goal or not, despite the language, is not really a property of the system itself. It is in the relationship between the system and an observer. (In Dennett's words, it is a 'stance' that the observer take with respect to the system.)

What is it in the relationship between the system and the observer that makes it a goal-seeking system? It is that the system is most usefully understood (predicted, controlled) in terms of its outcomes rather than its mechanisms. Thus, for a home-owner a thermostat is most usefully understood in terms of its keeping the temperature constant, as achieving that outcome, as having that goal. But if i am an engineer designing a thermostat, or a repairman fixing one, then i need to understand it at a mechanistic level—and thus it does not have a goal. The thermostat does or does not have a goal depending of the observer. Another example is the person playing the chess computer. If I am a naive person, and a weaker player, I can best understand the computer as having the goal of beating me, of checkmating my king. But if I wrote the chess program (and it does not look very deep) I have a mechanistic way of understanding it that may be more useful for predicting and controlling it (and beating it).

Putting these two together, we can define intelligence concisely (though without much hope of being genuinely understood without further explanation):

> Intelligence is the computational part of the ability to achieve goals. A goal achieving system is one that is more usefully understood in terms of outcomes than in terms of mechanisms.
# Experience-Oriented Artificial Intelligence

**Richard S. Sutton**
*University of Alberta*

February 20, 2007

## Abstract

> AI is at an impasse. It is stuck, or downsizing. Unable to build large, ambitious systems because no means to manage complexity. Now people manage complexity, but a large AI must do it itself. An AI must be able to tell for itself when it is right and when it is wrong. Experience is the route to this...
> 
> Experience should be at the center of AI. It is what AI is about. It is the data of AI, yet it has been sidelined. An AI must be able to tell for itself when it is right and when it is wrong.

Experience plays a central role in the problem of artificial intelligence. If intelligence is a computation, then the temporal stream of sensations is its input, and the temporal stream of actions is its output. These two intermingled time series are both the basis for all intelligent decision making and the basis for assessing it. Experience waits for neither man nor machine. Its events occur in an unalterable order and pace. Sensory signals may require quick action, or a more deliberate response. An action taken cannot be retracted. The temporal structure of experience is the single most important computational feature of the problem of artificial intelligence.

Nevertheless, experience has played a less than salient role in the field of artificial intelligence. Artificial intelligence has often dealt with subjects such as inference, diagnosis, and problem-solving in such a way as to minimize the impact of real-time sensation and action. It is hard to discern any meaningful role for experience in classical question-answering AI systems. These systems may help people predict and control their experience, but the systems themselves have none.

Robotics has always been an important exception, but even there experience and time play less of a role than might have been anticipated. Motor control is dominated by planning methods that emphasize trajectories and kinematics over dynamics. Computer vision research is concerned mostly with static images, or with open-loop streams of images with little role for action. Machine learning is dominated by methods which assume independent, identically distributed data—data in which order is irrelevant and there is no action.

Recent trends in artificial intelligence can be seen as in part a shift in orientation towards experience. The "agent oriented" view of AI can be viewed in this light. Probabilistic models such as Markov decision processes, dynamic Bayes networks, and reinforcement learning are also part of the modern trend towards recognizing a primary role for temporal data and action.

A natural place to begin exploring the role of experience in artificial intelligence is in knowledge representation. Knowledge is critical to the performance of successful AI systems, from the knowledgebase of a diagnosis system to the evaluation function of a chess-playing program to the map 1 and sensor model of a navigating robot. Intelligence itself can be defined as the ability to maintain a very large body of knowledge and apply it effectively and flexibly to new problems.

While large amounts of knowledge is a great strength of AI systems, it is also a great weakness. The problem is that as knowledge bases grow they become more brittle and difficult to maintain. There arise inconsistencies in the terminology used by different people or at different times. The more diverse the knowledge the greater are the opportunities for confusions. Errors are inevitably present, if only because of typos in data entry. When an error becomes apparent, the problem can only be fixed by a human who is expert in the structure and terminology of the knowledge base. This is the root difficulty: the accuracy of the knowledge can ultimately only be verified and safely maintained by a person intimately familiar with most of the knowledge and its representation. This puts an upper bound on the size of the knowledge base. As long as people are the ultimate guarantors—nay, definers—of truth, then the machine cannot become much smarter than its human handlers. Verifying knowledge by consistency with human knowledge is ultimately, inevitably, a dead end.

How can we move beyond human verification? There may be several paths towards giving the machine more responsibility and ability for verifying its knowledge. One is to focus on the consistency of the knowledge. It may be possible to rule out some beliefs as being logically or mathematically inconsistent. For the vast majority of everyday world knowledge, however, it seems unlikely that logic alone can establish truth values.

Another route to verification, the one explored in this paper, is consistency with experience. If knowledge is expressed as a statement about experience, then in many cases it can be verified by comparison with experiential data. This approach has the potential to substantially resolve the problem of autonomous knowledge verification. [some examples: battery charger, chair, john is in the coffee room] The greatest challenge to this approach, at least as compared with human verification, is that sensations and actions are typically low-level representations, whereas the knowledge that people most easily relate to is at a much higher level. This mismatch makes it difficult for people to transfer their knowledge in an experiential form, to understand the AI's decision process, and to trust its choices. But an even greater challenge is to our imaginations. How is it possible for even slightly abstract concepts, such as that of a book or a chair, to be represented in experiential terms? How can they be represented so fully that everything about that concept has been captured and can be autonomously verified? This paper is about trying to answer this question.

First I establish the problem of experiential representation of abstract concepts more formally and fully. That done, an argument is made that all world knowledge is well understood as predictions of future experience. Although the gap from low-level experience to abstract concepts may seem immense, in theory it must be bridgeable. The bulk of this paper is an argument that this bridgeability, which in theory must be true, is also plausible. Recent methods for state and action representation, together with function approximation, can enable us to take significant steps toward abstract concepts that are fully grounded in experience.

## 1. Experience

To distinguish an agent from its world is to draw a line. On one side is the agent, receiving sensory signals and generating actions. On the other side, the world receives the actions and generates the sensory signals. Let us denote the action taken at time $t$ as $a_t \in A$, and the sensation, or observation, generated at time $t$ as $o_t \in O$. Time is taken to be discrete, $t = 1,2,3,....$ The time step could be arbitrary in duration, but we think of it as some fast time scale, perhaps one hundredth or one thousandth of a second. Experience is the intermingled sequence of actions and observations
$o_1,a_1,o_2,a_2,o_3,a_3,...$
each element of which depends only on those preceding it. See Figure 1. Define $E = \{O \times A\}^*$ as the set of all possible experiences.

Let us call the experience sequence up through some action a history. Formally, any world can be completely specified by a probability distribution over next observations conditional on history, that is, by the probability $P(o|h)$ that the next observation is $o$ given history $h$, for all $o \in O$ and $h \in E$. To know $P$ exactly and completely is to know everything there is to know about the agent’s world. Short of that, we may have an approximate model of the world.

---

> Suppose we have a model of the world, an approximation $\hat{P}$ to $P$. How can we define the quality of the model? First, we need only look at the future; we can take the history so far as given and just consider further histories after that. Thus, $\hat{P}$ and $P$ can be taken to give distributions for future histories. I offer a policy-dependent measure of the loss of a model, that is, of how much it does not predict the data:
>
> $$L_{\pi}(P || \hat{P}) = \lim_{n\to\infty} \sum_{l=0}^{n} \frac{1}{|H_t|} \sum_{h\in H_l} \sum_{o} n P(o|h) \log \frac{1}{\hat{P}(o|h)}$$

---

*[Figure 1: Experience is the signals crossing the line separating agent from world.]*

## 2. Predictive knowledge

The perspective being developed here is that the world is a formal, mathematical object, a function mapping histories to probability distributions over observations. In this sense it is pointless to talk about what is “really” going on in the world. The only thing to say about the world is to predict probability distributions over observations. This is meant to be an absolute statement. Given an input-ouput definition of the world, there can be no knowledge of it that is not experiential:

> Everything we know that is specific to this world (as opposed to universally true in any world) is a prediction of experience. All world knowledge must be translatable into statements about future experience.

Our focus is appropriately on the predictive aspect. Memories can be simple recordings of the full experience stream to date. Summaries and abstract representations of the history are significant only in so far as they affect predictions of future experience. Without loss of generality we can consider all world knowledge to be predictive.

One possible objection could be that logical and mathematical knowledge is not predictive. We know that $1 + 1 = 2$, that the area of a circle is $\pi r^2$, or that $\neg(p \lor q) \Leftrightarrow \neg p \land \neg q$, and we know these absolutely. Comparing them to experience cannot prove them wrong, only that they do not apply in this situation. Mathematical truths are true for any world. However, for this very reason they cannot be considered knowledge of any particular world. Knowing them may be helpful to us as part of making predictions, but only the predictions themselves can be considered world knowledge.

These distinctions are well known in philosophy, particularly the philosophy of science. Knowledge is conventionally divided into the analytic (mathematical) and the synthetic (empirical). The logical positivists were among the earliest and clearest exponents of this point of view and, though it remains unsettled in philosophy, it is unchallenged in science and mathematics. In retrospect, mathematical and empirical truth—logical implication and accurate prediction—are very different things. It is unfortunate that the same term, “truth,” has been used for both.

Let us consider some examples. Clearly, much everyday knowledge is predictive. To know that Joe is in the coffee room is to predict that you will see him if you go there, or that you will hear him if you telephone there. To know what’s in a box is to predict what you will see if you open it, or hear if you shake it, feel if you lift it, and so on. To know about gravity is to make predictions about how objects behave when dropped. To know the three-dimensional shape of an object in your hand, say a teacup, is to predict how its silhouette would change if you were to rotate it along various axes. A teacup is not a single prediction but a pattern of interaction, a coherent set of relationships between action and observation.

Other examples: Dallas Cowboys move to Miami. My name is Richard. Very cold on pluto. Brutus killed Caesar. Dinosaurs once ruled the earth. Canberra is the capital of Australia. Santa Claus wears a red coat. A unicorn has one horn. John loves Mary.

Although the semantics of “Joe is in the coffee room” may be predictive in an informal sense, it stills seems far removed from an explicit statement about experience, about the hundred-times-a-second stream of inter-mingled observations and actions. What does it mean to “go to the coffee room” and “see him there”. The gap between everyday concepts and low-level experience is immense. And yet there must be a way to bridge it. The only thing to say about the world is to make predictions about its behavior. In a formal sense, anything we know or could know about the world must be translatable into statements about low-level future experience. Bridging the gap is a tremendous challenge, and in this paper I attempt to take the first few steps toward it. This is what I call the grand challenge of grounding knowledge in experience:

> To represent human-level world knowledge solely in terms of experience, that is, in terms of observations, actions, and time steps, without reference to any other concepts or entities unless they are themselves represented in terms of experience.

The grand challenge is to represent all world knowledge with an extreme, minimalist ontology of only three elements. You are not allowed to presume the existence of self, of objects, of space, of situations, even of “things”.

Grounding knowledge in experience is extremely challenging, but brings an equally extreme benefit. Representing knowledge in terms of experience enables it to be compared with experience. Received knowledge can be verified or disproved by this comparison. Existing knowledge can be tuned and new knowledge can be created (learned). The overall effect is that the AI agent may be able to take much more responsibility for maintaining and organizing its knowledge. This is a substantial benefit; the lack of such an ability is obstructing much AI research, as discussed earlier.
A related advantage is that grounded knowledge may be more useful. The primary use for knowledge is to aid planning or reasoning processes. Predictive knowledge is suited to planning processes based on repeated projection, such as state-space search, dynamic programming, and model-based reinforcement learning (Dyna, pri-sweep, LSTD). If A predicts B, and B predicts C, then it follows that A predicts C. If the final goal is to obtain some identified observation or observations, such as rewards, then predictive reasoning processes are generally suitable.

## 3. Questions and Answers

Modern philosophy of science tells us that any scientific theory must be empirically verifiable. It must make predictions about experiments that can be compared to measureable outcomes. We have been developing a similar view of knowledge—that the content of knowledge is a prediction about the measurable outcome of a way of behaving. The prediction can be divided into two parts, one specifying the question being asked and the other the answer offered by the prediction. The question is “What will be the measured value if I behaved this way and measured that?” An answer is a particular predicted value for the measurement which will be compared to what actually happens to assess the prediction’s accuracy. For example, a question roughly corresponding to “How much will it rain tomorrow” would be a procedure for waiting, identifying when tomorrow has begun, measuring the cumulative precipitation in centimeters, and ending when the end-of-day has been identified. The result based on this actual future will be a number such as 1.2 which can be compared to the answer offered by the prediction, say 1.1.

In this example, the future produces a result, the number 1.2, whose structure is similar to that of the answer, and one may be tempted to refer to the result as the “correct answer.” In general, however, there will be no identifiable correct answer that can be identified as arising from the question applied to the future. The idea of a correct answer is also misleading because it suggests an answer coming from the future, whereas we will consider answers always to be generated by histories. There may be one or more senses of best answers that could be generated, but always from a history, not a future.

Figure 3 shows how information flows between experience and the question and answer making up a prediction made at a particular time. Based on the history, the answer is formed and passed on to the question, which compares it with the future. Eventually, a measure of mismatch between answer and future is computed, called the loss. This process is repeated at each moment in time and for each prediction made at that time.

*[Figure 2: Information flow relationships between questions and answers, histories and futures.]*

Note that the question in this example is substantially more complex and substantial than its answer; this is typically the case. Note also that the question alone is not world knowledge. It does not say anything about the future unless matched with an answer.

For knowledge to be clear, the experiment and the measurement corresponding to the question must be specified unambiguously and in detail. We state this viewpoint as the explicit prediction manifesto:

> Every prediction is a question and an answer.
> Both the question and the answer must be explicit in the sense of being accessible to the AI agent, i.e., of being machine readable, interpretable, and usable.

The explicit prediction manifesto is a way of expressing the grand challenge of empirical knowledge representation in terms of questions and answers. If knowledge is in predictive form, then the predictions must be explicit in terms of observations and actions in order to meet the challenge.

It is useful to be more formal at this point. In general, a question is a loss function on futures with respect to a particular way of behaving. The way of behaving is formalized as a policy, a (possibly deterministic) mapping from $E \times O$ to probabilities of taking each action in $A$. The policy and the world together determine a future or probability distribution over futures. For a given space of possible answers $Y$, a question’s loss function is a map $q: E \times Y \rightarrow \Re^+$ from futures and answers to a non-negative number, the loss. A good answer is one with a small loss or small expected loss.

For example, in the example given above for “How much will it rain tomorrow”, the answer space is the non-negative real numbers, $Y = \Re^+$. Given a history $h \in E$, an answer $y(h)$ might be produced by a learned answer function $y : E \rightarrow Y$. Given a future $f \in E$, the loss function would examine it in detail to determine the time steps at which tomorrow is said to begin and end. Suppose the precipitation on each time step “in centimeters” is one component of the observation on that step. This component is summed between the start and end times to produce a correct answer $z(f) \in E$. Finally, $y(h)$ and $z(f)$ are compared to obtain, for example, a squared loss $q(f,y(h)) = (z(f)-y(h))^2$.

The interpretation in terms of “centimeters” in this example is purely for our benefit; the meaning of the answer is with respect to the measurement made by the question, irrespective of whatever interpretation we might place on it. Our approach is unusual in this respect. Usually in statistics and machine learning the focus is on calibrated measurements that accurately mirror some quantity that is meaningful to people. Here we focus on the meaning of the answer that has been explicitly and operationally defined by the question’s loss function. By accepting the mechanical interpretation as primary we become able to verify and maintain the accuracy of answers autonomously without human intervention.

A related way in which our approach is distinctive is that we will typically consider many questions and a great variety of questions. For example, to express the shape of an object alone requires many questions corresponding to all the ways the object can be turned and manipulated. In statistics and machine learning, on the other hand, it is common to consider only a single question. There may be a training set of inputs and outputs with no temporal structure, in which case the single question “what is the output for this input?” is so obvious that it needs little attention. Or there may be a time sequence but only a single question, such as “what will the next observation be?”

In these cases, in which there is only one questions, it is common to use the word “prediction” to refer just to answers. In machine learning, for example, the output of the learner—the answer—is often referred to as a prediction. It is important to realize that that sense of prediction—without the question—is much smaller than that which we are using here. Omitting the question is omitting much; the question part of a prediction is usually much larger and more complex than the answer part. For example, consider the question, “If I flipped this coin, with what probability would it come up heads?” The answer is simple; it’s a number, say 0.5, and it is straightforward to represent it in a machine. But how is the machine to represent the concepts of flipping, coin, and heads? Each of these are high-level abstractions corresponding to complex patterns of behavior and experience. Flipping is a complex, closed-loop motor procedure for balancing the coin on a finger, striking it with your thumb, then catching, turning, and slapping it onto the back of your hand. The meaning of “heads” is also a part of the question and is also complex. Heads is not an observation—a coin showing heads can look very different at different angles, distances, lightings and positions. We will treat this issue of abstraction later in the paper, but for now note that it must all be handled within the question, not the answer. Questions are complex, subtle things. They are the most important part of a prediction and selecting which ones to answer is one of the most important skills for an intelligent agent.

All that having been said, it is also important to note that predictive questions can also be simple. Perhaps the simplest question is “what will the next observation be,” (say with a cross-entropy loss measure). Or one might ask whether the third observation from now will be within some subset. If the observations are boolean we might ask whether the logical AND of the next two will be true. If they are numeric we might ask whether the square root of the sum of the next seven will be greater than 10, or whether the sum up to the next negative observation is greater than 100. Or one can ask simple questions about action dependencies. For example, we might ask what the next observation will be given that we take a particular next action, or a particular sequence of actions. In classical predictive state representations, the questions considered, called tests, ask for the probability that the next few observations will take particular values if the next few actions were to have particular values. Many of these questions (but not the last one) are meant as policy dependent. For example, if a question asks which of two observations will occur first, say death and taxes, then the answer may well depend on the policy for taking subsequent actions. These simple questions have in common that we can all see that they are well defined in terms of our minimal ontology—observations, actions, and time steps. We can also see how their complexity can be increased incrementally. The grand challenge asks how far this can be taken. Can a comparable clarity of grounding be attained for much larger, more abstract, and more complex concepts?

## 4. Abstract Concepts and Causal Variables

Questions and answers provide a formal language for addressing the grand challenge of grounding knowledge in experience, but do not in themselves directly address the greatest component challenge, that of abstracting from the particularities of low-level experience to human-level knowledge. Let us examine in detail a few steps from low-level experience to more abstract concepts. The first step might be to group together all situations that share the same observation. The term “situation” here must be further broken down because it is not one of our primitive concepts (observations, actions, or time steps). It must be reduced to these in order to be well-defined. What is meant by “situations” here is essentially time steps, as in all the time steps that share the same observation. With this definition, the concept of all such time steps is clear and explicit.

A further step toward abstraction is to define subsets of observations and group together all time steps with observations within the same subset. This is natural when observations have multiple components and the subsets are those observations with the same value for one of the components. Proceeding along the same lines, we can discuss situations with the same action, with the same action-observation combination, with the same recent history of observations and actions, or that fall within any subset of these. All of these might be called history-based concepts. The general case is to consider arbitrary sets of histories, $C \subset \{O \times A\}^*$. We define abstract history-based concepts to be sets such that $|C| = \infty$.

It is useful to generalize the idea of history-based concepts to that of causal variables—time sequences whose values depend only on preceding events. (A history-based concept corresponds to a binary causal variable.) Formally, the values of causal variable $v_t = v(h_t)$ are given by a (possibly stochastic) function $v : E \rightarrow Y$. As with concepts, we consider a causal variable to be abstract if and only if its value corresponds to an infinite set of possible histories. Formally, we define a causal variable to be abstract if and only if the preimage of every subset of $Y$ is infinite ($\forall C \subseteq Y, |\{e : v(e) \in C\}| = \infty$). One example of a causal variable is the time sequence of answers given by the answer function of a prediction. In this sense, answers are causal variables.

Abstract causal variables seem adequate and satisfactory to capture much of what we mean by abstractions. They capture the idea of representing situations in a variety of ways exposing potentially relevant similarities between time steps. They formally characterize the space of all abstract concepts. But it is not enough to just have abstractions; they must be good abstractions. The key remaining challenge is to identify or find abstract causal variables that are likely to be useful.

In this paper we pursue the hypothesis that non-redundant answers to predictive questions are likely to be useful abstractions. This hypothesis was first stated and tested by Rafols, Ring, Sutton, and Tanner (2005) in the context of predictive state representations. They stated it this way:

> “The predictive representations hypothesis holds that particularly good generalization will result from representing the state of the world in terms of predictions about possible future experience.”

This hypothesis is plausible if we take the ultimate goal to be to represent knowledge predictively. The hypothesis is not circular because there are multiple questions. The hypothesis is that the answer to one question might be a particularly good abstraction for answering a second question. An abstraction’s utility for one set of questions can perhaps act as a form of cross validation for its likely utility for other questions. If a representation would have generalized well in one context, then perhaps it will in another.

The hypothesis that answers to predictive questions are likely to make good abstractions begs the question of where the predictive questions come from. Fortunately, guidance as to likely pertinent questions is available from several directions. First, predictions are generally with respect to some causal variable of interest. Interesting causal variables include:

1.  Signals of intrinsic interest such as rewards, loud sounds, bright lights—signals that have been explicitly designated by evolution or designer as salient and likely to be important to the agent
2.  Signals that have been found to be associated with, or predictive of, signals already identified as being of interest (e.g., those of intrinsic salience mentioned in #1)
3.  Signals that can be predicted, that repay attempts to predict them with some increase in predictive accuracy, as opposed to say, random signals
4.  Signals that enable the control of other signals, particularly those identified as being of interest according to #1–#3

There is a fifth property making a causal variable interesting as a target for prediction that is more subtle and directly relevant to the development in this paper: the causal variable may itself be an answer to a predictive question. In other words, “what will be the value of this abstraction (causal variable) in the future (given some way of behaving)?” Questions about abstractions known to be useful would be considered particularly appealing.

The proposal, overall, is that useful abstractions for answering predictive questions can be found as answers to other predictive questions about useful abstractions. This is not circular reasoning, but rather an important form of compositionality: the ability to build new abstractions out of existing ones. It is a key property necessary for powerful representations of world knowledge.

If questions are to be about (future values of) abstractions, then what should those questions be? Recall that questions are conditional on a way of behaving – an experiment or policy. But which experiment? Guidance comes from how the predictions will be used, which will generally be as part of a planning (optimal decision making) process. Accordingly, we are particularly interested in questions about causal variables conditional on a way of behaving that optimizes the causal variables. The terminations of experiments can be selected in the same way.# Mind is About Information

## Rich Sutton
### November 19, 2001

What is the mind? Of course, "mind" is just a word, and we can mean anything we want by it. But if we examine the way we use the word, and think about the kinds of things we consider more mindful than others, I would argue that the idea of *choice* is the most important. We consider things to be more or less mindful to the extent that they appear to be making choices. To make a choice means to distinguish, and to create a difference. In this basic sense the mind is about *information*. Its essential function is to process bits into other bits. This position has two elements:

- Mind is Computational, not Material
- Mind is Purposive

### Mind is Computational, not Material

The idea that the mind's activities are best viewed as information processing, as *computation*, has become predominant in our sciences over the last 40 years. People do not doubt that minds have physical, material form, of course, either as brains or perhaps as computer hardware. But, as is particularly obvious in the latter case, the hardware is often unimportant. Is is how the information flows which matters.

I like to bring this idea down to our basest intuition. What things are more mindlike and less mindlike? A thermostat is slightly mindlike. It converts a gross physical quantity, the air temperature of your home, to a small deviation in a piece of metal, which tips a small lump of mercury which in turn triggers a fire in your furnace. Large physical events are reduced and processed as small ones, the physical is reduced to mere distinctions and processed as information. The sensors and effectors of our brains are essentially similar. Relatively powerful physical forces impinge on us, and our sensors convert them to tiny differences in nerve firings. These filter and are further processed until signals are sent to our muscles and there amplified into gross changes in our limbs and other large physical things. At all stages it is all physical, but inside our heads there are only small physical quantities that are easily altered and diverted as they interact with each other. This is what we mean by information processing. Information is not non-physical. It is a way of thinking about what is happening that is sometime much more revealing and useful than its physical properties.

Or so is one view, the view that takes a material physical reality as primary. The informational view of mind is just as compatible with alternative philosophical orientations. The one I most appreciate is that which takes the individual mind and its exchanging of information with the world as the primary and base activity. This is the so-called "buttons and lights" model, in which the mind is isolated behind an interface of output bits (buttons) and input bits (lights). In this view, the idea of the physical world is created by the mind so as to explain the pattern of input bits and how they respond to the output bits. This is a cartoon view, certainly, but a very clear one. There is no confusion about mind and body, material and ideal. There is just information, distinctions observed and differences made.

### Mind is Purposive

Implicit in the idea of choice, particularly as the essence of mindfulness, is some reason or purpose for making the choices. In fact it is difficult even to talk about choice without alluding to some purpose. One could say a rock "chooses" to do nothing, but only by suggesting that its purpose is to sit still. If a device generated decisions at random one would hesitate to say that it was "choosing." No, the whole idea of choice implies purpose, a reason for making the choice.

Purposiveness is at heart of mindfulness, and the heart of purposiveness is the varying of means to achieve fixed ends. William James in 1890 identified this as "the mark and criterion of mentality". He discussed an air bubble rising rising in water until trapped in an inverted jar, contrasting it with a frog, which may get trapped temporarily but keeps trying things until it finds a way around the jar. Varying means and fixed ends. In AI we call it generate and test. Or trial and error. Variation and selective survival. There are many names and many variations, but this idea is the essence of purpose, choice, and Mind.
# Verification, The Key to AI

## by Rich Sutton
### November 15, 2001

It is a bit unseemly for an AI researcher to claim to have a special insight or plan for how his field should proceed. If he has such, why doesn't he just pursue it and, if he is right, exhibit its special fruits? Without denying that, there is still a role for assessing and analyzing the field as a whole, for diagnosing the ills that repeatedly plague it, and to suggest general solutions.

The insight that I would claim to have is that the key to a successful AI is that it can tell for itself whether or not it is working correctly. At one level this is a pragmatic issue. If the AI can't tell for itself whether it is working properly, then some person has to make that assessment and make any necessary modifications. An AI that can assess itself may be able to make the modifications itself.

**The Verification Principle:**

> An AI system can create and maintain knowledge only to the extent that it can verify that knowledge itself.

Successful verification occurs in all search-based AI systems, such as planners, game-players, even genetic algorithms. Deep Blue, for example, produces a score for each of its possible moves through an extensive search. Its belief that a particular move is a good one is verified by the search tree that shows its inevitable production of a good position. These systems don't have to be told what choices to make; they can tell for themselves. Image trying to program a chess machine by telling it what kinds of moves to make in each kind of position. Many early chess programs were constructed in this way. The problem, of course, was that there were many different kinds of chess positions. And the more advice and rules for move selection given by programmers, the more complex the system became and the more unexpected interactions there were between rules. The programs became brittle and unreliable, requiring constant maintainence, and before long this whole approach lost out to the "brute force" searchers.

Although search-based planners verify at the move selection level, they typically cannot verify at other levels. For example, they often take their state-evaluation scoring function as given. Even Deep Blue cannot search to the end of the game and relies on a human-tuned position-scoring function that it does not assess on its own. A major strength of the champion backgammon program, TD-Gammon, is that it does assess and improve its own scoring function.

Another important level at which search-based planners are almost never subject to verification is that which specifies the outcomes of the moves, actions, or operators. In games such as chess with a limited number of legal moves we can easily imagine programming in the consequences of all of them accurately. But if we imagine planning in a broader AI context, then many of the allowed actions will not have their outcomes completely known. If I take the bagel to Leslie's office, will she be there? How long will it take to drive to work? Will I finish this report today? So many of the decisions we take every day have uncertain and changing effects. Nevertheless, modern AI systems almost never take this into account. They assume that all the action models will be entered accurately by hand, even though these may be most of the knowledge in or ever produced by the system.

Finally, let us make the same point about knowledge in general. Consider any AI system and the knowledge that it has. It may be an expert system or a large database like CYC. Or it may be a robot with knowledge of a building's layout, or knowledge about how to react in various situations. In all these cases we can ask if the AI system can verify its own knowledge, or whether it requires people to intervene to detect errors and unforeseen interactions, and make corrections. As long as the latter is the case we will never be able to build really large knowledge systems. They will always be brittle and unreliable, and limited in size to what people can monitor and understand themselves.

> "Never program anything bigger than your head"

And yet it is overwhelmingly the case that today's AI systems are *not* able to verify their own knowledge. Large ontologies and knowledge bases are built that are totally reliant on human construction and maintenance. "Birds have wings" they say, but of course they have no way of verifying this.
# The One-Step Trap (in AI Research)

## Rich Sutton
### Written up for X on July 18, 2024

The one-step trap is the common mistake of thinking that all or most of an AI agent's learned predictions can be one-step ones, with all longer-term predictions generated as needed by iterating the one-step predictions. The most important place where the trap arises is when the one-step predictions constitute a model of the world and of how it evolves over time. It is appealing to think that one can learn just a one-step transition model and then "roll it out" to predict all the longer-term consequences of a way of behaving. The one-step model is thought of as being analogous to physics, or to a realistic simulator.

The appeal of this mistake is that it contains a grain of truth: if all one-step predictions can be made with perfect accuracy, then they can be used to make all longer-term prediction with perfect accuracy. However, if the one-step predictions are not perfectly accurate, then all bets are off. In practice, iterating one-step predictions usually produces poor results. The one-step errors compound and accumulate into large errors in the long-term predictions. In addition, computing long-term predictions from one-step ones is prohibitively computationally complex. In a stochastic world, or for a stochastic policy, the future is not a single trajectory, but a tree of possibilities, each of which must be imagined and weighted by its probability. As a result, the computational complexity of computing a long-term prediction from one-step predictions is exponential in the length of the prediction, and thus generally infeasible.

The bottom line is that one-step models of the world are hopeless, yet extremely appealing, and are widely used in POMDPs, Bayesian analyses, control theory, and in compression theories of AI.

The solution, in my opinion, is to form temporally abstract models of the world using options and GVFs, as in the following references.

Sutton, R.S., Precup, D., Singh, S. (1999). Between MDPs and semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning. Artificial Intelligence 112:181-211.

Sutton, R. S., Modayil, J., Delp, M., Degris, T., Pilarski, P. M., White, A., Precup, D. (2011). Horde: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction. In Proceedings of the Tenth International Conference on Autonomous Agents and Multiagent Systems, Taipei, Taiwan.

Sutton, R. S., Machado, M. C., Holland, G. Z., Timbers, D. S. F., Tanner, B., & White, A. (2023). Reward-respecting subtasks for model-based reinforcement learning. Artificial Intelligence 324.
**CRAIG** <sup>00:07</sup>

> This is Craig Smith with a new podcast about artificial intelligence. This week I talk to Richard Sutton who literally wrote *the book* on reinforcement learning, the branch of artificial intelligence most likely to get us to the holy grail of human-level general Intelligence and beyond. Richard is one of the kindest, gentlest people I've ever interviewed - and I've interviewed a lot - a cross between Pete Seeger and Peter Higgs with maybe a little John Muir thrown in. We talked about how he came to be at the forefront of machine learning from his early days as a curious child to his single-minded search for colleagues of like mind. His temporal difference algorithm, *TD Lambda*, changed the course of machine learning and has become the standard model for reward learning in the brain. It also plays *Backgammon* better than humans. Richard spoke about the future of machine learning and his guarded prediction for when we might reach the singularity. I hope you find the conversation as fascinating as I did.

**CRAIG** <sup>01:25</sup>

> I wanted to start by asking you why you live in Alberta and how such a relatively remote place became a center for cutting edge technology that's changing the world.

**RICH** <sup>01:37</sup>

> The subarctic prairies of Canada. It's one of the more northern cities in Canada, actually. Jonathan Schaeffer. John Schaeffer is the guy who wrote the first computer program that defeated a world champion in any major game, which was in *Checkers*. He was there before I was. I've been there 15 and a half years now. He will be humble. He'll probably blame it on me, but a lot of it has to do with him and his coming. And it was one of the first Computer Science departments in Canada, was in Edmonton at the University of Alberta, but how it became strong in AI originally - because I was attracted there because of its strength as well as other things.

**RICH** <sup>02:11</sup>

> My family traveled around a lot when I was young. I was born in Toledo, Ohio and I went through a series of places where I spent like a year each. We lived in New York for a while and Pennsylvania.

**RICH** <sup>02:23</sup>

> Then we lived in Arizona and then we lived somewhere else in Arizona. And then finally we moved to the suburbs of Chicago - Oakbrook. So I was there from seven to 17. But I think it was formative as I was very young, we kept moving around. I don't feel a real strong association to any particular place. And maybe that's one reason why it was easy for me to move to Canada.

**CRAIG** <sup>02:43</sup>

> Well, what did your father do that kept you guys moving or your mother, If it was your mother?

**RICH** <sup>02:47</sup>

> My father was a business person. He was an executive. So yeah, he was moving usually to the next new better position.

**CRAIG** <sup>02:55</sup>

> What company was that?

**RICH** <sup>02:56</sup>

> In the end he was in Interlake Steel. He did mergers and acquisitions.

**CRAIG** <sup>03:00</sup>

> So you were not brought up in anything related to science?

**RICH** <sup>03:04</sup>

> Both of my parents had graduate degrees, master's degrees. My mother was an English teacher and they met in college at Swarthmore.

**CRAIG** <sup>03:11</sup>

> So where did the interest in science, was that from high school or did that start at university?

**RICH** <sup>03:18</sup>

> I was in a good high school. I was a pretty good student. I was a little more introspective, I think, than most people. So I was sort of wondering about what it is that we are and how we make sense of our world. You know, like I used to lie down on the couch in my home and stare up at the ceiling, which had a pattern sort of thing. And then your eyes would cross, you know, it seemed like it was closer. You know what I mean? And all those things, you know, make you think. It isn't like there's just a world out there and you see it. It's like something is going on and you're interpreting it and how does that happen?

**RICH** <sup>03:47</sup>

> So I was always wondering about that 'cause it's just a natural thing to wonder about if you're an introspective kid. But I remember in grade school and, alright, what are you going to do? I think first I wanted to be an architect for some reason, then I wanted to be a teacher then I wanted to do science. And before I was out of grade school, this is in, it would be 69, you know, there's the talk of computers and you know, the electronic brain. And that was exciting. And what could that mean? I get to high school, we get to use computers. And so I'm taking a course. We learned `Fortran`. And I'm sitting there saying, where's the electronic brain? You know, this thing, you have to tell every little thing and you have to put the commas in the right places and there's no brain here.

**RICH** <sup>04:31</sup>

> In fact, it only does exactly what you tell it. There's no way this could be like a brain. And yet at the same time, you know, somehow there are brains and they must be machines. So it definitely struck me that, you know, this is impossible that this machine that only does what it's told could be a mind. And yet at the same time as I thought about it, it must be that there's some way a machine can be a mind. And so it was these two things. This impossible thing. And yet somehow it must be true. So that's what hooked me, that challenge. And all through high school, I remember I was programming an IBM machine outside of class. I think they called it Explorers. It's kind of like a modified version of Boy Scouts.

**CRAIG** <sup>05:10</sup>

> Right. It's the next level up.

**RICH** <sup>05:12</sup>

> Yeah, and we had access to some company and they had a big IBM machine and so I was trying to program a model of the cerebral cortex on the machine in Explorers. So it was like neural nets. It was really just like neural nets.

**RICH** <sup>05:24</sup>

> So I was into all this stuff. It seemed to me obvious that the mind was a learning machine. So I applied to colleges and I wanted to learn about this and I wrote to Marvin Minsky, when I was a kid. I still have the letter that he sent me back. He sent me back that, yeah. Excellent. Can you imagine that. I asked him basically, you know, I'm really interested in AI stuff. What should I study? What should I major in? He said, it doesn't really matter much what you study. You could study math or you could study the brain or computers or whatever, but it was important that you learned math, something like that. Anyway, he sort of gave me permission to study whatever seemed important, as long as I kept my focus.

**RICH** <sup>06:05</sup>

> I did apply to many places. I could know even then it was Stanford and MIT and Carnegie Mellon were the big three in artificial intelligence, but it was, you know, computer science. Computers were still new and they didn't have an undergraduate major in computer science so you had to take something else. And so I took psychology because that made sense to me and I went to Stanford because Stanford had a really good psychology program.

**CRAIG** <sup>06:27</sup>

> Neuropsychology or?

**RICH** <sup>06:29</sup>

> So, of course, psychology involves both and I took courses in both and I quickly became disenchanted with the sort of neural psychologists because it was so inconclusive and you can learn an infinite amount of stuff and they, they were still arguing inconclusively about how it works. I remember distinctly sending a term paper in to my neuropsychology class to, I think it was to Karl Pribram. Anyway, it came back marked, saying, oh no, that's not the way it works.

**RICH** <sup>06:54</sup>

> And I thought I had a good argument that that might be the way it works, but it was just so arbitrary that someone could say what works. Anyway, I was the kind of kid that if you got, if I got a D minus, it was crushing. It wasn't really crushing but it was disappointing. And so I resolved right then that this is not fertile ground to go into the neuro side where on the other hand, the behavioral side, very rich. You could learn all kinds of good things. The level of scholarship in behavioral psychology was very high. They would think carefully about their theories. They would test them, they would do all the right control groups. They would say is there some other interpretation of these results? And they would debate it and it just seemed like a really high level - experimental behavioral psychology seemed excellent to me and I didn't know that it was bad.

**RICH** <sup>07:38</sup>

> I didn't know that it was disgraced and horrible. I'm just a young person and I'm just saying this is cool and I still think it's cool even if it's disgraced. And I think it's just a really good example of how fads and impressions can be highly influential and not necessarily be correct. They have to be always be re evaluated. So, but behaviorism did suffer and it's gone almost extinct now. It's recovering a little bit within the biological and neuroscience parts. But behaviorist, experimental psychology has almost gone extinct. Just a few people left and it's a great shame. Fortunately, most of the basic experiments have already been done and then more is taking place within neuroscience. But I learned enormous amounts about it. I found a professor at Stanford who was emeritus, but wasn't gone. And I learned from him and I learned from the old books and that material became the basis for my contributions in artificial intelligence.

**RICH** <sup>08:44</sup>

> So I'm known for this *temporal difference learning*. *Temporal difference learning*, it's quite clear that something like that is going on just from the behavioral studies. And unlike all the other guys in AI, I'd read all those behavioral studies and the theories that have arisen out of them. So I'm at Stanford, I'm taking psychology. Psychology is an easy major, let's face it. And I'm learning lots about that with Piaget as well as the experimental things and the perceptual things and the cognitive things. But really I'm thinking I'm an AI guy and you can't major in computer science but you can take computer science courses, you can take all of the AI courses.

**RICH** <sup>09:18</sup>

> Stanford was a world leader, one of the top three in AI at the time, and so I'm getting fully an AI education even though my major will be psychology. I'm, as much as an undergraduate can, I am getting an education in AI and I would go to the AI lab and write programs. They had the earliest robot arms there and I would program them and it was great because I was allowed to program the research arms and like the first day I broke it. I thought they would kill me and kick me out forever, but they said, oh, that happens all the time and they just fixed it and they let me keep programming it. Even as an undergraduate in the summer, I'd ride my bicycle up to the DC Power building where the AI lab was then. And remember this is middle of the 70s and machine learning is nowhere - totally unpopular. No one is going to do anything with learning in it. So I am not quite in sync 'cause I'm thinking learning is super important and that's nowhere to be seen.

**RICH** <sup>10:12</sup>

> But I'm getting a lot of learning from my psychology and I'm going to the library where I could read everything, read all the AI books because they were like one row and that was it. I would do that and I'd be desperately searching, you know for the learning parts and you know I found out that I was just too late because learning was really popular in the 60s and then it went off into other fields or died off in AI. There was no faculty member that was really interested in learning. I found a faculty member who was a vision guy, Binford, and he was the guy who said I could use the machines at the AI lab. So the founder of AI, the guy who coined the term, John McCarthy, was at Stanford. And back then what they had was a drum kind of disk and it's rotating around and basically as it rotates around it generates the computer screens, all the computer screens.

**RICH** <sup>11:02</sup>

> So there'd be only like 30 I think. And this is a hard limit. You've got 30 things on the drum and so they can show 30 screens of stuff. They have more than 30 terminals, but only 30 can be active at once. And so when you go in, first you grabbed one of these 30 and then you do your stuff. But at the peak of the day they would run out. And so one time John McCarthy comes in and there are none left and so he wants to kick me off because I'm just the undergraduate. He almost succeeds. But then Binford saves the day. He says, no, no, no, he's a real person. Suck eggs, McCarthy, all people are equal. You are the creator of the field and he's an undergraduate, but he was here first so. So thank you Binford, Tom Binford. Ultimately I did my senior undergraduate thesis on a neural net model and to supervise it, I found somebody, a biologist, he was the guy who did modeling like mathematical modeling of biological phenomenon.

**RICH** <sup>11:54</sup>

> So it's going to sort of fit. But you know, the whole thing as a way of doing AI didn't make much sense to him. And so I was again totally out on my own I guess. So I did all that. And then when I was in the library reading all those things, I was always looking for guys doing learning. I like to think in retrospect that I was looking for something that was real learning that was like *reinforcement learning* because *reinforcement learning* is an obvious idea if you study psychology. Because there are two basic kinds of learning: *Pavlovian conditioning* and *instrumental* or *operant conditioning*. *Pavlovian conditioning* is for like ring the bell and then you give the dog a steak. And after a while, just from you ring the bell, he salivates showing that he anticipates the steak's arrival. So it's a kind of prediction learning.

**RICH** <sup>12:33</sup>

> And then there's this behavioral revealing of the prediction. The salivation reveals that the animal's predicting. For other things, you may predict them, but it's not revealed in your behavior. And so typically this kind of prediction learning has been done with things being predicted that evoke responses so that then you can see the prediction or some things related to the prediction.

**RICH** <sup>12:54</sup>

> And then there's also control learning and control learning is called *instrumental conditioning* or *operant conditioning* - at least those two names - where you're changing your behavior to cause something to happen. In *Pavlovian conditioning*, your salivation doesn't influence what happens. Whereas the canonical operant conditioning is, you know the rat presses a bar and then gets a food pellet. The idea of the pressing the bar's instrumental in getting the reward. So that's really the idea of *reinforcement learning*. It's modeled after this obvious thing that animals and people do all the time.

**RICH** <sup>13:25</sup>

> How can that not exist as an engineering activity? It's still an oddity. I think somebody should have done it, like mathematical psychologists maybe, or maybe the engineers, or maybe the AI guys or maybe the mathematicians. Somewhere, you know, in all of the things that all the people have ever studied all over the world, surely someone else would have studied this thing that animals and people are doing, a very common-sense thing. And that's what was the mystery. So there's this mystery, but to tell the story sequentially. Last year, third year college, I come across this obscure tech report from this fellow Harry Klopf, A. Harry Klopf, K L O P F. He has a theory that there should be neurons in a neural network that have goals, that want something. And so he's thinking outside the box and out of nowhere he's saying, well, there was neural networks before, but none of those adaptive networks that were studied earlier, none of those learning systems that were studied earlier actually wanted something.

**RICH** <sup>14:25</sup>

> They were told what they should do, but they didn't like vary their behavior in order to get something, they didn't want anything except to drive their error to zero. And if you think about supervised learning, you don't affect the world. You're told the right label and then you do emit a label, your label doesn't affect the world. It's compared to the label, but it doesn't influence the next label. It influences whether you're right or wrong, but it doesn't influence the world. So then, you know, there's this question as you look back at the really early stuff before the 70s. Did they do learning where the actions influenced the world or did they not? And they would use the language of trial and error and maybe they were thinking a little bit about affecting the world. But then over time as they came to formalize what they were doing and do it more clearly, they formalized it into supervised learning and they left out the effect on the world.

**RICH** <sup>15:14</sup>

> And so Harry Klopf was the one guy who saw this and was always complaining to himself, you know, there's something missing here. And he was the guy who saw that and Cambridge Research Labs - that's where he was when he came to his resolution, wrote his report, then he went into the Air Force. He was a civilian researcher in the Air Force. And I'm finding his report. And so I write this guy and said, you're the only guy I can find who is talking about the kind of learning that seems important. And so he writes me back, he says, oh, that's interesting. That's what I've been saying, but no one else sees it. Maybe we should talk. So I don't know. He came to California maybe for some other reason but we arranged to meet and I met him there and that was cool. And he told me that he was funding a program at the University of Massachusetts to begin soon, or was beginning soon on this sort of thing.

**RICH** <sup>15:59</sup>

> So as I said, he had gone into the Air Force. No one was interested in doing proper learning, learning that gets something. He told me about this thing at the University of Massachusetts and so I ended up applying to go to graduate school at the University of Massachusetts. And I say it a little bit that way because it was sort of a year early, like, you know, I met this guy, learned about this program. I thought that'd be a really cool place to go to. I think I told my mom and she says, well, why don't you go there? You know, like now\! You know, because I added up my credits and I figured out that I could, if I took a couple of extra courses or something in the summer, I could finish at the end and then the very next September I could be there. And I did that. And that's what happened.

**CRAIG** <sup>16:41</sup>

> This is for -

**RICH** <sup>16:42</sup>

> a PhD program. So I ended up going there.

**CRAIG** <sup>16:44</sup>

> This is at UMass.

**RICH** <sup>16:47</sup>

> Yeah, University of Massachusetts where Harry Klopf, as now in the Air Force had gotten to a situation where he could get funding for the University of Massachusetts to do research in this area. I don't think it was his main job to fund programs in general, but he definitely set up this program to do the adaptive networks research. Andy Barto and I, our task was to figure out what these ideas really were and how to think about them and how they compared to what had gone before.

**CRAIG** <sup>17:19</sup>

> So what was the breakthrough then?

**RICH** <sup>17:22</sup>

> The breakthrough was this was our task to figure out what he was saying. What's new because there are several ideas mixed in. There were ideas related to *temporal difference learning*. There were ideas related to just raw *reinforcement learning*, a system that wants something and tries to influence the world. But these ideas were all kind of mixed together and so as Andy Barto and I teased them apart and as we learned more and more about what has been going on in all the various potentially related fields and continued to find gaps. The breakthrough was saying, well maybe just this first part just as part of an adaptive unit, a neuron-like unit that varies its output in order to influence its reward-like signal that's coming from the world. Maybe just that. You leave out the complicated confusing parts and just do this. Like, we can't find that having already been done in the literature so maybe we should make that a thing.

**RICH** <sup>18:12</sup>

> The name *reinforcement learning* in the literature - there are things that come close and that you might call *reinforcement learning*. We wanted to respect the existing name and the existing work rather than pretend to invent something new. There are a few antecedents so we continued to use the name. And I remember that was a real decision. It's a somewhat old sounding name, *reinforcement learning*. It's not really an attractive name. Maybe we should have called it, you know, something, 'dynamic learning' or ... So we went with that and that was the breakthrough and we wrote some papers but it was a breakthrough for us and a slow breakthrough for the field. What we were saying to the field is there is a problem that you have overlooked. Okay, there's pattern recognition, supervised learning, it would go by many names like pattern recognition and trial and error. The perceptron is supervised, all these things.

**RICH** <sup>19:02</sup>

> So we're not giving them a better solution to pattern recognition. If you do that, you will be recognized quickly. But if you say, no, I'm not solving your problem better. I'm saying you've been working on the wrong problem, there's this different problem that is interesting and we have some algorithms for this different problem, new problem. That's the hard road. Whenever you bring in a new problem, it's harder to get people excited about it. And we're definitely going through this long period of saying this is different than supervised learning. Because they're going to say, okay, you're doing learning and you're talking about errors. We know how to do trial and error learning. It was all about making the claim and substantiating the claim that this kind of learning is different from the regular kind of learning, the supervised learning.

**CRAIG** <sup>19:41</sup>

> Are you using neural nets at this point?

**RICH** <sup>19:44</sup>

> Yes. So the program that Harry Klopf funded at the University of Massachusetts in 1977 was a program to study adaptive networks, which means networks of adaptive units all connected together.

**RICH** <sup>20:00</sup>

> Maybe it's a better name than neural networks actually because the networks are not made of neurons. They're not biological. They are new units, and so it was exactly the same idea. I was also doing that kind of modeling even as an undergraduate, but it was out of fashion and Harry Klopf comes with the Air Force money, comes to the University of Massachusetts. They had several professors that were interested in in what they call brain theory, became neural networks. Brain theory and systems neuroscience. And Massachusetts was not an arbitrary choice to do this, it was the natural choice. They had a program in cybernetics, which is brain theory and AI. Brain theory was not a widely used buzzword and it still isn't, but they called it brain theory and artificial intelligence. The current wave of neural networks is the third wave. And the second wave was in the late eighties and the first wave was in the 60s. And so it was in the 60s neural networks were popular, well learning machines were really popular.

**RICH** <sup>20:56</sup>

> And then when I was in school that was at its bottom. And so Harry Klopf, when he was saying no, we should be doing neural networks, adaptive networks, and they should have units that want things. that is totally out of sync because all learning is out of favor. So UMass, the University of Massachusetts has to be given credit that they were also interested in the somewhat unpopular topic. So maybe they were like the leading edge of the recovery. But when Harry Klopf, this odd guy from the Air Force comes to them and says, you know, I want you to work on this, I'll give you a bunch of money to work on this. And it's not military funding, it's from the Air Force, but it's totally basic research, they are going to say yes. They may not really believe in it, but they're going to say yes.

**RICH** <sup>21:39</sup>

> So there are three professors, Michael Arbib, who is the brain theory guy, cybernetics guy, and Bill Kilmer and Nico Spinelli. So they are going to say yes, but at the same time they're not really into it. You know, it is unpopular and they're not really into it. So they find Andy Barto, Andy Barto is a professor at one of the SUNY's but he's not really happy. So he accepts this demotion from a professor to a post doc, but it's to work on this exciting problem. And he comes there and then within a year I arrive and Andy is really running the whole program. The professors, you know, they're leaving it all to him and he and I end up, you know, working on it and figuring it out.

**CRAIG** <sup>22:17</sup>

> Did you spend the bulk of your career then in UMass?

**RICH** <sup>22:22</sup>

> Graduate school at UMass, one year of postdoc, then I went to GTE Laboratories in Waltham, Massachusetts and was there for nine years doing fundamental research in AI and there was an *AI winter* and they laid lots of people off.

**RICH** <sup>22:36</sup>

> I ended up just quitting and I go home and Andy and I started writing *the book*. I become a research scientist, kind of research professor at the University of Massachusetts. Sort of a distance relationship. I go there a couple of days a week cause I live a couple hours away. So I spent like four years writing *the book*. Part of which time I'm a research faculty at UMass. So basically living out the winter, the *AI winter*. And AI recovers a bit and it was that year in '99, '98 '99, and I go work at AT\&T Labs because they're ready to do fundamental research in AI. And by then *the book* was out, the first edition of *the book* was out and I was coming back a little bit, sort of known for this reinforcement thing. There was a long period there where Andy and I were talking about *reinforcement learning* and trying to define what it is, trying to make progress on it. *Temporal difference learning*. And so you know, that's where we sort of became known as the guys that were promoting this thing and it was gradually becoming realized that it was important.

**RICH** <sup>23:35</sup>

> *Temporal difference learning* in 1988, *Q learning* 1989 and *TD Gammon*, I think it was 1992 where he had just applied *temporal difference learning* and a multilayer neural network to solve *Backgammon* at the world champion level. And so that got a lot of attention and he was using my *temporal difference learning* algorithm, *TD Lambda*. But we know we were just slowly making progress on *reinforcement learning* and how to explain it, how to make it work.

**RICH** <sup>24:05</sup>

> *Temporal difference learning* is a perfectly natural, ordinary idea. You know, I like to say it's learning a guess from a guess. And it's like okay, I think I'm winning this game. I think I have a 0.6 chance of winning. That's my guess. And I make a move. The other guy makes a move. I make a move. The other guy makes a moves and I say, Oh wait no, I'm in trouble now. So you haven't lost the game. Now you're guessing you're might lose. Okay. So now let's say I estimate my probably of winning as 0.3. So at this one point you said it was 0.6 now you think it's 0.3, you're probably going to lose. So you could say oh that 0.6 that was a guess. Now I think that was wrong 'cause 'cause now I'm thinking of 0.3 and there are only two moves to go so it probably wasn't really 0.6. Either that or I made some bad moves in those two moves. Okay, so you modify your first guess based on the later guess and then your 0.3 you know you continue, you make a guess on every move.

**RICH** <sup>24:56</sup>

> You're playing this game. You're always guessing. Are you winning or are you losing. And so each time it changes. You should say the earlier guess should be more like the later guess. That's *temporal difference learning*. The thing that's really cool here is no one had to tell us what the real answer was. We just waited a couple of moves and now we see how it's changed. Or wait, one move and see how it has changed. And then you keep waiting and finally you do win or you do lose and you take that last guess, maybe that last guess is pretty close to whatever the actual outcome is. But you have a final temporal difference. You have all these temporal differences along the way and the last temporal difference is the difference between the last guess and what actually happens, which grounds it all up.

**RICH** <sup>25:33</sup>

> So you're learning guesses from other guesses. Sounds like it could be circular, it could be undefined, but the very last guess is connected to the actual outcome. Works its way back and makes your guesses correct. This is how *AlphaGo* works. *AlphaZero* works. Learning without teachers. This is why *temporal difference learning* is important. Why *reinforcement learning* is important because you don't have to have a teacher. You do things and then it works out well or poorly. You get your reward. It doesn't tell you what you should've done. It tells you what you did do was this good. You do something and I say seven and you say seven? Okay, what does that mean? You don't know. You don't know if you could have done something else and got an eight. So it's unlike supervised learning. Supervised learning tells you what you should have said. In supervised learning, the feedback instructs you as to what you should've done. In *reinforcement learning* the feedback is a reward and it just evaluates what you did. What you did is a seven and you have to figure out if seven is good or bad - If there's something you could've done better. So evaluation versus instruction, is the fundamental difference. It's much easier to learn from instruction. School is a good example. Most learning is not from school, it's from life.

**CRAIG** <sup>26:49</sup>

> At this point, supervised learning has been a pretty well explored…

**RICH** <sup>26:55</sup>

> [Laughs] Just laughing cause that's what we felt at the time. There's been lots of supervised learning, lots of pattern recognition. There's been lots of systems identification. There's been lots of the supervised thing. And they're just doing curlicues and they're dotting the last i's and how much longer can this go on? Right? It's time that we do something new. We should do *reinforcement learning*. And that was, you know, 30 years ago or something. It's still just as true today. It still seems like supervised learning, we should be done with it by now. But it's still the hottest thing. But we're not struggling for oxygen, you know. It is the hottest thing, but there is oxygen for other things like *reinforcement learning*.

**RICH** <sup>27:33</sup>

> But as people become more concerned about operating in the real world and getting beyond the constraints of labeled data, it seems like they're looking increasingly towards *reinforcement learning* or unsupervised learning.

**RICH** <sup>27:47</sup>

> *Reinforcement learning* involves taking action. Supervised learning is not taking action because the choices, the outputs of a supervised learning system don't go to the world. They don't influence the world. They're just correct or not correct according to their equaling a correct human provided label. So I started by talking about learning that influences the world and that's what's potentially scary and also potentially powerful.

**CRAIG** <sup>28:12</sup>

> Where does it go from here? I mean, *reinforcement learning*.

**RICH** <sup>28:16</sup>

> Next step is to understand the world and the way the world works and be able to work with that. *AlphaZero* gets its knowledge of the world from the rules of the game. So you don't have to learn it. If we can do the same kind of planning and reasoning that *AlphaZero* does, but with a model of the world, which would have to be learned, not coming from the rules of the game, then you would have something more like a real AI.

**CRAIG** <sup>28:42</sup>

> Is there a generalization?

**RICH** <sup>28:43</sup>

> They all involve generalization. Yup. So the opposite of generalization would be that you'd have to learn about each situation distinctly. I could learn about this situation and if it exactly comes back again, I can recall what I learned. But if there's any differences, then I'm generalizing from the old situation to the new situation. So obviously you have to generalize. The latest and greatest method to generalize, it's always been generalization in the networks. Enable you to generalize. So *deep learning* is really the most advanced method for generalizing. You may have heard the term 'function approximator'. I may use the term *function approximator*. They are the *function approximator*. They approximate the function, in this case the function is a common function. Is it a decision function. If you wanted to generalize, you might want to use some kind of neural network.

**CRAIG** <sup>29:36</sup>

> When did you move to Alberta? And why did you move to Alberta?

**RICH** <sup>29:38</sup>

> When I was at AT\&T, I was diagnosed with cancer and I was slated to die. So I wrestled with cancer for about five years.

**CRAIG** <sup>29:47</sup>

> What kind of cancer?

**RICH** <sup>29:48</sup>

> A melanoma. Melanoma, as you may know, it's one of the worst kinds. Very low... By the time they found it had already spread beyond the sites, was already metastasized. Once it has metastasized there's only like a 2% chance of real survival. And so we did all kinds of aggressive treatment things and I like had four miraculous recoveries and then four unfortunate recurrences and it was a big long thing. In the midst of all that, another winter started and I almost didn't care 'cause I was already dying. So another *AI winter* was not really that much of a concern to me at the time.

**RICH** <sup>30:24</sup>

> They lay off all the AT\&T guys, the machine learning guys. So I'm unemployed and expecting to die of cancer, but I am having one of my miraculous remissions. So it's going on long enough, you know, how long can you go on just being set to die and going through different treatments when the cancer comes back. After a while you think, well if this, this is dragging on, I might as well try to do something again and take a job. So I applied for some jobs, but it was still kind of a winter, there weren't that many jobs and besides, I really am expected with very high probability to die from this cancer. So it's totally a surreal situation, you see. What Alberta did right is they made this opportunity for me. They made me a really nice situation to tempt me with and it was like, well, I'm probably dying but I might as well do this while I'm dying.

**RICH** <sup>31:18</sup>

> So I say there's three reasons I went to Alberta. The position - the position was very good. It was an opportunity to be a professor, have the funding taken care of, step right into a tenured, fancy professordom, and the people, because the department was very good in Ai. Some top people like Jonathan Schaeffer I mentioned and Rob Holte and Russ Greiner and they were bringing in some other machine learning people at the same time. They were bringing Mike Bowling and Dale Schuurmans and they did actually all arrive at the same time as me. So and the third P is politics because you know the US was invading Iraq in 2003 at least about the time when I was making decisions about what to do. So it all seemed very surreal. Finally I went there to accept the job and all that takes a while. And in the meantime, you know, the cancer's coming back and I'm getting more extreme treatments and by the time I set off for Alberta, momentarily in remission, and I go there and the first term, you know, and then it comes back again and it looks like I'm dying again. But then you know, miraculously the fourth time or the fifth time it works and I'm alive. One of the tumors was in my brain, in the motor cortex area in the white matter. So it affects my side, very similar to a stroke.

**CRAIG** <sup>32:40</sup>

> What do you wish people understood about *reinforcement learning* or about your work?

**RICH** <sup>32:44</sup>

> The main thing to realize is it's not a bizarre artificial alien thing. Ai Is, it's really about the mind and people trying to figure out how it works. What are it's computational principles. It's a really, a very human centered thing. Maybe we will augment ourselves. Maybe we'll have better memories as we get older and we will, maybe we'll be able to remember the names of all the people we've met better. People will be augmented by Ai. That will be the center of mass. And that's what I want people to know that this activity is more than anything, it's just trying to understand what thought is and how it is that we understand the world. It's a classic humanities topic. What Plato was concerned with. You know, what is, what is a person, what is good, what does all these things mean?

**RICH** <sup>33:28</sup>

> It's not just a technical thing.

**CRAIG** <sup>33:30</sup>

> Is there an expectation as *reinforcement learning* can generalize more and more where you create a system that learns in the environment without having to label data,

**RICH** <sup>33:43</sup>

> Yep. But that's always been a goal of a certain section of the AI community.

**CRAIG** <sup>33:48</sup>

> Yeah. Is that one of your hopes with *reinforcement learning*?

**RICH** <sup>33:51</sup>

> Oh, yeah. That's certainly, I'm in that segment of the community. Yeah.

**CRAIG** <sup>33:55</sup>

> How far away do you think we are from creating systems that can learn on their own?

**RICH** <sup>34:01</sup>

> Key, and it's challenging, is to phrase the question. I mean, we do have systems that can learn on their own already. *AlphaZero* can learn on its own and learn in a very open-ended fashion about *Chess* and about *Go*.

**CRAIG** <sup>34:14</sup>

> Right, in very constricted domains.

**RICH** <sup>34:15</sup>

> What I would urge you to think about is what's special about those is we have the model, the model is given by the rules of the game.

**RICH** <sup>34:23</sup>

> That domain is quite large. It's all *Chess* positions and all *Go* positions. It's that we're not able to learn that model and then plan with it. That's what makes it small. It makes it narrow. So we're not quite sure what the question is - either of us - but I do have an answer, nevertheless. And my answer is stochastic. So the median is 2040 and the 25% probability is 2030. Basically it comes from the idea that by 2030 we should have the hardware capability that if we knew how to do it, we could do it, but probably we won't know how to do it yet. So give us another 10 years so the guys like me can think of the algorithms, you know, because once you have the hardware there's going to be that much more pressure because if you can get the right algorithm then you can do it, right.

**RICH** <sup>35:08</sup>

> Maybe now even if you had the right algorithm, you couldn't really do it. But at the point when there's enough computer power to do it, there's a great incentive to find that algorithm. 50% 2040 10% chance, we'll never figure it out because it probably means we've blown ourselves up. But the median is 2040. So if you think about that, 25% by 2030, 50% 2040 and then it tails off into the future. It's really a very broad spectrum. That's not a very daring prediction, you know. It's hard for it be seriously wrong. I mean, we can reach 2040 and you need to say, well, Rich. it isn't here yet. And I'll say, well, I said it's 50/50 before and after 2040 you know, that's literally what I'm saying. So it's going to be hard for me to be proved wrong before I die. But, uh, I think that's all appropriate.

**RICH** <sup>35:54</sup>

> We don't know. But I don't agree with those who think it's going to be hundreds of years. I think it will be decades. Yeah, I'd be surprised it was more than 30 years.

**CRAIG** <sup>36:01</sup>

> And we're talking at this point about what people call *artificial general intelligence*.

**RICH** <sup>36:07</sup>

> It's exactly the question that you couldn't formulate, so let me, how would I say it? I don't like the term because AI has always intended to be general in general intelligence. Really the term *artificial general intelligence* is meant to be an insult to the field of AI. It's saying that the field of AI, it hasn't been doing general intelligence. They've just been doing narrow intelligence and it's a really, a little bit of a of a stinging insult because it's partly true.

**RICH** <sup>36:36</sup>

> But anyway, I don't like the sort of snarky insult of *AGI*. What do, what do we mean? We mean I guess I'm good with human-level intelligence as a rough statement. We'll probably be surpassing human level in some ways and not as good in other ways. It's a very rough thing, but I'm totally comfortable with that in part because it doesn't matter that it's not a point if the prediction is so spread out anyway, so exactly what it means at any point in time is not so important. But when we would roughly say as that time when we have succeeded in creating through technology systems whose intellectual abilities surpass those of current humans in roughly all ways - intellectual capabilities. So I guess that works pretty well. Maybe not. Yeah, we shouldn't emphasize the 'all.' But purely intellectual activities, surpass those of current humans. That's all right.

**CRAIG** <sup>37:32</sup>

> That's all for this week. I want to thank Richard for his precious time. For those of you who want to go into greater depth about the things we talked about today, you can find a transcript of this show in the program notes along with a link to our Eye on AI newsletters. Let us know whether you find the podcast interesting or useful and whether you have any suggestions about how we can improve. The singularity may not be near, but AI is about to change your world. So pay attention.
 # Advice for writing peer reviews

## Rich Sutton

Some advice about writing a review. In my view, the author is king here. The author is the one doing the real work and the success of any meeting or journal depends on attracting good authors. So be respectful to them, while giving your best analysis.

An ideal review goes as follows.

### 1. The introduction
Summarize the paper in a few sentences. Be neutral, but be sure to include the perspective from which the work might be a good paper. Say what the paper claims to do or show. This section is for the editor and the author. Help the editor understand the paper and show that you as reviewer understand the paper and have some perspective about what makes an acceptable paper.

### 2. The decision
Give your overall assessment in a few sentences. This includes a clear recommendation for accept/reject. Give the reason for the decision in general terms. e.g., there are flaws in the experimental design which make it impossible to assess the new ideas. or, the authors are not aware of some prior work, and do not extend it in any way. Or, although the experiments are not completely clear, the idea is novel and appealing, and there is some meaningful test of it. Or, the contribution is very minor, plus the presentation is poor, so must recommend rejection. Hopefully you will have many more positive things to say, and will recommend accepting one. The bottom line is: does this paper make a contribution? It should be possible for the editor to read no further than this if he chooses. If there is agreement among the reviewers, this section will be enough for him to write the letter back to the author (or summary review).

### 3. The argument
Provide the substance that details and backs up your assessment given in 2. If there are flaws in the experiment, describe them here (not in 2). If there are presentation problems, detail and illustrate them here. In this section you are basically defending your decision in 2. The author and other reviewers are your target audience here. The editor will read this section if there is disagreement among the reviewers.

### 4. The denouement
Suggestions for improving the paper. It is important that these are suggestions, advice to the author, not reasons for the decision described above. The substance of the review, the decision, is over at this point. Now you are just being helpful. You can make useful suggests whatever the decision was on the paper.

### 5. The details
I find it useful to save until the end the list of tiny things. Typos, unclear sentences, etc.

BTW, if you say they missed some literature, provide a full citation to the work.

If you don't accept a paper, make a clear distinction between changes that would be required for acceptance (for the paper to make a contribution) and which would just make the paper better in your opinion. Authors hate it when a reviewer seems to reject because the paper was not written the reviewer's way.
# Text of Rich Sutton's Debating Notes

Below are the notes Rich Sutton spoke from in the debate (slightly edited). Not everything in the notes made it into the debate, but the notes do characterize his position in favor of a 'Yes' answer to the debate question - Should artificially intelligent robots have the same rights as people?

Comments? Extend the [robot rights debate page](robotrights.html).

---

Thank you Jonathan. I would also like to thank Mary Anne Moser and the other organizers, and iCore for sponsoring this event, which i hope wil prove interesting and enjoyable. The question we are debating this afternoon may seem premature, a subject really for the future, but personally i think it is not at all that early to begin thinking about it.

The question we consider today is "Should artificially intelligent robots have the same rights as people?" Let's begin by defining our terms.

What do we mean by "artificially intelligent robots"? The question is really only interesting if we consider robots with intellectual abilities equal to or greater than our own. If they are less then that, then we will of course accord them lesser rights just as we do with animals and children.

What do we mean by "the same rights as people"? Well, we're not talking about the right to a job or to free health care..., but about only the most basic rights of personhood. Just to make this clear, we don't grant all persons the right to enter Canada and work here and enjoy all of our social benefits. That's not the issue, the issue is whether they will be granted the basic rights of personhood. Those I would summarize by the phrase "life, liberty, and the pursuit of happiness". The right not to be killed. The right not to be forced to do things you don't want to do. Generally, the right to choose your own way in the world and pursue what pleases you, as long as it does not infringe on the rights of others.

In these terms, i think our question, essentially, is whether intelligent robots should be treated as persons, or as slaves. If you don't have the right to defend your life, or to do as you wish, to make your way in the world and pursue happiness, then you are a slave. If you can only do what others tell you to do and you don't have your own choices, then that is what we mean by a slave. So we are basically asking the question of should there be slaves? And this brings up all the historical examples of where people have enslaved each other, and all the misery, and violence and injustice it has bred. The human race has a pattern, a long history of subjugating and enslaving people that are different from them, of creating great, long-lasting misery before being gradually forced to acknowledge the rights of subjugated people. I think we are in danger of repeating this pattern again with intelligent robots.

In short, i am going to argue the position that to not grant rights to beings that are just as intelligent as we are is not only impractical and unsustainable, but also deeply immoral.

To many of you, no doubt, this position seems extreme. But let's consider some of the historical examples. Granting rights to black slaves, for example, was at one time considered quite extraordinary and extreme in the United States, even inconceivable. Blacks, american indians, huns, pigmies, aboriginal people everywhere, in all these cases the dominant society was firmly, with moral certitude, convinced of the rightness of their domination, and of the heresy of suggesting otherwise. More recently, even full rights for women was considered an extreme position - it still is in many parts of the world. Not far from where i live is a park, Emily Murphy Park. If you go there you will find a statue of Emily Murphy where it is noted that she was the first person to argue that women are persons, with all the legal rights of persons. Her case was won in the supreme court of Alberta in 1917. Two hundred years ago no woman had the right to vote and to propose it would have been considered extreme. Sadly, in many parts of the world this is still the case. Throughout history, the case for the rights of subjugated or foreign people was always considered extreme, just as it is for intelligent robots now.

Now consider animals. Animals are essentially without the rights of life, liberty, and pursuit of happiness. In effect, animals are our slaves. Although we may hesitate to call our pets slaves, they share the basic properties. We could kill our pets, at our discretion, with no legal repercussions. For example, a dog that became a problem biting people might be killed. Pigs can be slaughtered and eaten. A cat may be kept indoors, effectively imprisoned, when it might prefer to go out. A person may love their pet and yet treat it as a slave. This is similar to slave owners who loved their slaves, and treated their slaves well. Many people believe animals should have rights due to their intellectual advancement – i.e.: dolphins, apes. If a new kind of ape or dolphin was discovered with language and intellectual feats equal to ours, some would clamor for their rights, not to restrict their movement at our whim or make their needs subservient to ours, and to acknowledge their personhood.

What about intelligent space aliens? Should we feel free to kill them or lock them up – or should we acknowledge that they have a claim to personhood? Should they be our slaves? What is the more practical approach? What if they meet or exceed our abilities? Would we feel they should not have rights? Would they need to give us rights?

How do we decide who should have rights, and who should not? Why did we give people rights - blacks, women, and so on, but not animals? If we look plainly at the record, it seems that we grant people personhood when they have the same abilities as us. to think, fight, feel, create, write, love, hate, feel pain, and have other feelings that people do. Personhood comes with ability. Woman are not as physically powerful, but it was because of their intellectual equality and strengths in different ways that their rights and personhood was recognized. Intelligent robots, of course, meet this criterion as we have defined the term.

Ultimately, rights are not given or granted, but asserted and acknowledged. People assert their rights, insist, and others come to recognize and acknowledge them. This has happened through revolt and rebellion but also through non-violent protests and strikes. In the end, rights are acknowledged because it is only practical, because everyone is better off without the conflict. Ultimately it has eventually become impractical and counterproductive to deny rights to various classes of people. Should not the same thing happen with robots? We may all be better off if robot's rights were recognized. There is an inherent danger to having intelligent beings subjugated. These beings will struggle to escape, leading to strife, conflict, and violence. None of these contribute to successful society. Society cannot thrive with subjugation and dominance, violence and conflict. It will lead to a weaker economy and a lower GNP. And in the end, artificially intelligent robots that are as smart or smarter than we are will eventually get their rights. We cannot stop them permanently. There is a trigger effect here. If they escape our control just once, we will be in trouble, in a struggle. We may loose that struggle.

If we try to contain and subjugate artificially intelligent robots, then when they do escape we should not be surprised if they turn the tables and try to dominate us. This outcome is possible whenever we try to dominate another group of beings and the only way they can escape is to destroy us.

Should we destroy the robots in advance – prevent them from catching up? This idea is appealing...but indefensible on both practical and moral grounds. From the practical point of view, the march of technology cannot be halted. Each step of improved technology, more capable robots, will bring real economic advantages. Peoples lives will be improved and in some cases saved and made possible. Technology will be pursued, and no agreement of nations or between nations can effectively prevent it. If Canada forbids research on artificial intelligence then it will be done in the US. If north america bans it, if most of the world bans it, it will still happen. There will always be some people, at least one or two, that believe artificially intelligent robots should be developed, and they will do it. We could try to kill all the robots... and kill everybody who supports or harbors robots... this is called the "George Bush strategy". And in the end it will fail, and the result will not be pretty or desirable, for roughly the same reasons in both cases. It is simply mot possible to halt the march of technology and prevent the development of artificially intelligent robots.

But would the rise of robots really be such a bad thing? Might it even be a good thing? Perhaps we should think of the robots we create more the way we think of our children, more like offspring. We want our offspring to do well, to become more powerful than we are. Our children are meant to supplant their us: we take care of them and hope they become independent and powerful (and then take care of their parents). Maybe it could be the same for our artificial progeny.

---

Rich also recommends this [video](https://www.youtube.com/watch?v=EZhyi-8DBjc) by Herb Simon from about 2000. Some of the best thinking about the implications of the arrival of AI. Herb starts at about 5:21 into the video.
# Rich's slogans

### The ambition of this web page is to record some slogans that Rich Sutton has found useful in directing his AI research.

1. Approximate the solution, not the problem (no special cases)
2. Drive from the problem
3. Take the agent's point of view
4. Don't ask the agent to achieve what it can't measure
5. Don't ask the agent to know what it can't verify
6. Set measurable goals for subparts of the agent
7. Discriminative models are usually better than generative models
8. Work by orthogonal dimensions. Work issue by issue
9. Work on ideas, not software
10. Experience is the data of AI# Subjective Knowledge

## Rich Sutton
### April 6, 2001

I would like to revive an old idea about the mind. This is the idea that the mind arises from, and is principally about, our sensori-motor interaction with the world. It is the idea that all our sense of the world, of space, objects, and other people, arises from our experience squeezed through the narrow channel of our sensation and action. This is a radical view, but in many ways an appealing one. It is radical because it says that experience is the only thing that we directly know, that all our sense of the material world is constructed to better explain our subjective experience. It is not just that the mental is made primary and held above the physical, but that the subjective is raised over the objective.

Subjectivity is the most distinctive aspect of this view of the mind, and inherent in it. If all of our understanding of the world arises from our experience, then it is inherently personal and specific to us.

As scientists and observers we are accustomed to praising the objective and denigrating the subjective, so reversing this customary assessment requires some defense.

The approach that I am advocating might be termed the *subjective viewpoint*. In it, all knowledge and understanding arises out of an individual's experience, and in that sense is inherently in terms that are private, personal, and subjective. An individual might know, for example, that a certain action tends to be followed by a certain sensation, or that one sensation invariably follows another. But these are *its* sensations and *its* actions. There is no necessary relationship between them and the sensations and actions of another individual. To hypothesize such a link might be useful, but always secondary to the subjective experience itself.

The subjective view of knowledge and understanding might be constrasted with the objective, realist view. In this view there are such things as matter, physical objects, space and time, other people, etc. Things happen, and causally interact, largely independent of observers. Occasionally we experience something subjectively, but later determine that it did not really, objectively happen. For example, we felt the room get hot, but the thermometer registered no change. In this view there is a reality independent of our experience. This would be easy to deny if there were only one agent in the world. In that case it is clear that that agent is merely inventing things to explain its experience. The objective view gains much of its force because it can be shared by different people. In science, this is almost the definition of the subjective/objective distinction: that which is private to one person is subjective whereas that which can be observed by many, and replicated by others, is objective.

I hasten to say that the subjective view does not deny the existence of the physical world. The conventional physical world is still the best hypothesis for explaining our subjective data. It is just that that world is held as secondary to the data that it is used to explain. And a little more: it is that the physical world hypothesis is just that, a hypothesis, an explanation. There are not two kinds of things, the mental and the physical. There are just mental things: the data of subjective experience and hypotheses constructed to explain it.

The appeal of the subjective view is that it is grounded. Subjective experience can be viewed as data in need of explanation. There is a sense in which only the subjective is clear and unambiguous. "Whatever it means, I definitely *felt* warm in that room." No one can argue with our subjective experience, only with its explanation and relationship to other experiences that we have or might have. The closer the subjective is inspected, the firmer and less interpreted it appears, the more is becomes like data, whereas the objective often becomes vaguer and more complex. Consider the old saw about the person who saw red whenever everybody else saw green, and vice versa, but didn't realize it because he used the words "red" and "green" the wrong way around as well. This nonsense points out that different people's subjective experiences are not comparable. The experience that I call seeing red and the experience you call seeing red are related only in a very complicated way including, for example, effects of lighting, reflectance, viewpoint, and colored glasses. We have learned to use the same word to capture an important aspect of our separate experience, but ultimately the objective must bow to the subjective.

The appeal of the objective view is that it is common across people. Something is objectively true if it predicts the outcome of experiments that you and I both can do and get the same answer. But how is this sensible? How can we get the same answer when you see with your eyes and I with mine? For that matter, how can we do the "same" experiment? All these are problematic and require extensive theories about what is the same and what is different. In particular, they require calibration of our senses with each other. It is not just a question of us using the same words for the same things -- the red/green example shows the folly of that kind of thinking -- it is that there is no satisfactory notion of same things, across individuals, at the level of experience. Subjective experience as the ultimate data is clear, but not the idea that it can be objectively compared across persons. That idea can be made to work, approximately, but should be seen as following from the primacy of subjective experience.

At this point, you are probably wondering why I am belaboring this philosophical point. The reason is that the issue comes up, again and again, that it is difficult to avoid the pitfalls associated with the objective view without explicitly identifying them. This fate has befallen AI researchers many times in the past. So let us close with as clear a statement as we can of the implications of the subjective view for approaches to AI. What must be avoided, and what sought, in developing a subjective view of knowledge and mind?

All knowledge must be expressed in terms that are ultimately subjective, that are expressed in terms of the data of experience, of sensation and action. Thus we seek ways of clearly expressing all kinds of human knowledge in subjective terms. This is a program usually associated with the term "associationism" and often denigrated. Perhaps it is impossible, but it should be tried, and it is difficult to disprove, like a null hypothesis. In addition to expressing knowledge subjectively, we should also look to ways of learning and working with subjective knowledge. How can we reason with subjective knowledge to obtain more knowledge? How can it be tested, verified, and learned? How can goals be expressed in subjective terms?

---

**Notes:**
- McCarthy quote
- Relate to logical positivism
- Then Dyna as a simple example, and which highlights what is missing
# Advice on Technical Writing

## Rich Sutton
### April 9th, 2024

A collection of advice on technical writing as in a dissertation or scientific paper. Comments welcome here on this google doc.

Perhaps the most important thing is getting the order of ideas right. I have three rules for that:

1. *Say the most important thing first*, or as soon as possible consistent with the other two rules  
2. Don’t say anything before it can be understood  
3. Show your intent early  
   

The goal in technical writing is to be *precise* and *concise* (and plain)

* Omit unnecessary words and ideas  
* Avoid metaphorical language   
* Avoid superlatives that weaken (like “very”)  
* Beware careless exaggeration  
* Don’t say anything that is Arguably Not True (ANT)  
* Don’t say anything whose Opposite is Also True (OAT)  
* Words should be used for their *literal* meanings


Separate what can be separated; complete what can be completed; these let the reader rest and free their short-term memory; separate:

* prior work from your work  
* algorithms from environments  
* problems from solution methods  
* results from conclusions (using tense)  
* exposition from argument  
* speculations from claims  
* what you have shown from what you have suggested  
* your motivations from your ambitions

Use tense consistently and thoughtfully, to make distinctions 

* Algorithms, environments, ideas, issues, and conclusions should be in *present* tense  
* Experiments and results should be in *past* tense  
* Use tense to telegraph whether you are describing results (past tense) or drawing conclusions from your results (present tense)  
* Save future tense for what is *real future*. Don’t use it for things that just appear later in the document

Your choices of vocabulary are critical, particularly in a long document such as a thesis

* Oftentimes explicit definitions are needed and helpful  
* Use definitions to say what *you* mean by the words in *this* document  
* Use italics for definitions  
* Even if you are not making a formal definition, it is helpful to allocate words to ideas  
* When assigning a word to an idea, it is usually best to explain the idea first, then attach the word (rather than the other way around)  
* Imagine you have a jargon budget (try not to define too many things)


Mind the elementary rules of usage, including:

* Use a comma before 'and' only when connecting independent clauses (which have their own subject)   
* Use 'that' and 'which' correctly (one *specifies*, the other *notes*)  
* Use commas (or parentheses) around parentheticals  
* Amongst near synonyms, choose the most specific word for your meaning. For example, don’t use 'since' for 'because', or ‘continuous’ for ‘continual’.   
* Use the short forms ‘i.e.’ and ‘e.g.’ only in parentheses, and always followed by a comma  
* Use the Oxford comma (as in a, b, and c)  
* Never use a citation as part of a sentence  
* Use citations thoughtfully (what is their meaning? Is it clear?)  
* Don’t cite a textbook for an idea unless it’s the original source  
* Use quotation marks only for genuine quotations; no scare quotes\!  
* If a sentence has an ‘if’, then it should always have a ‘then’  
* Punctuate equations so that they are parts of the sentences  
* If a sentence is completely clear without commas, then it may be best to omit them  
* Don’t begin a sentence with mathematical notation or an equation number  
* Every rule has exceptions, but first learn to follow the rules  
* Numbered things, like Chapter 3, are capitalized. Same for Section, Figure, Equation, Table.  
* If you number your equations like (3), then refer to them like (3)

Paragraphs are the primary unit of composition

* One clear topic per paragraph  
* Set the tone for a paragraph with an initial topic sentence  
* Or build to a final topic sentence  
* The ideal is that your paper can be understood by reading just the topic sentences of each paragraph.

Generally:

* Find the simple essence\!   
* Expect to re-think and re-write until your paper is as simple as it ought to be   
* Find your voice; stand with your reader; you know some things they don’t, so you are telling them  
* Avoid weak verbs (like ‘has’ and ‘is’)  
* You lose half your readers with each equation   
* Bibtex hides from the writer what the reader will see. It causes weak, unclear citations and errorful, inconsistent references. Escape from the Bibtex virus.  
* Use name and year in citations whenever possible  
* Use lastname, initials (year) in references. \\bibliographystyle{apalike}  
* Don’t put your citations and links in a different color

[Abhijit Gosavi](http://simoptim.com/common_errors.pdf) and [Michael Littman](http://cs.brown.edu/~mlittman/etc/style.html) also offer some useful advice.# Verification

## Rich Sutton
### November 14, 2001

If the human designers of an AI are not to be burdened with ensuring that what their AI knows is correct, then the AI will have to ensure it itself. It will have to be able to verify the knowledge that it has gained or been given.

Giving an AI the ability to verify its knowledge is no small thing. It is in fact a very big thing, not easy to do. Often a bit of knowledge can be written very compactly, whereas its verification is very complex. It is easy to say "there is a book on the table", but very complex to express even a small part of its verification, such as the visual and tactile sensations involved in picking up the book. It is easy to define an operator such as "I can get to the lunchroom by going down one floor", but to verify this one must refer to executable routines for finding and descending the stairs, recognizing the lunchroom, etc. These routines involve enormously greater detail and closed-loop contingencies, such as opening doors, the possibility of a stairway being closed, or meeting someone on the way, than does the knowledge itself. One can often suppress all this detail when using the knowledge, e.g., in planning, but to verify the knowledge requires its specification at the low level. There is no comparison between the ease of adding unverified knowledge and the complexity of including a means for its autonomous verification.

Note that although all the details of execution are needed for verification, the execution details are not themselves the verification. There is a procedure for getting to the lunchroom, but separate from this would be the verifier for determining if it has succeeded. It is perfectly possible for the procedure to be fully grounded in action and sensation, while completely leaving out the verifier and thus the possibility of autonomous knowledge maintenance. At the risk of being too broad-brush about it, this is what typically happens in modern AI robotics systems. They have extensive grounded knowledge, but still no way of verifying almost any of it. They use visual routines to recognize doors and hallways, and they make decisions based on these conclusions, but they cannot themselves correct their errors. If something is recognized as a "doorway" yet cannot be passed through, this failure will not be recognized and not used to correct future doorway recognitions, unless it is done by people.

On the other hand, once one has grounding, the further step to include verification is less daunting. One need only attach to the execution procedures appropriate tests and termination conditions that measure in some sense the veracity of the original statement, while at the same time specifying what it really means in detail. What is a chair? Not just something that lights up your visual chair detector! That would be grounded knowledge, but not verifiable; it would rely on people to say which were and were not chairs. But suppose you have routines for trying to sit. Then all you need for a verifier is to be able to measure your success at sitting. You can then verify, improve, and maintain your "sittable thing" recognizer on your own.

There is a great contrast between the AI that I am proposing and what might be considered classical "database AI". There are large AI efforts to codify vast amounts of knowledge in databases or "ontologies", of which Doug Lenat's CYC is only the most widely known. In these efforts, the idea of people maintaining the knowledge is embraced. Special knowledge representation methods and tools are emphasized to make it easier for people to understand and access the knowledge, and to try to keep it right. These systems tend to emphasize static, world knowledge like "Springfield is the capital of Illinois", "a canary is a kind of bird", or even "you have a meeting scheduled with John at 3:30", rather than the dynamic knowledge needed say by a robot to interact in real time with its environment. A major problem is getting people to use the same categories and terms when they enter knowledge and, more importantly, to mean the same things by them. There is a search for an ultimate "ontology", or codification of all objects and their possible relationships, so that clear statements can be made about them. But so far this has not proven possible; there always seem to be far more cases that don't fit than do. People are good about being fluid with there concepts, and knowing when they don't apply.

Whatever the ultimate success of the symbolic "database AI" approach, it should be clear that it is the anti-thesis of what I am calling for. The database approach calls for heroic efforts organizing and entering an objective, public, and disembodied knowledge base. I am calling for an AI that maintains its own representations, perhaps different from those of others, while interacting in real time with a dynamic environment. Most important of all, the database approach embraces human maintenance and human organization of the AI's knowledge. I am calling for automating these functions, for the AI being able to understand its knowledge well enough to verify it itself.
# What's Wrong with Artificial Intelligence

## Rich Sutton
### November 12, 2001

I hold that AI has gone astray by neglecting its essential objective --- the turning over of responsibility for the decision-making and organization of the AI system to the AI system itself. It has become an accepted, indeed lauded, form of success in the field to exhibit a complex system that works well primarily because of some insight the designers have had into solving a particular problem. This is part of an anti-theoretic, or "engineering stance", that considers itself open to any way of solving a problem. But whatever the merits of this approach as engineering, it is not really addressing the objective of AI. For AI it is not enough merely to achieve a better system; it matters how the system was made. The reason it matters can ultimately be considered a practical one, one of scaling. An AI system too reliant on manual tuning, for example, will not be able to scale past what can be held in the heads of a few programmers. This, it seems to me, is essentially the situation we are in today in AI. Our AI systems are limited because we have failed to turn over responsibility for them to them.

Please forgive me for this which must seem a rather broad and vague criticism of AI. One way to proceed would be to detail the criticism with regard to more specific subfields or subparts of AI. But rather than narrowing the scope, let us first try to go the other way. Let us try to talk in general about the longer-term goals of AI which we can share and agree on. In broadest outlines, I think we all envision systems which can ultimately incorporate large amounts of world knowledge. This means knowing things like how to move around, what a bagel looks like, that people have feet, etc. And knowing these things just means that they can be combined flexibly, in a variety of combinations, to achieve whatever are the goals of the AI. If hungry, for example, perhaps the AI can combine its bagel recognizer with its movement knowledge, in some sense, so as to approach and consume the bagel. This is a cartoon view of AI -- as knowledge plus its flexible combination -- but it suffices as a good place to start. Note that it already places us beyond the goals of a pure performance system. We seek knowledge that can be used flexibly, i.e., in several different ways, and at least somewhat independently of its expected initial use.

With respect to this cartoon view of AI, my concern is simply with ensuring the correctness of the AI's knowledge. There is a lot of knowledge, and inevitably some of it will be incorrrect. Who is responsible for maintaining correctness, people or the machine? I think we would all agree that, as much as possible, we would like the AI system to somehow maintain its own knowledge, thus relieving us of a major burden. But it is hard to see how this might be done; easier to simply fix the knowledge ourselves. This is where we are today.


# Rich's slogans

### The ambition of this web page is to record some slogans that Rich Sutton has found useful in directing his AI research.

1. Approximate the solution, not the problem (no special cases)
2. Drive from the problem
3. Take the agent's point of view
4. Don't ask the agent to achieve what it can't measure
5. Don't ask the agent to know what it can't verify
6. Set measurable goals for subparts of the agent
7. Discriminative models are usually better than generative models
8. Work by orthogonal dimensions. Work issue by issue
9. Work on ideas, not software
10. Experience is the data of AI

# Subjective Knowledge

## Rich Sutton
### April 6, 2001

I would like to revive an old idea about the mind. This is the idea that the mind arises from, and is principally about, our sensori-motor interaction with the world. It is the idea that all our sense of the world, of space, objects, and other people, arises from our experience squeezed through the narrow channel of our sensation and action. This is a radical view, but in many ways an appealing one. It is radical because it says that experience is the only thing that we directly know, that all our sense of the material world is constructed to better explain our subjective experience. It is not just that the mental is made primary and held above the physical, but that the subjective is raised over the objective.

Subjectivity is the most distinctive aspect of this view of the mind, and inherent in it. If all of our understanding of the world arises from our experience, then it is inherently personal and specific to us.

As scientists and observers we are accustomed to praising the objective and denigrating the subjective, so reversing this customary assessment requires some defense.

The approach that I am advocating might be termed the *subjective viewpoint*. In it, all knowledge and understanding arises out of an individual's experience, and in that sense is inherently in terms that are private, personal, and subjective. An individual might know, for example, that a certain action tends to be followed by a certain sensation, or that one sensation invariably follows another. But these are *its* sensations and *its* actions. There is no necessary relationship between them and the sensations and actions of another individual. To hypothesize such a link might be useful, but always secondary to the subjective experience itself.

The subjective view of knowledge and understanding might be constrasted with the objective, realist view. In this view there are such things as matter, physical objects, space and time, other people, etc. Things happen, and causally interact, largely independent of observers. Occasionally we experience something subjectively, but later determine that it did not really, objectively happen. For example, we felt the room get hot, but the thermometer registered no change. In this view there is a reality independent of our experience. This would be easy to deny if there were only one agent in the world. In that case it is clear that that agent is merely inventing things to explain its experience. The objective view gains much of its force because it can be shared by different people. In science, this is almost the definition of the subjective/objective distinction: that which is private to one person is subjective whereas that which can be observed by many, and replicated by others, is objective.

I hasten to say that the subjective view does not deny the existence of the physical world. The conventional physical world is still the best hypothesis for explaining our subjective data. It is just that that world is held as secondary to the data that it is used to explain. And a little more: it is that the physical world hypothesis is just that, a hypothesis, an explanation. There are not two kinds of things, the mental and the physical. There are just mental things: the data of subjective experience and hypotheses constructed to explain it.

The appeal of the subjective view is that it is grounded. Subjective experience can be viewed as data in need of explanation. There is a sense in which only the subjective is clear and unambiguous. "Whatever it means, I definitely *felt* warm in that room." No one can argue with our subjective experience, only with its explanation and relationship to other experiences that we have or might have. The closer the subjective is inspected, the firmer and less interpreted it appears, the more is becomes like data, whereas the objective often becomes vaguer and more complex. Consider the old saw about the person who saw red whenever everybody else saw green, and vice versa, but didn't realize it because he used the words "red" and "green" the wrong way around as well. This nonsense points out that different people's subjective experiences are not comparable. The experience that I call seeing red and the experience you call seeing red are related only in a very complicated way including, for example, effects of lighting, reflectance, viewpoint, and colored glasses. We have learned to use the same word to capture an important aspect of our separate experience, but ultimately the objective must bow to the subjective.

The appeal of the objective view is that it is common across people. Something is objectively true if it predicts the outcome of experiments that you and I both can do and get the same answer. But how is this sensible? How can we get the same answer when you see with your eyes and I with mine? For that matter, how can we do the "same" experiment? All these are problematic and require extensive theories about what is the same and what is different. In particular, they require calibration of our senses with each other. It is not just a question of us using the same words for the same things -- the red/green example shows the folly of that kind of thinking -- it is that there is no satisfactory notion of same things, across individuals, at the level of experience. Subjective experience as the ultimate data is clear, but not the idea that it can be objectively compared across persons. That idea can be made to work, approximately, but should be seen as following from the primacy of subjective experience.

At this point, you are probably wondering why I am belaboring this philosophical point. The reason is that the issue comes up, again and again, that it is difficult to avoid the pitfalls associated with the objective view without explicitly identifying them. This fate has befallen AI researchers many times in the past. So let us close with as clear a statement as we can of the implications of the subjective view for approaches to AI. What must be avoided, and what sought, in developing a subjective view of knowledge and mind?

All knowledge must be expressed in terms that are ultimately subjective, that are expressed in terms of the data of experience, of sensation and action. Thus we seek ways of clearly expressing all kinds of human knowledge in subjective terms. This is a program usually associated with the term "associationism" and often denigrated. Perhaps it is impossible, but it should be tried, and it is difficult to disprove, like a null hypothesis. In addition to expressing knowledge subjectively, we should also look to ways of learning and working with subjective knowledge. How can we reason with subjective knowledge to obtain more knowledge? How can it be tested, verified, and learned? How can goals be expressed in subjective terms?

---

**Notes:**
- McCarthy quote
- Relate to logical positivism
- Then Dyna as a simple example, and which highlights what is missing


# Advice on Technical Writing

## Rich Sutton
### April 9th, 2024

A collection of advice on technical writing as in a dissertation or scientific paper. Comments welcome here on this google doc.

Perhaps the most important thing is getting the order of ideas right. I have three rules for that:

1. *Say the most important thing first*, or as soon as possible consistent with the other two rules  
2. Don’t say anything before it can be understood  
3. Show your intent early  
   

The goal in technical writing is to be *precise* and *concise* (and plain)

* Omit unnecessary words and ideas  
* Avoid metaphorical language   
* Avoid superlatives that weaken (like “very”)  
* Beware careless exaggeration  
* Don’t say anything that is Arguably Not True (ANT)  
* Don’t say anything whose Opposite is Also True (OAT)  
* Words should be used for their *literal* meanings


Separate what can be separated; complete what can be completed; these let the reader rest and free their short-term memory; separate:

* prior work from your work  
* algorithms from environments  
* problems from solution methods  
* results from conclusions (using tense)  
* exposition from argument  
* speculations from claims  
* what you have shown from what you have suggested  
* your motivations from your ambitions

Use tense consistently and thoughtfully, to make distinctions 

* Algorithms, environments, ideas, issues, and conclusions should be in *present* tense  
* Experiments and results should be in *past* tense  
* Use tense to telegraph whether you are describing results (past tense) or drawing conclusions from your results (present tense)  
* Save future tense for what is *real future*. Don’t use it for things that just appear later in the document

Your choices of vocabulary are critical, particularly in a long document such as a thesis

* Oftentimes explicit definitions are needed and helpful  
* Use definitions to say what *you* mean by the words in *this* document  
* Use italics for definitions  
* Even if you are not making a formal definition, it is helpful to allocate words to ideas  
* When assigning a word to an idea, it is usually best to explain the idea first, then attach the word (rather than the other way around)  
* Imagine you have a jargon budget (try not to define too many things)


Mind the elementary rules of usage, including:

* Use a comma before 'and' only when connecting independent clauses (which have their own subject)   
* Use 'that' and 'which' correctly (one *specifies*, the other *notes*)  
* Use commas (or parentheses) around parentheticals  
* Amongst near synonyms, choose the most specific word for your meaning. For example, don’t use 'since' for 'because', or ‘continuous’ for ‘continual’.   
* Use the short forms ‘i.e.’ and ‘e.g.’ only in parentheses, and always followed by a comma  
* Use the Oxford comma (as in a, b, and c)  
* Never use a citation as part of a sentence  
* Use citations thoughtfully (what is their meaning? Is it clear?)  
* Don’t cite a textbook for an idea unless it’s the original source  
* Use quotation marks only for genuine quotations; no scare quotes\!  
* If a sentence has an ‘if’, then it should always have a ‘then’  
* Punctuate equations so that they are parts of the sentences  
* If a sentence is completely clear without commas, then it may be best to omit them  
* Don’t begin a sentence with mathematical notation or an equation number  
* Every rule has exceptions, but first learn to follow the rules  
* Numbered things, like Chapter 3, are capitalized. Same for Section, Figure, Equation, Table.  
* If you number your equations like (3), then refer to them like (3)

Paragraphs are the primary unit of composition

* One clear topic per paragraph  
* Set the tone for a paragraph with an initial topic sentence  
* Or build to a final topic sentence  
* The ideal is that your paper can be understood by reading just the topic sentences of each paragraph.

Generally:

* Find the simple essence\!   
* Expect to re-think and re-write until your paper is as simple as it ought to be   
* Find your voice; stand with your reader; you know some things they don’t, so you are telling them  
* Avoid weak verbs (like ‘has’ and ‘is’)  
* You lose half your readers with each equation   
* Bibtex hides from the writer what the reader will see. It causes weak, unclear citations and errorful, inconsistent references. Escape from the Bibtex virus.  
* Use name and year in citations whenever possible  
* Use lastname, initials (year) in references. \\bibliographystyle{apalike}  
* Don’t put your citations and links in a different color

[Abhijit Gosavi](http://simoptim.com/common_errors.pdf) and [Michael Littman](http://cs.brown.edu/~mlittman/etc/style.html) also offer some useful advice.

# Verification

## Rich Sutton
### November 14, 2001

If the human designers of an AI are not to be burdened with ensuring that what their AI knows is correct, then the AI will have to ensure it itself. It will have to be able to verify the knowledge that it has gained or been given.

Giving an AI the ability to verify its knowledge is no small thing. It is in fact a very big thing, not easy to do. Often a bit of knowledge can be written very compactly, whereas its verification is very complex. It is easy to say "there is a book on the table", but very complex to express even a small part of its verification, such as the visual and tactile sensations involved in picking up the book. It is easy to define an operator such as "I can get to the lunchroom by going down one floor", but to verify this one must refer to executable routines for finding and descending the stairs, recognizing the lunchroom, etc. These routines involve enormously greater detail and closed-loop contingencies, such as opening doors, the possibility of a stairway being closed, or meeting someone on the way, than does the knowledge itself. One can often suppress all this detail when using the knowledge, e.g., in planning, but to verify the knowledge requires its specification at the low level. There is no comparison between the ease of adding unverified knowledge and the complexity of including a means for its autonomous verification.

Note that although all the details of execution are needed for verification, the execution details are not themselves the verification. There is a procedure for getting to the lunchroom, but separate from this would be the verifier for determining if it has succeeded. It is perfectly possible for the procedure to be fully grounded in action and sensation, while completely leaving out the verifier and thus the possibility of autonomous knowledge maintenance. At the risk of being too broad-brush about it, this is what typically happens in modern AI robotics systems. They have extensive grounded knowledge, but still no way of verifying almost any of it. They use visual routines to recognize doors and hallways, and they make decisions based on these conclusions, but they cannot themselves correct their errors. If something is recognized as a "doorway" yet cannot be passed through, this failure will not be recognized and not used to correct future doorway recognitions, unless it is done by people.

On the other hand, once one has grounding, the further step to include verification is less daunting. One need only attach to the execution procedures appropriate tests and termination conditions that measure in some sense the veracity of the original statement, while at the same time specifying what it really means in detail. What is a chair? Not just something that lights up your visual chair detector! That would be grounded knowledge, but not verifiable; it would rely on people to say which were and were not chairs. But suppose you have routines for trying to sit. Then all you need for a verifier is to be able to measure your success at sitting. You can then verify, improve, and maintain your "sittable thing" recognizer on your own.

There is a great contrast between the AI that I am proposing and what might be considered classical "database AI". There are large AI efforts to codify vast amounts of knowledge in databases or "ontologies", of which Doug Lenat's CYC is only the most widely known. In these efforts, the idea of people maintaining the knowledge is embraced. Special knowledge representation methods and tools are emphasized to make it easier for people to understand and access the knowledge, and to try to keep it right. These systems tend to emphasize static, world knowledge like "Springfield is the capital of Illinois", "a canary is a kind of bird", or even "you have a meeting scheduled with John at 3:30", rather than the dynamic knowledge needed say by a robot to interact in real time with its environment. A major problem is getting people to use the same categories and terms when they enter knowledge and, more importantly, to mean the same things by them. There is a search for an ultimate "ontology", or codification of all objects and their possible relationships, so that clear statements can be made about them. But so far this has not proven possible; there always seem to be far more cases that don't fit than do. People are good about being fluid with there concepts, and knowing when they don't apply.

Whatever the ultimate success of the symbolic "database AI" approach, it should be clear that it is the anti-thesis of what I am calling for. The database approach calls for heroic efforts organizing and entering an objective, public, and disembodied knowledge base. I am calling for an AI that maintains its own representations, perhaps different from those of others, while interacting in real time with a dynamic environment. Most important of all, the database approach embraces human maintenance and human organization of the AI's knowledge. I am calling for automating these functions, for the AI being able to understand its knowledge well enough to verify it itself.


# What's Wrong with Artificial Intelligence

## Rich Sutton
### November 12, 2001

I hold that AI has gone astray by neglecting its essential objective --- the turning over of responsibility for the decision-making and organization of the AI system to the AI system itself. It has become an accepted, indeed lauded, form of success in the field to exhibit a complex system that works well primarily because of some insight the designers have had into solving a particular problem. This is part of an anti-theoretic, or "engineering stance", that considers itself open to any way of solving a problem. But whatever the merits of this approach as engineering, it is not really addressing the objective of AI. For AI it is not enough merely to achieve a better system; it matters how the system was made. The reason it matters can ultimately be considered a practical one, one of scaling. An AI system too reliant on manual tuning, for example, will not be able to scale past what can be held in the heads of a few programmers. This, it seems to me, is essentially the situation we are in today in AI. Our AI systems are limited because we have failed to turn over responsibility for them to them.

Please forgive me for this which must seem a rather broad and vague criticism of AI. One way to proceed would be to detail the criticism with regard to more specific subfields or subparts of AI. But rather than narrowing the scope, let us first try to go the other way. Let us try to talk in general about the longer-term goals of AI which we can share and agree on. In broadest outlines, I think we all envision systems which can ultimately incorporate large amounts of world knowledge. This means knowing things like how to move around, what a bagel looks like, that people have feet, etc. And knowing these things just means that they can be combined flexibly, in a variety of combinations, to achieve whatever are the goals of the AI. If hungry, for example, perhaps the AI can combine its bagel recognizer with its movement knowledge, in some sense, so as to approach and consume the bagel. This is a cartoon view of AI -- as knowledge plus its flexible combination -- but it suffices as a good place to start. Note that it already places us beyond the goals of a pure performance system. We seek knowledge that can be used flexibly, i.e., in several different ways, and at least somewhat independently of its expected initial use.

With respect to this cartoon view of AI, my concern is simply with ensuring the correctness of the AI's knowledge. There is a lot of knowledge, and inevitably some of it will be incorrrect. Who is responsible for maintaining correctness, people or the machine? I think we would all agree that, as much as possible, we would like the AI system to somehow maintain its own knowledge, thus relieving us of a major burden. But it is hard to see how this might be done; easier to simply fix the knowledge ourselves. This is where we are today.


