<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
  <head>

    <meta http-equiv="content-type" content="text/html; charset=windows-1252">
    <title>The Bitter Lesson</title>
    <style type="text/css">
<!--
.style1 {font-family: Palatino}
-->
  </style>
  </head>
  <body>
    <span class="style1">
      <h1>The One-Step Trap (in AI Research)<br>
      </h1>
      <h2>Rich Sutton</h2>
      <h3>Written up for X on July 18, 2024<br>
      </h3>
      The one-step trap is the common mistake of thinking that all or
      most of an AI agent�s learned predictions can be one-step ones,
      with all longer-term predictions generated as needed by iterating
      the one-step predictions. The most important place where the trap
      arises is when the one-step predictions constitute a model of the
      world and of how it evolves over time. It is appealing to think
      that one can learn just a one-step transition model and then �roll
      it out� to predict all the longer-term consequences of a way of
      behaving. The one-step model is thought of as being analogous to
      physics, or to a realistic simulator.<br>
      <br>
      The appeal of this mistake is that it contains a grain of truth:
      if all one-step predictions can be made with perfect accuracy,
      then they can be used to make all longer-term prediction with
      perfect accuracy. However, if the one-step predictions are not
      perfectly accurate, then all bets are off. In practice, iterating
      one-step predictions usually produces poor results. The one-step
      errors compound and accumulate into large errors in the long-term
      predictions. In addition, computing long-term predictions from
      one-step ones is prohibitively computationally complex. In a
      stochastic world, or for a stochastic policy, the future is not a
      single trajectory, but a tree of possibilities, each of which must
      be imagined and weighted by its probability. As a result, the
      computational complexity of computing a long-term prediction from
      one-step predictions is exponential in the length of the
      prediction, and thus generally infeasible. <br>
      <br>
      The bottom line is that one-step models of the world are hopeless,
      yet extremely appealing, and are widely used in POMDPs, Bayesian
      analyses, control theory, and in compression theories of AI.<br>
      <br>
      The solution, in my opinion, is to form temporally abstract models
      of the world using options and GVFs, as in the following
      references.<br>
      <br>
      Sutton, R.S., Precup, D., Singh, S. (1999). Between MDPs and
      semi-MDPs: A Framework for Temporal Abstraction in Reinforcement
      Learning. Artificial Intelligence 112:181-211.<br>
      <br>
      Sutton, R. S., Modayil, J., Delp, M., Degris, T., Pilarski, P. M.,
      White, A., Precup, D. (2011). Horde: A scalable real-time
      architecture for learning knowledge from unsupervised sensorimotor
      interaction. In Proceedings of the Tenth International Conference
      on Autonomous Agents and Multiagent Systems, Taipei, Taiwan.<br>
      <br>
      Sutton, R. S., Machado, M. C., Holland, G. Z., Timbers, D. S. F.,
      Tanner, B., &amp; White, A. (2023). Reward-respecting subtasks for
      model-based reinforcement learning. Artificial Intelligence 324.<br>
      <br>
    </span>
  </body>
</html>
