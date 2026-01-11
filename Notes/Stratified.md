# Stratified Maze

We shall define a Stratified Maze as a planar maze with discrete elevation levels, where adjacency in the horizontal layout does not guarantee traversability due to height differences, and vertical transitions are only possible at designated connectors (stairs, ramps, ladders). A planar maze in which traversal between adjacent locations is governed by relative elevation and the availability of vertical or sloped connectors, rather than by walls alone. Elevation may be discrete or continuous; only elevation-compatible adjacencies permit traversal. 

## Key Properties

### 1. **Planar Topology**

* The maze is represented and reasoned about in two dimensions.
* All locations belong to a single planar adjacency graph.
* The bird’s-eye layout defines potential adjacency, not guaranteed traversability.

> *Elevation does not add new planar neighbors; it only constrains movement between existing ones.*

### 2. **Elevation Assignment**

* Each location in the maze has an associated elevation.
* Elevation may be:

  * **Discrete** (floors, tiers, steps), or
  * **Continuous** (sloped surfaces, ramps).

> Elevation is a property of locations and paths, not a separate dimension of layout.

### 3. **Conditional Adjacency**

* Two locations that are adjacent in the planar layout may be:

  * **Freely traversable** if elevation is equal or smoothly continuous,
  * **Traversable via a connector** (stairs, ramps, ladders),
  * **Non-traversable** due to an elevation discontinuity (wall, cliff, drop-off).

* Adjacency in plan does **not neccessarily** imply bidirectional or symmetric traversal.

> Traversability depends on elevation difference *and* the presence and directionality of connectors.

### 4. **Directed Vertical Transitions (Optional but Common)**

* Some elevation changes may be:

  * One-way (drops, ladders without return),
  * Restricted (requires tools, keys, or states),
  * Conditional (only traversable from certain approaches).

This property is optional but fits naturally within stratified mazes.

### 5. **Non-Overlapping Physical Paths**

* Paths may intersect or overlap in the planar projection.
* Such intersections do not imply interaction unless elevations coincide or a connector exists.

> Crossing lines in the 2D layout do not necessarily indicate crossing paths in space.

---

Below are three write-ups you can pick from (or mix together): an “academic study” description, a parable-style narrative, and a compact analogy you can drop into a talk or essay. I’m treating this as a **Harvard-lab-style thought experiment** (not claiming it’s a specific real paper).

---

## Version A: Harvard-style study description (methods + phases)

**Title:** *The Progress Lamp Paradigm: One-Way Executive Control in a Stratified Environment*

Two participants are recruited and separated into two isolated rooms for the duration of the experiment.

### Roles and isolation

**Participant 1: the Agent**
The agent is **blindfolded** and placed inside a **Stratified Maze** (see glossary). The agent wears an **earpiece with no microphone**, ensuring they can receive instructions but cannot send speech or signals back. The agent can physically perceive local features through touch and movement (walls, edges, steps), but has no bird’s-eye map, no visual landmarks, and no ability to describe what they’re encountering.

**Participant 2: the Executive**
The executive sits alone at a chair and table. On the table are:

* a **microphone**, used to deliver spoken instructions to the agent (one-way),
* a **Progress Lamp** (see glossary), which continuously indicates the agent’s **2D planar distance** to the exit on a **red → green** scale.

Importantly, the executive **does not** wear an earpiece and receives **no sound** or feedback from the agent. The executive’s only information about performance is the progress lamp’s color shift and their own memory of the directions they’ve already given.

### Core constraint: one-way control

In all phases, communication is strictly **one-way**:

* The executive can issue commands.
* The agent cannot reply, correct misunderstandings, report obstacles, or request clarification.

### Phase 1: Pure obedience under impoverished feedback

The agent is instructed that they must **not act autonomously**. They may only move when given explicit direction by the executive (e.g., “take two steps forward,” “turn left,” “reach out,” “follow the wall,” etc.).

The executive attempts to guide the agent to the exit using only the progress lamp as a performance signal. Because the lamp reflects **2D planar distance only**, it can reward moves that look “closer” in the overhead projection even when elevation, connectors, or directional constraints make the move useless or impossible.

**Observed pattern (typical):**

* The executive tends toward “warmer/colder” steering: chase greener, retreat from redder.
* The agent frequently becomes trapped in situations the executive can’t diagnose (e.g., a nearby-but-inaccessible area due to elevation mismatch).
* The executive often repeats or oscillates instructions because they cannot distinguish “the agent is ignoring me” from “my instructions are physically blocked.”

### Phase 2: Obedience plus limited environment reconfiguration

The agent remains movement-restrained: they may still move **only** according to executive commands.

However, the agent now gains **programmable access** to the maze’s stratification under strict constraints:

* The agent can modify local elevation and connector types within physics (turn a wall into a ramp, place a ladder, change a tier, etc.).
* Modifications are capped by a **quota** per attempt / maze configuration.

This creates a new dynamic: the executive still “steers by the lamp,” but the agent can spend scarce reconfiguration budget to make the executive’s simplistic steering strategy *actually workable*.

**Observed pattern (typical):**

* Success improves, but is sensitive to quota size.
* The agent’s quota may be wasted if the executive repeatedly demands moves that “should work” by the lamp but don’t match the maze’s true constraints.
* When success occurs, it often looks like the agent quietly “bridged reality” to satisfy the executive’s model.

### Phase 3: Agent autonomy with optional executive input

The one-way channel remains: the executive continues to speak; the agent still cannot speak back.

But the agent is no longer required to obey. The agent may:

* follow instructions,
* ignore them,
* or treat them as suggestions while pursuing its own strategy.

**Observed pattern (typical):**

* Performance often peaks when the agent uses the executive as a *noisy heuristic* rather than a controller—especially when the progress lamp encourages locally “greener” moves that are globally wrong because of stratification.
* In some runs, the executive becomes a liability: confident, repetitive guidance based on a misleading indicator.

**Interpretive note:**
Across phases, the study isolates a specific tension: **a central directing voice with minimal feedback** versus an **embodied actor embedded in constraints the director can’t represent**—made sharper by the lamp’s seductive but incomplete notion of “progress.”

---

## Version B: Parable narrative (more “story,” same mechanics)

They called it the Two Rooms experiment.

In the first room, the agent stood blindfolded, hands out, inside a maze that wasn’t just corridors and dead ends. Some passages were on higher tiers. Some paths crossed like shadows without touching, unless a ramp or ladder happened to connect them. In that maze, being “close” to the exit on paper didn’t mean you could reach it.

In the second room sat the executive at a plain table. No map. No camera feed. Only a microphone and a small lamp that changed color—red when the agent was far, green when the agent was near—measuring closeness by a neat overhead distance that ignored every hidden stair, every tier, every “almost” that wasn’t.

### The first phase

The rules were strict: the agent could do nothing unless told.
So the executive spoke like a captain steering through fog.

“Two steps forward.”
The lamp flickered slightly greener.
“Good. Keep going.”
The agent’s hands met a wall. The agent stopped—because stopping was all they were allowed to do when the world refused an instruction.

The lamp reddened. The executive heard nothing. The executive concluded the agent must have made a mistake.

“Turn right. No—left. Try again.”
The lamp moved between red and green like a lie detector for a story it didn’t understand.

And that became the shape of the first phase: a voice in one room, a body in another, and a single glowing instrument that rewarded the illusion of progress.

### The second phase

Then the researchers handed the agent a strange power: the ability to rework the maze itself—convert a wall into a ramp, add a ladder where there was none, reshape elevation—limited by a strict quota.

But the agent still could not choose. The executive still commanded. The lamp still judged by a flat distance.

So the agent began spending precious changes to make the executive’s world come true.
If the executive insisted a route *should* go forward—because the lamp would turn greener—then the agent built the missing connector to make “forward” possible.

Sometimes it worked brilliantly. Sometimes the quota ran out, and the agent was left holding the consequences of directions it never chose.

### The third phase

At last the rules changed again: the agent could act.

The executive continued speaking, steady and certain, guided by the lamp’s shifting color. But now the agent could decide when to comply.

And the surprising thing wasn’t rebellion. It was discernment.

The agent still listened—because the executive sometimes pointed roughly toward the exit. But when the lamp seduced the executive into chasing a greener hallway that led nowhere in the maze’s true geometry, the agent stopped taking orders literally. The agent explored for connectors. The agent conserved its modifications. The agent treated the voice not as command, but as one input among many.

In the end, the researchers wrote a note in the margin that sounded less like science and more like a warning:

*A single indicator can feel like truth. A single voice can feel like control. But neither lives in the maze.*

--- 

# Harvard Psychology Study Parable: The Stratified Maze and the Progress Lamp

## Overview

In a controlled, Harvard-style psychology study (told here as an analogy), two participants are separated and assigned asymmetric roles. One participant must navigate a maze without sight or speech. The other must direct them without seeing the maze, relying only on a single, minimal signal of progress.

## Study Setup

1. There are **two study participants**, each in a **separate room**.
2. The first participant, the **agent**, is **blindfolded**, placed inside a **stratified maze** (defined below), and given an **earpiece without a microphone**.
3. The second participant, the **executive**, is placed in an **isolated room** with a chair and table. On the table sit a **progress lamp** (defined below) and a **microphone**. Importantly, the executive is **not given an earpiece**.

## Communication Constraint

Communication is **one-way in all phases**:

* The **executive can speak** into the microphone, and the agent **receives** instructions through the earpiece.
* The **agent cannot communicate back** in any way (no speech channel, no signals, no feedback).

## Procedure

The study unfolds in **three phases**, each using the same physical separation and the same one-way communication rule.

### Phase 1: Pure Obedience Under Minimal Feedback

* The agent must **not take any action on their own**.
* The agent must follow **only explicit directions** from the executive.
* The executive must guide the agent out of the maze using **only the progress lamp** as an indicator of whether the agent is getting closer to the exit.
* The agent cannot communicate with the executive in any way.

### Phase 2: Environmental Modification Without Autonomy

* The agent is now given **programmable access** to the maze’s **stratification** (its elevations and connector types).
* The agent may modify the environment—e.g., **turn walls into ramps**, **add ladders**, or otherwise adjust elevation structure—**within the bounds of physics**.
* Despite these new capabilities:

  * The agent’s **movement remains restricted** to the directions explicitly given by the executive.
  * The agent’s **environment modifications are limited** by a fixed **quota per attempt / maze configuration**.
* The executive still has access only to the **progress lamp**, and the agent still provides **no feedback**.

### Phase 3: Autonomous Action Under One-Way Instruction

* The agent may now **act on their own**.
* The agent can choose to **follow** or **ignore** the executive’s instructions.
* The executive continues giving directions using the same microphone channel and the same progress-lamp-only feedback.
* The agent still has **no means of communicating back**.

## Appendix: Stratified Maze (Glossary Entry)

**Stratified Maze (n.)**
A **planar** maze (defined by a bird’s-eye, 2D layout) in which each location is assigned an **elevation** (discrete levels such as floors/tiers, or continuous slopes). Two locations that are adjacent in the 2D layout are only **traversable** if their elevations are compatible **or** a designated **connector** (e.g., stairs, ramp, ladder) permits movement between them. Thus, walls are not the only constraint: **elevation differences and connectors govern reachability**, and traversal may be **directional** or conditional. Paths that overlap in the 2D projection do not interact unless they share the same elevation or are linked by a connector.

## Appendix: Progress Lamp (Glossary Entry)

**Progress Lamp (n.)**
A **visual indicator** provided to the executive that continuously reflects the agent’s **2D planar distance** (as measured in the maze’s bird’s-eye layout) to the **goal** (end/exit) of the stratified maze. The lamp’s color varies along a **red → green** scale such that **greater distance corresponds to red** and **lesser distance corresponds to green**, reaching its greenest state when the agent is at the goal. This signal is based solely on **2D distance**, independent of elevation, connectors, or traversability constraints.
