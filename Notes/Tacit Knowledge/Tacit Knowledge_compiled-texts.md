---
title: "The Tacit Knowledge Series"
source: "https://commoncog.com/the-tacit-knowledge-series/"
author:
  - "[[Cedric Chin]]"
published: 2020-06-23
created: 2025-05-30
description: "A series about tacit knowledge, which is knowledge that cannot be captured through words alone. We explore what this means for expertise, and why it is important if you are serious at self improvement."
tags:
  - "clippings"
---
![Feature image for The Tacit Knowledge Series](https://commoncog.com/content/images/size/w600/2023/11/tacit_knowledge_series-1.jpg)

**Tacit knowledge is ‘knowledge that cannot be captured through words alone’.**

This series explores how expertise is tacit, why the research around extracting tacit knowledge is more important than the literature on deliberate practice, and how to go about acquiring tacit knowledge in the pursuit of skill acquisition.

*Also related: my [summary](https://commoncog.com/peak-book-summary/) of K. Anders Ericsson's Peak, and [The Problems With Deliberate Practice](https://commoncog.com/the-problems-with-deliberate-practice/).*

## The Series

1. [Why Tacit Knowledge is More Important Than Deliberate Practice](https://commoncog.com/tacit-knowledge-is-a-real-thing/) — In which we talk about the existence of tacit knowledge, how it relates to expertise, and why the research around tacit knowledge extraction is more interesting and more underrated than the deliberate practice literature.
2. [Copying Better: How To Acquire The Tacit Knowledge of Experts](https://commoncog.com/how-to-learn-tacit-knowledge/) — The sort of tacit knowledge I am interested in is that of ‘expert intuition’. Because this is our goal, we should focus our attention on domains that are most focused on studying it. This post walks through the Recognition-Primed Decision Making model (RPD) from the field of Naturalistic Decision Making. The models helps us understand why expertise is tacit, and gives us levers to extract tacit knowledge from the heads of practitioners.
3. [The Three Kinds of Tacit Knowledge](https://commoncog.com/three-kinds-of-tacit-knowledge/) — It turns out there are three kinds of tacit knowledge, all of which are ‘things that cannot be captured through words alone’. This post is about definitions, and it is a summary of Harry Collins's *Explicit and Tacit Knowledge* — the best book I've read on the topic.
4. [How to Use YouTube to Learn Tacit Knowledge](https://commoncog.com/youtube-learn-tacit-knowledge/) — We can't really talk about tacit knowledge today without talking about YouTube. We take a look at using YouTube to learn Judo, computer programming, and music, before generalising several principles from all three domains.
5. [An Easier Method for Extracting Tacit Knowledge](https://commoncog.com/an-easier-method-for-extracting-tacit-knowledge/) — We discuss Laura Militello and Robert Hutton's Applied Cognitive Task Analysis (ACTA), a simplified method for extracting the tacit expertise of others.
6. [The Tricky Thing About Creating Training Programs](https://commoncog.com/creating-training-programs/) — We take a look at how you might turn extracted, tacit expertise into a training program for yourself or others.
7. [An Extracted Tacit Mental Model of Business Expertise](https://commoncog.com/tacit-mental-model-of-business/) — We take a look at the work of Lia DiBello, an NDM researcher who specialises in the tacit mental model of business expertise. As it turns out, great business practitioners all share the same common mental model of business. We take a look at what that is.
8. [Tacit Expertise Extraction, Software Engineering Edition](https://commoncog.com/tacit-expertise-extraction-software-engineer/) — An interview with Stephen Zerfas, who used the Recognition-Primed Decision Making model to accelerate his software engineering skills at Lyft.
- [John Cutler's Product Org Expertise](https://commoncog.com/john-cutlers-product-expertise/) — An example of using ACTA to extract intuitive, tacit expertise.
- [Accelerated Expertise](https://commoncog.com/accelerated-expertise/) — A summary of the best book on NDM training methods currently available.
  

Continue Reading: 👓 Commoncog’s Best Series

- [The Business Expertise Series](https://commoncog.com/business-expertise-series/) — All the best businesspeople share a common mental model of business. This is what we know of that model, and why it works.
- [Becoming Data Driven in Business](https://commoncog.com/becoming-data-driven-in-business/) — The data literacy course you never had.
- [A Framework for Putting Mental Models to Practice](https://commoncog.com/a-framework-for-putting-mental-models-to-practice/) — Mental model thinking is typically a fad. Here’s how to actually put Munger’s observations to practice.

Originally published , last updated .

digraph tacitness {
    rankdir=LR;
    node [shape=box, style=rounded];
    Q1 [label="Has knowledge asset\\nbeen explicated?"];
    Q2 [label="Is it easy to\\nreplicate?"];
    Q3 [label="Is it easy to\\narticulate?"];
    Q4 [label="Is it endemic to the\\nfirm’s culture?"];
    Q5 [label="Are its origins\\ndeep and obscure?"];

    Low     [shape=ellipse, label="Low"];
    LowMed  [shape=ellipse, label="Low/Medium"];
    Med     [shape=ellipse, label="Medium"];
    MedHigh [shape=ellipse, label="Medium/High"];
    High    [shape=ellipse, label="High"];
    Ultra   [shape=ellipse, label="Ultra"];

    Q1 -> Low     [label="yes"];
    Q1 -> Q2      [label="no"];

    Q2 -> LowMed  [label="yes"];
    Q2 -> Q3      [label="no"];

    Q3 -> Med     [label="yes"];
    Q3 -> Q4      [label="no"];

    Q4 -> MedHigh [label="no"];
    Q4 -> Q5      [label="yes"];

    Q5 -> High    [label="no"];
    Q5 -> Ultra   [label="yes"];
}

Tacitness                                                       Degree
Has knowledge asset been explicated? 
├─ yes ─────────────────────────────────────────────────────►   Low
└─ no
   └─ Is it easy to replicate?
      ├─ yes ───────────────────────────────────────────────►   Low/Medium
      └─ no
         └─ Is it easy to articulate?
            ├─ yes ─────────────────────────────────────────►   Medium
            └─ no
               └─ Is it endemic to the firm’s culture?
                  ├─ no ────────────────────────────────────►   Medium/High
                  └─ yes
                     └─ Are its origins deep and obscure?
                        ├─ no ──────────────────────────────►   High
                        |
                        |      
                        └─ yes ─────────────────────────────►   Ultra


---
title: "Why Tacit Knowledge is More Important Than Deliberate Practice"
source: "https://commoncog.com/tacit-knowledge-is-a-real-thing/"
author:
  - "[[Cedric Chin]]"
published: 2020-06-09
created: 2025-05-30
description: "What tacit knowledge is, and why it is the most interesting topic in the study of expertise today."
tags:
  - "clippings"
---
By

![Feature image for Why Tacit Knowledge is More Important Than Deliberate Practice](https://commoncog.com/content/images/size/w600/2020/06/tacit_knowledge.jpg)

## Table of Contents

## Fan of great business?

Join 8,000+ sharp investors and operators like yourself, and we'll send you a collection of Commoncog's best articles right away:

*Note: this is Part 1 in a [series about tacit knowledge](https://commoncog.com/the-tacit-knowledge-series/).*

I want to spend an essay talking about tacit knowledge, and why I think it is *the* most interesting topic in the domain of skill acquisition. If you are a longtime Commonplace reader, you’ll likely have come across this idea before, because I’ve written about it [numerous](https://commoncog.com/putting-mental-models-to-practice/) [times](https://commoncog.com/you-cant-teach-what-they-arent-ready-to-know/) [in](https://commoncog.com/the-mental-model-faq/) [the](https://commoncog.com/the-mental-model-fallacy/) [past](https://commoncog.com/chicken-sexing-and-perceptual-learning-as-a-path-to-expertise/). But I think it’s still good idea to dedicate a whole piece to the topic.

The reason I think it’s important to write this piece is because every time I touch on the topic of tacit knowledge, inevitably someone will pop up on Twitter or Hacker News or Reddit or email and protest that tacit knowledge doesn’t exist. I want something to link to whenever I come up against someone who says this, mostly so that I don’t have to repeat myself.

There is one other reason I think it is good to explore what tacit knowledge is: tacit knowledge *does* exist, and understanding that it does exist is one of the most useful things you can have happen to you. Once you understand that tacit knowledge exists, you will begin to see that big parts of *any* skill tree is tacit in nature, which means that you can go hunting for it, which in turn means you can start to ask the really useful question when it comes to expertise, which is: that person has it; that person is really good at it; how can I have it too?

## What is Tacit Knowledge?

Tacit knowledge is knowledge that **cannot be captured through words alone**.

Think about riding a bicycle. Riding a bicycle is impossible to teach through descriptions. Sure, you can *try* to explain what it is you’re doing when you’re cycling, but this isn’t going to be of much help when you’re teaching a kid and they fall into the drain while you’re telling them to “BALANCE! JUST IMAGINE YOU ARE ON A TIGHTROPE AND BALANCE!”.

Undoubtedly there will be people who say that if you just explain better, if you could find the right *words* to talk to your students, you can teach them to ride bikes. If you happen to believe this … well, I urge you to test this theory yourself. Go find a couple of kids (or adults!) who haven’t yet learned to ride a bike and see if you can teach them — through the power of your explanations alone! — how to cycle without scraping their shins. And if you are successful, *be critical with yourself*: how much of your success is due to the bike rider figuring it out on their own? And how much of it is due to your verbal instructions?

This explanation bit deserves some attention. In pedagogy, this is known as ‘ [transmissionism](https://andymatuschak.org/books/?ref=commoncog.com) ’, and it is regarded amongst serious educators with the same sort of derision you and I might have about flat-earth theories today. It goes something like this: some people believe that it is possible to teach technique by *explaining* things to others. They think that if you can find just the right combinations of words with the right sorts of analogies; if you can really break things down into the right level of atomic details, things will magically click in their students’s heads and they will succeed. Such people have likely never attempted to do so in a serious manner. And if you are one such person, then I hope that you pick up coaching at some point, in a sport that I am interested in, so that my kids can go up against your kids, and then I get to watch as *they absolutely crush yours.*

How do you teach tacit knowledge? How do you teach bike riding, for that matter?

When I was a kid, I taught myself how to ride a bike … by accident. And then I taught my sisters and then my cousin and then another kid in the neighbourhood who was interested but a little scared. They were zooming around in about an hour each. The steps were as follows:

1. Pick a small bike, much smaller than the body of your learner, and therefore not far off the ground. This is useful for the learner because they will be able to put their feet down and save themselves at any moment. It’s also less scary to be not far off the ground.
2. Have them push with their legs and go forward a short distance, again with their feet only inches off the ground. Do this repeatedly, by pushing off, putting both feet down, then stopping and repeating.
3. Make the pushes harder and the distances longer. Eventually, they will spend tens of seconds gliding with their feet held up. The goal here is have them learn how it feels like to balance on a bike.
4. On one of those glides, once they feel ready (and likely a little bored — the point is to get them to do step 3 so many times that they’re feeling secure and eager to move on) — well, once you reach that point, tell them to start peddling.
5. Voilà. You have a bicycling kid.

Notice how little verbal instruction is involved. What is more important is emulation, and action — that is, a focus on the embodied feelings necessary to ride a bicycle successfully. And this exercise was quite magical for me, for within the span of an hour I could watch a kid go from conscious incompetence to conscious competence and finally to unconscious competence.

In other words, tacit knowledge instruction happens through things like imitation, emulation, and apprenticeship. You learn by copying what the master does, blindly, until you internalise the principles behind the actions.

You might see this and think “ahh, this works for physical things like cycling and tennis and Judo, but what about more cerebral things?” To which I say: tacit knowledge exists all around us. Researcher Samo Burja explains tacit knowledge by way of example: he [calls it](https://medium.com/@samo.burja/the-youtube-revolution-in-knowledge-transfer-cb701f82096a?ref=commoncog.com) “the ability to create great art or assess a startup (…) examples include woodworking, metalworking, housekeeping, cooking, dancing, amateur public speaking, assembly line oversight, rapid problem solving, and heart surgery.”

If you are a knowledge worker, tacit knowledge is a lot more important to the development of your field of expertise than you might think.

## Tacit Knowledge in Knowledge Work

It took me years before I realised that my method of teaching people to bicycle contained a number of principles that might be applicable to other domains.

In my previous job, my technical lead, Hieu, had an uncanny ability to sit in on requirements meetings and, within minutes, sketch out a program structure that would be the simplest possible solution with the fewest moving parts. That sketch was often the one we ended up implementing, and yet I noticed that Hieu always left enough wiggle room for the inevitable changes that came with any software project. (To be clear, he usually implemented a throwaway prototype to confirm the sketch, before passing on the design). When I designed implementations, something always had to be redesigned later. I simply wasn’t as good. Eventually, I asked him how he did this, and tried multiple times to get it out of him over the years we worked together. Our conversations would inevitably go something like the following:

“Well” Hieu would begin, “When you hear there is an external API, you should focus your program around that because there is a lot of risk there.”

“Yeah but then why didn’t you worry about the calendar API?”

“Oh, because I’ve worked with it before and I think it is easy to implement.”

“Why focus so much on Firebase?”

“Because we want to use it as a database layer. Quite risky ah.”

“So always focus on a core layer first, because more important?”

“Yes. Try to focus on the most dangerous bits first.”

“But how come you didn’t worry about the inventory API? We’ve never integrated with that before.”

“Ya that one not that important now I think. The client might change it later. Or maybe our feature is going to change. We do the basics first.”

I thought back to my Viki days, when I was a software engineering intern and was writing software tests for the first time. A senior software engineer took a few seconds to look at about a hundred lines of code I’d written, and said “Oh, that’s not good, this would be a problem later. Structure it this way.”

I asked him how he knew, within five seconds, that it was bad. He gave me a long explanation about software engineering principles. I waved him away and asked how he did it *in five seconds*. He said “Well, it just *felt* right. Ok, let’s go to lunch, you can fix it afterwards.”

I’ve written about this Viki episode in my [post about perceptual learning](https://commoncog.com/chicken-sexing-and-perceptual-learning-as-a-path-to-expertise/#perceptual-exposure-as-natural-learning-strategy). I don’t mean to say that Hieu or the senior software engineer couldn’t explain their judgment, or that they couldn’t make explicit the principles they used to evaluate the tradeoffs between a dozen or so variables: they could, even if terribly. *My point is that their explanations would not lead me to the same ability that they had.*

Why is this the case? Well, take a look at the conversation again. When I pushed these people on their judgments, they would try to explain in terms of principles or heuristics. But the more I pushed, the more exceptions and caveats and potential gotchas I unearthed.

This is actually generalisable. People with expertise in any sufficiently complicated domain will always explain their expertise with things like: “Well, do X. Except when you see Y, then do Z, because A. And if you see B, then do P. But if you see A and C but not B, then do Q, because reason D. And then there are weird situations where you do Z but then see thing C emerge, then you should switch to Q.”

And if you push further, eventually they might say “Ahh, it just feels right. Do it long enough and it’ll feel right to you too.”

Eventually I realised that the way to learn Hieu’s techniques was to copy him: to design some software and then ask for his feedback. And I realised that if you ever hear someone explaining things in terms of a long list of caveats, the odds are good that you’re looking at tacit knowledge in action.

This phenomenon is actually well established in the study of expertise. It has also been written about, many times, by practitioners in other fields. As an example, here’s surgeon Atul Gawande on [appendicitis surgery](https://www.newyorker.com/magazine/2011/10/03/personal-best?ref=commoncog.com):

> Say you’ve got a patient who needs surgery for appendicitis. These days, surgeons will typically do a laparoscopic appendectomy. You slide a small camera—a laparoscope—into the abdomen through a quarter-inch incision near the belly button, insert a long grasper through an incision beneath the waistline, and push a device for stapling and cutting through an incision in the left lower abdomen. Use the grasper to pick up the finger-size appendix, fire the stapler across its base and across the vessels feeding it, drop the severed organ into a plastic bag, and pull it out. Close up, and you’re done. That’s how you like it to go, anyway. But often it doesn’t.  
>   
> Even before you start, you need to make some judgments. Unusual anatomy, severe obesity, or internal scars from previous abdominal surgery could make it difficult to get the camera in safely; you don’t want to poke it into a loop of intestine. You have to decide which camera-insertion method to use—there’s a range of options—or whether to abandon the high-tech approach and do the operation the traditional way, with a wide-open incision that lets you see everything directly. If you do get your camera and instruments inside, you may have trouble grasping the appendix. Infection turns it into a fat, bloody, inflamed worm that sticks to everything around it—bowel, blood vessels, an ovary, the pelvic sidewall—and to free it you have to choose from a variety of tools and techniques. You can use a long cotton-tipped instrument to try to push the surrounding attachments away. You can use electrocautery, a hook, a pair of scissors, a sharp-tip dissector, a blunt-tip dissector, a right-angle dissector, or a suction device. You can adjust the operating table so that the patient’s head is down and his feet are up, allowing gravity to pull the viscera in the right direction. Or you can just grab whatever part of the appendix is visible and pull really hard.  
>   
> Once you have the little organ in view, you may find that appendicitis was the wrong diagnosis. It might be a tumor of the appendix, Crohn’s disease, or an ovarian condition that happened to have inflamed the nearby appendix. Then you’d have to decide whether you need additional equipment or personnel—maybe it’s time to enlist another surgeon.  
>   
> Over time, you learn how to head off problems, and, when you can’t, you arrive at solutions with less fumbling and more assurance. After eight years, I’ve performed more than two thousand operations. Three-quarters have involved my specialty, endocrine surgery—surgery for endocrine organs such as the thyroid, the parathyroid, and the adrenal glands. The rest have involved everything from simple biopsies to colon cancer. For my specialized cases, I’ve come to know most of the serious difficulties that could arise, and have worked out solutions. For the others, I’ve gained confidence in my ability to handle a wide range of situations, and to improvise when necessary.

Notice how Gawande includes all sorts of caveats in his explanation of his expertise. This is probably tacit knowledge in action. Learning this type of complicated judgment — this instantaneous solution selection that happens to balance dozens of considerations against each other — this is what is valuable to learn. And it is almost impossible to learn it through explanation alone.

![Explicit and tacit knowledge triangle](https://commoncog.com/content/images/2018/11/Different-types-of-knowledge.png)

## Can Tacit Knowledge Be Made Explicit?

It is worth it to re-examine that last sentence, above. *Could it — in principle — be possible to externalise tacit knowledge into a list of instructions?* Perhaps we could extract expert decision-making into a multi-branched process, and code that into an ‘expert system’. Or perhaps we could turn it into a process list, and give that to every practitioner in the field, instead of having them learn tacit knowledge the traditional way.

The consensus answer to that question seems to be: “Yes, in principle it is possible to do so. In practice it is very difficult.” My take on this is that it is *so* difficult that we shouldn’t even bother; assuming that you are reading this because you want to get good in your career, you should give up on turning tacit knowledge into explicit knowledge and just go after tacit knowledge itself.

Why do we know this?

In the 1970s, a bunch of organisations — amongst them the US military — commissioned a number of studies to look into the possibility of building out all sorts of expert systems to augment or replace human agents. This notion of replacing humans with expert systems was a bit of a fad at the time, the same way that neural networks are big on the hype train today, but expert systems have fallen to the wayside in the decades since.

What many researchers found in the wake of that fad was that it is extremely difficult to encode *all* the possible branches and gotchas and nuances from a human expert into an expert system.

One of these projects, commissioned shortly after the Yom Kippur war, was to enter log books for Israeli aircraft maintenance into a US database, with the goal of perhaps replacing aircraft maintenance officers with an expert system. And a young researcher at the time — by the name of Gary Klein — was tasked with the job of interviewing aircraft maintenance officers to get at the core of their expertise.

He describes that project in a recent [podcast episode](https://naturalisticdecisionmaking.org/the-ndm-podcasts-archive/?ref=commoncog.com#Gary-Klein-from-ShadowBox-LLC):

> “I could fill out 80% of the forms, but the other 20%... maybe something was left blank … or maybe there was an error. And then I would interview them and ask “how did you fill this in?”  
>   
> And then they (the maintenance officers) said: “OK look at this damage. Look at the size of the damage. And then the plane was back in service two days later. Now for this kind of damage, I know, if it severed any of these lines, it’s going to take a week or two (to repair). So obviously it didn’t sever those lines. So here’s the way the damage must have propagated, that they could get it back in service (so quickly).”  
>   
> In other words, these guys could go beyond the logbooks, and imagine what it was like to be a maintenance technician, working on these kinds of problems.  
>   
> (…) So when I wrote my report, the report was: here’s what I could do that was formulaic, but here are the critical incidents that I collected, so you can understand their expertise. And as a result the Air Force decided they could not build an expert system to replace these guys, and they had to keep paying them.”

You can sort of squint and see this result being repeated across many organisations, in many domains, during the same period. (Wikipedia calls this problem the ‘ [knowledge acquisition problem](https://en.wikipedia.org/wiki/Expert_system?ref=commoncog.com#Disadvantages) ’, which is a nice way of putting it; it was what ultimately caused expert systems to decline in popularity). As people rapidly discovered, it wasn’t so easy to get the ‘rules’ out of experts’s heads in the first place.

But there are other objections, of course. Klein — now considered one of the pioneers of the Naturalistic Decision Making (NDM) branch of psychology — likes to say that over-reliance on procedures makes human operators fragile (Chapter 15, *[Source of Power](https://www.goodreads.com/book/show/65229.Sources_of_Power?ref=commoncog.com)*). In other words, giving people a list of procedures to execute, blindly, denies them the ability to build expertise, which in turns prevents them from doing the sorts of creative problem solving that is common amongst expert operators. It also means that when something goes drastically wrong — and something *always* goes drastically wrong in the real world — they would not be able to adapt.

Are there indications that this isn’t a completely hopeless enterprise? Yes, it turns out there are:

1. I have friends in artificial intelligence research who argue that it *is* possible for expert systems to make a comeback, perhaps by using more contemporary AI methods.
2. Gary Klein himself has made a name developing techniques for extracting pieces of tacit knowledge and making it explicit. (The technique is called ‘The Critical Decision Method’, but it is difficult to pull off because it demands expertise in CDM itself).
3. It is sometimes possible for a particularly gifted person to synthesise an entire field’s worth of tacit knowledge into an effective and explicit method of instruction. Arguably, [John Boyd](https://en.wikipedia.org/wiki/John_Boyd_\(military_strategist\)?ref=commoncog.com) did this with the first fighter tactics manual for the US Air Force — where before people thought that dogfighting was an art that couldn’t be organised into explicit principles. But Boyd somehow managed to do so anyway, after a couple of years of teaching, and a full year of writing.

All of these points are valid, and worth thinking about — but whether or not they turn out to be generally correct is irrelevant, I think. You may say “actually, since all tacit knowledge can be made explicit, there is no such thing as tacit knowledge” — but that is a pedantic exercise that is uninteresting to me. It is not reasonable to wait for an expert systems revival, nor is it fruitful to expect CDM to be applied to your field, or for a Boyd-like genius to appear. We should act as if tacit knowledge were a fact, because it is more useful to think about ways to gain that tacit knowledge directly, instead of hoping for some breakthrough to make tacit knowledge explicit.

## Learning Tacit Knowledge

What does this mean for us? It means that we should start looking into the published research on tacit knowledge if we want to pursue expertise in our fields.

“But wait,” I hear you say — “What about the field of deliberate practice? Isn’t *that* the predominant subfield most concerned with the development of expertise?” And the answer to that is *no*, it is not.

In my review of Ericssons’ [*Peak*](https://commoncog.com/peak-book-summary/), and in my summary of [The Problems with Deliberate Practice](https://commoncog.com/the-problems-with-deliberate-practice/), I explained that deliberate practice is defined as possible *only* in domains with a long history of well-established pedagogy. In other words, deliberate practice **can only exist in fields like music and math and chess**.

K. Anders Ericsson lays out this narrow definition in *Peak*, and then does a cop-out, arguing that while he hasn’t studied practice outside of such domains, the ideas from deliberate practice may be applied to pedagogically less established fields. But Ericsson is well aware that NDM methods exist  — he was one of the editors, alongside many names from the NDM community — who worked on the *[Cambridge Handbook for Expertise and Expert Performance](https://www.cambridge.org/my/academic/subjects/psychology/cognition/cambridge-handbook-expertise-and-expert-performance-2nd-edition?format=HB&isbn=9781316502617&ref=commoncog.com)*.

And so if you are a programmer, or designer, or businessperson, an investor or a writer reading about deliberate practice, you may be asking: “Well, what about *my* field? What if there are no established pedagogical techniques for me?” And if you have started to ask this question, then you have begun travelling a more interesting path; this is really the right question to ask.

The answer, of course, is that the field of NDM is a lot more useful if you find yourself in one of these fields. The process of learning tacit knowledge looks something like the following: you find a master, you work under them for a few years, and you learn the ropes through emulation, feedback, and osmosis — *not* through deliberate practice. (Think: Warren Buffett and the years he spent under Benjamin Graham, for instance). The field of NDM is focused on ways to make this practice more effective. And I think much of the world pays too much attention to deliberate practice and to cognitive bias research, and not enough to tacit knowledge acquisition.

If tacit knowledge exists — and I believe it does — then the most useful tools about skill acquisition will come out of the fields that study it. Late last year, the NDM community got together and published *[The Oxford Handbook of Expertise](https://www.oxfordhandbooks.com/view/10.1093/oxfordhb/9780198795872.001.0001/oxfordhb-9780198795872?ref=commoncog.com).* It is the single most comprehensive overview of the field that we know today.

It took about 30 years before Ericsson’s work about deliberate practice penetrated mainstream consciousness, due in part to the success of Malcolm Gladwell’s *[Outliers](https://en.wikipedia.org/wiki/Outliers_\(book\)?ref=commoncog.com)*. If we use that as a comparison, how long before NDM methods seep into the mainstream?

Perhaps we’ll talk about tacit knowledge in 2030 the same way people talk about the ‘10,000 hour rule’ today. But if you’re reading this post, you’ll have a head start. Keep an eye out on NDM methods, and watch for things that focus on tacit knowledge. They are — in this writer’s opinion — the most interesting, overlooked topic in expertise today.

---

*Note: this is the first entry in a series of posts about tacit knowledge, and what is known about how to learn it effectively. Subscribe to the [Commonplace newsletter](https://commoncog.com/subscribe-to-the-commonplace-newsletter/) if you want updates; if you can't wait, I've written about NDM's methods in Parts [4](https://commoncog.com/putting-mental-models-to-practice/) and [5](https://commoncog.com/putting-mental-models-to-practice-part-5-skill-extraction/) of my series on [Putting Mental Models To Practice](https://commoncog.com/a-framework-for-putting-mental-models-to-practice/). But, be warned: those two posts are embedded in a 30,000 word series that summarises large parts of the judgment and decision making literature. If you don't want to read all of that, you might just want to wait for the next one.*

***Update:***[Part 2 is out here](https://commoncog.com/how-to-learn-tacit-knowledge/).

Originally published , last updated .

This article is part of the [Expertise Acceleration](https://commoncog.com/expertise) topic cluster. Read [more from this topic here→](https://commoncog.com/expertise)

Previous Post

#### ← Every Actionable Book is Actually Two Books Inside

Next Post

[![Feature image for Tacit Expertise Extraction, Software Engineering Edition](https://commoncog.com/content/images/size/w300/2024/02/tacit_knowledge_software_engineering.jpg)](https://commoncog.com/tacit-expertise-extraction-software-engineer/)[![Feature image for One Definition of Wisdom](https://commoncog.com/content/images/size/w300/2024/02/charlie_munger_wisdom.jpg)](https://commoncog.com/one-definition-of-wisdom/)[![Feature image for Mental Strength in Judo, Mental Strength in Life](https://commoncog.com/content/images/size/w300/2023/05/mental_strength_judo_life-1.jpg)](https://commoncog.com/mental-strength-judo-life/)[![Feature image for Creating New Drills for Deliberate Practice](https://commoncog.com/content/images/size/w300/2023/04/deliberate_practice_create_new_drills.jpg)](https://commoncog.com/creating-drills-deliberate-practice/)

---
title: "Copying Better: How To Acquire The Tacit Knowledge of Experts"
source: "https://commoncog.com/how-to-learn-tacit-knowledge/"
author:
  - "[[Cedric Chin]]"
published: 2020-06-16
created: 2025-05-30
description: "Much of expertise is tacit: that is, it cannot be captured through words alone. We look at techniques to learn the tacit knowledge of experts."
tags:
  - "clippings"
---
By

![Feature image for Copying Better: How To Acquire The Tacit Knowledge of Experts](https://commoncog.com/content/images/size/w600/2020/06/blacksmith_expertise_tacit_knowledge.jpg)

## Table of Contents

## Fan of great business?

Join 8,000+ sharp investors and operators like yourself, and we'll send you a collection of Commoncog's best articles right away:

*Note: This is part 2 in a [series about tacit knowledge](https://commoncog.com/the-tacit-knowledge-series/). Read [Part 1](https://commoncog.com/tacit-knowledge-is-a-real-thing/) here.*

Let’s say that you believe tacit knowledge exists. Let’s also say that you agree that tacit knowledge is mostly learnt through emulation, apprenticeship and osmosis.

The question that follows naturally is this: are there ways to get better at it? Is there research that tells us how to get better at emulation, or to be more effective at apprenticeship? I mentioned in my previous piece that the field of Naturalistic Decision Making (NDM) is the most promising subfield of expertise research I’ve found; I also mentioned that learning ideas and methods from NDM has made the biggest difference in my own pursuit of mastery.

This post will explore what those methods are.

## Two Caveats

Before I begin, two quick caveats.

First, I must admit that my usage of the term ‘tacit knowledge’ isn’t ideal. ‘Tacit knowledge’ comes from the field of epistemology, which is the philosophical study of truth. By defining ‘tacit knowledge’ as ‘knowledge that cannot be captured through words alone’, I’ve left myself open to all sorts of confusion and pedantry.

I was aware of this when I chose the term for this series.

Let’s take a quick tour of all the ways ‘tacit knowledge’ is confusing, just so you know that I am skipping over large parts of the philosophical literature:

- Tacit knowledge may be ‘embodied’ — that is, activities like riding a bike or swinging a tennis racquet have to do with the body, but not necessarily the conscious mind. Some philosophers believe that such skills are a type of tacit knowledge that is different from the tacit knowledge in more cerebral fields like programming and designing and writing. (I do not believe this is true, but the philosophical argument for why they believe this is the case will take around 50 pages of writing, so we’re not going to go there).
- Many types of tacit knowledge may be made explicit: for instance, bike riding or ball kicking may be turned into a collection of physics equations in a research paper. This is a form of explicit knowledge, and it may help you build a bike-riding or ball-kicking robot. But from a pedagogical perspective, this form of explicit knowledge is useless: it won’t help you learn to ride a bike or kick a ball. So is this *really* tacit knowledge? Or is it explicit? Or is it a bit of both?
- What about the kinds of tacit knowledge that can be communicated via video, or through pictures? I may not be able to *explain* to you how to do a complicated series of yoga poses through words alone, but if I record a quick video, or draw a series of diagrams, you might be able to learn to do so. Is such knowledge tacit, or explicit?
- And how about organisational tacit knowledge? In 1988, Taiichi Ohno published *The Toyota Production System*, which laid out the ideas behind lean manufacturing for the first time. He did so after resisting codification of those principles for many, many years; Ohno feared that the publication of TPS would destroy Toyota’s competitive advantage. But it turned out that the publication of Ohno’s book by itself did not lead to widespread adoption amongst Toyota’s competitors. More bizarrely, competitors who poached TPS experts away from Toyota were not able to replicate Toyota’s system in their company. It appeared that TPS required an organisational culture to develop alongside the implementation of the system itself. This phenomenon sparked off an entire subfield of research into organisational tacit knowledge.

I’ll skip ahead here and tell you that none of these questions are particularly interesting to me — at least, not in the context of this blog. This blog is a practitioner’s blog: it is interested in what is useful, not what is ‘true’. The answers to these questions will matter to you if you are a philosopher, or if you are a pedantic commenter in some online forum and you are familiar with Wittgenstein, Dreyfus, or Polanyi. But if you are a practitioner and you observe someone in your company demonstrating high levels of expertise, you probably aren’t interested in the answers to any of these things. You just want to learn how to acquire what is in your more skilled colleague’s head.

So here’s my way of side-stepping everything above: when I say tacit knowledge in this series, what I am really interested in is ‘expert intuition’ or ‘expert judgment’. This is the “expertise, full of caveats” that I talked about in [Part 1](https://commoncog.com/tacit-knowledge-is-a-real-thing/). Expert intuition is tacit because it is incredibly difficult to investigate, incredibly difficult to make explicit, and incredibly difficult to teach.

In principle, it is possible to make many forms of expert intuition explicit: if you can find just the right sort of skilled practitioner, or if you have access to someone who is trained in the NDM tradition, he or she may be able to extract and explain the tacit mental models of expertise that is responsible for the practitioner's performance. But this leaves us with two problems:

1. First, you may not find such a person in your domain. NDM research is considered niche, even in the field of judgment and decision making; people with both high levels of expertise and the ability to synthesise that expertise into discrete principles are also exceedingly rare.
2. Second, such an explanation might not be helpful to you if you want to gain their skills. (Think of the physics equations around bike riding — none of that is going to help you learn to ride a bike!) Similarly, an expert programmer might explain her judgment — full of caveats and gotchas! — but this might not be very useful to you.

I’ve mentioned both of these problems in passing in my [previous piece](https://commoncog.com/tacit-knowledge-is-a-real-thing/). The crux of it, I think, is that regardless of whether or not tacit knowledge in your domain may be made explicit, it is more useful for you to act as if it is were not possible. This stance frees you to hunt for learning methods that do not involve explanations alone.

My second caveat is the following: even if tacit knowledge is incredibly confusing, *it is still useful to introduce it*. I’ve learnt that people aren’t typically interested in the ideas from NDM when I explain it to them. But if I first explain the concept of tacit knowledge — the idea that some forms of knowledge, embedded in human expertise, cannot be captured through words alone — they immediately map this phenomenon to their own experiences, and then they become more receptive to my gushing about NDM training methods.

So, with these two caveats in mind: let us begin.

## How The Naturalistic Decision Makers Do It

In order to get better at learning through emulation, we need to understand the *nature* of the tacit knowledge that we are after. The dangerous thing about calling this sort of tacit knowledge ‘expert intuition’ is that it is very easy to hear ‘intuition’ and go “oh, so there’s nothing we can do to learn it, move along now.”

But this is *not* true. The entire field of NDM research is built around taking expert intuition, extracting it and turning that into training programs for organisations (in most cases, the military). In many of these projects, the organisations measure training outcomes, find improvements, pay the researchers, and then have the researchers move on to the next project.

How do they do this?

As far as I can tell, NDM researchers have three tools in their toolbox. First, they conduct something called ‘Cognitive Task Analysis’ (CTA), which is a set of interviewing techniques designed to explicate tacit mental models of expertise. This is not an easy task. An earlier version of this was called the ‘Critical Decision Method’, which focused on ‘tough cases’ of expertise-driven performance, in order to elicit whatever it is that was going on in the practitioners’s heads. Today, the set of techniques is much larger, and they are all lumped together under the umbrella of CTA.

If you study the development of psych approaches, CTA is simply an extension of older techniques used by the behaviourists to analyse psychological effects under experimental conditions. (Paul Harmon, for instance, talks about this evolution [here](https://www.bptrends.com/working-minds-a-practitioners-guide-to-cognitive-task-analysis-by-beth-crandall-gary-klein-and-robert-r-hoffman/?ref=commoncog.com)). NDM researchers typically perform cognitive task analysis at the outset of their projects — this is how they extract tacit knowledge from the experts they study.

Second, many NDM researchers rely on a model of expert intuition called the recognition-primed decision making model (RPD). This model, developed by Gary Klein and his collaborators in the 90s, lies at the heart of their understanding of expert intuition. RPD explains how expert intuition works, and it tells us why experts have so much difficulty when it comes to explaining their expertise. NDM practitioners use the RPD model as a guide during their interviewing process, and they also use it as a guide to the development of their training programs.

Third, and last, NDM researchers develop training programs designed to let less experienced practitioners acquire expert judgment of their own. NDM practitioners typically have some familiarity with pedagogical techniques, but in this, they, too, are guided by the RPD model.

It’s a common misconception to look at this and think: “oh, when you can make tacit knowledge explicit, then you can just explain things to people and they will get it.” This is not the right conclusion to have. As we shall soon see, large portions of expertise is tacit, and for certain types of expertise, the training methods must *also* be tacit in nature — that is, built around copying, and emulation, and scenario training, not explicit instruction. What NDM gives us is a rigorous method to reason about such things.

Because RPD is so central to the work of NDM researchers, we should probably start with the model and explore everything that it implies if we want to use it ourselves.

## The Recognition-Primed Decision Making Model

Imagine that you were in my shoes, and you were seated next to a senior programmer who skimmed your code and gave you several recommendations after a couple of seconds. “How did you do that?” you ask, and he replies: “Oh, it just *felt* right.”

In order to get at his expertise we must ask: what went on in his head during those few seconds? The RPD model gives us a clue.

The recognition-primed decision making model describes what humans do when they are problem solving in the real world. It tells us that when an expert encounters a problem in the wild, their brain observes the situation in a changing environment and immediately pattern matches it against a collection of *prototypes*. If they recognise what they see as an example of a prototype — even if the situation they see is non-routine! — their brain immediately generates four things:

1. **A set of ‘expectancies’** — When diagnosing a situation, experts will construct mental simulations of how the events have been evolving and will continue to evolve. In other words, they will have some expectations for what happens next. The more experienced they are, the more clear-cut these expectancies become. For example, an experienced firefighter might take in a scene and know instantly where the fire might travel, or how a bad situation might develop. A programmer reading a codebase might find several weird kludges, and immediately suspect a submodule to be a persistent source of bugs.
2. **A set of plausible goals** — The expert would know what to prioritise in the moment, and what to defer to a latter time. When under fire, for instance, a Marine Corps squad leader would have to prioritise between keeping his people alive, getting to better cover, and achieving mission objectives. His recognised prototype tells him where his priorities lie in a given situation, freeing cognitive resources to conduct other forms of thinking. Similarly, a programmer may receive a set of business requirements, and immediately generate a prioritised list of goals in their head according to the recognised prototype.
3. **A set of relevant cues** — Experts know what to pay attention to; novices do not. Recognised prototypes come with a set of cues — for instance, when you’ve just started driving, you may find yourself overwhelmed with the dials and knobs and mirrors you have to keep track of. After a few months, however, you do these things automatically, and shift your attention to specific affordances in your car depending on the situation. (For instance, when turning a corner, you know to check your side mirrors and you know what to watch out for.)
4. **An action script** — Last, but not least, if the situation is typical, the expert would have a course of action immediately generated in their heads. If the situation is *not* typical, the expert’s brain would still generate a set of actions, but the expert would slow down to walk through each action step in their head.

The recognition of goals, cues, expectancies and actions is part of what it means to recognise a situation. What is important to understand is that *all of this recognition happens within implicit memory.* This is why experts are not able to verbalise what they are doing.

![](https://commoncog.com/content/images/2019/01/Paper.Commonplace.42.png)

Implicit memory operations are subconscious. Our ability to recognise faces, for instance, is an implicit memory operation, and we cannot say how it happens. When your friend Mary walks into the room, you immediately recognise her face. But notice that remembering her *name* is a different process entirely: this is because facial recognition is *recognition*; name retrieval is *recall*, and the two operations use different subsystems in your brain. ([Gillund & Shiffrin, 1984](https://pdfs.semanticscholar.org/7ebc/6fb776eb2250da7c925946424b9fee37097c.pdf?ref=commoncog.com))

When an expert says *“it just felt right”*, what they mean to say is that they recognised the problem as an example of a prototype in their heads, which generated the four by-products; this implicit memory operation happens so quickly that they cannot verbalise how they came up with it. They can only say “it just felt right!”, the same way you might say of an acquaintance: “I *know* her, I just can’t remember her name!”

The RPD model that we have explored so far is not complete. Experts sometimes come across situations that do not match what they have in their heads. Or, they uncover evidence that violates their original expectancies. In such an event, the expert immediately recognises that their matched prototype is mistaken, and so they drop back down to pattern recognition. They try to construct a narrative or a simulation to explain what they are seeing in front of them, in order to match with a different prototype. This is where a firefighter or programmer or Marine Corps squad leader might choose to gather more information — the firefighter might take a step back to look for other signs of smoke; the programmer might choose to dig deeper at a particular subsystem; the squad leader might decide to scout by a different path.

Our RPD model now looks like this. Everything in red is the second variation of the RPD model, which includes repeated cycles of pattern matching:

![](https://commoncog.com/content/images/2019/01/Paper.Commonplace.40.png)

The third, and last variation of the RPD model concerns itself with *actions*. In normative decision-making theory, humans pick multiple choices, evaluate their pros and cons against each other, and then pick the best possible option, perhaps by using some expected value calculation. In the real world, however, Klein and his fellow researchers discovered that experts do *not* default to this strategy. Instead, they ‘ [satisfice](https://en.wikipedia.org/wiki/Satisficing?ref=commoncog.com) ’ — they pick the first available option that fits their criteria.

To be more concrete, what happens is that the expert will pick a course of action, simulate it step-by-step in her head, and then either choose it, or discard it due to inadequacy. They will repeatedly go through each imagined course of action until they find the first one that meets all their criteria. If they run out of actions to perform, they will fall back to prototype recognition. But if they find an action that seems to work in their mental simulations, they will immediately carry it out.

![](https://commoncog.com/content/images/2020/06/Paper.Commonplace.41.png)

Klein explains (*Sources of Power*, Chapter 7) that a strategy like ‘cycle through each course of action one at a time, until a good one is found’ is more common when:

- Time pressure is greater.
- People are experienced in the domain. This makes them more confident in their judgments, and therefore allows their brains to default to this problem solving strategy.
- The conditions are more dynamic (meaning the context shifts rapidly, rendering decision analysis costly, because everything has to be thrown out).
- When the goals are ill-defined.

In contrast, people are more likely to use comparative evaluation when:

- They have to justify their choices — perhaps to a boss.
- When conflict resolution is at play (you have to satisfy multiple stakeholders with different requirements).
- When people have less experience, and therefore less confidence in the options that their brains generate.
- When they have to optimise — that is, when they are required to find the *best* course of action.
- And, finally, when the situation is computationally complex — for instance, when analysing an investment portfolio to select the best strategy. Tasks like these involve so many considerations that they tend to overwhelm the pattern-recognition machinery in our heads.

In most cases, however, the default human response is to satisfice: that is, to generate and then cycle through actions one at a time until the first suitable one is found. This is both good or bad, depending on the situation: in fact, much of the cognitive biases and heuristics tradition argue that this method of thinking is what leads to erroneous human judgment. Klein simply points out that *this is how our brains work*.

This final discussion concludes our examination of the recognition-primed decision making model. The complete model now looks like this:

![](https://commoncog.com/content/images/2019/01/Paper.Commonplace.41.png)

RPD is useful because it *gives us a model with which to understand human expertise*. When you apprentice under someone, what’s actually happening is that you are building up prototypes in your implicit memory — that is, you are identifying cues, learning plausible goals, internalising action scripts, and storing expectancies. At the same time, you are also collecting the experiences necessary to simulate action scripts in your head.

Klein mentions that RPD is similar to other models of expertise, such as [Lee Beach and Terry Mitchell](https://www.sciencedirect.com/science/article/abs/pii/0001691887900345?ref=commoncog.com) on image theory, the [skills-rules-knowledge scheme](http://nas.psych.uidaho.edu/~ad.uidaho.edu%5Cbdyre/psyc562/readings/Human_Performance_Models/Rasmussen_1983.pdf?ref=commoncog.com) by Jens Rasmussen, and other models of expertise by [P. A. Anderson](https://www.jstor.org/stable/pdf/2392618.pdf?ref=commoncog.com), [Wohl](https://ieeexplore.ieee.org/document/4308760?ref=commoncog.com), and [Dreyfus and Dreyfus](https://www.goodreads.com/book/show/1039572.Mind_Over_Machine?ref=commoncog.com). According to Klein (*Sources of Power*, Chapter 7), RPD’s contribution is merely that:

- It appears to describe the decision strategy used most frequently by people with experience.
- It explains how people can use experience to make difficult decisions.
- It demonstrates that people can make effective decisions **without using a rational choice strategy.***(emphasis mine)*

Which begs the question: how do we use this?

## How Do We Use RPD?

One of the most obvious things that falls out of the RPD model is that we can now use it to carry out cognitive task analysis. In other words, we *actually have a shot* at understanding what went through my senior programmer’s head.

When interviewing an expert, NDM practitioners will always zero in on ‘tough cases’ that the expert has experienced, in order to examine the recognition-driven process in their heads. Specifically, CTA interviewers will establish a timeline of events of the case, and then ask questions to elicit:

- What **cues** did they notice first?
- In that particular situation, what did they expect to happen next? (**expectancies**)
- What priorities (**goals**) did they have?
- What courses of **actions** immediately came to mind?

These four by-products is *how you get at the tacit knowledge of practitioners*.

Later, when they are designing training programs, NDM researchers would be able to rely on these explicated prototypes and the four by-products to come up with effective training scenarios. The goal, of course, is to **expand the set of experiential prototypes in their learners’s heads**.

The RPD model should immediately suggest certain training methods to you. Let’s go through some of these levers, which I think should be obvious if you pause to think about them:

### 1) Systematically Expand the Set of Prototypes You Have

The first method of application follows easily from the model: because so much of expertise is pattern matching, many of the training programs that are designed by NDM practitioners are oriented around expanding the set of prototypes in their learners’s heads.

In 1993, NDM pioneer Beth Crandall studied a group of expert neonatal intensive care unit (NICU) nurses who could ‘just tell’ if a premature baby was suffering from sepsis — way before other nurses or doctors could. The nurses insisted that they were able to tell from ‘intuition’ and ‘experience’, and that was all there was to it.

Crandall conducted a series of CTA interviews with them and successfully extracted the perceptual cues they noticed. She was then told to turn this into a presentation to be used for onboarding all the NICU nurses at the hospital group she was working with. Several perceptual cues these nurses were using to identify sepsis were unknown to the medical literature at the time; the nurses *themselves* did not know what they were looking for, since they were responding to a ‘gestalt’ — an overall picture that the baby presented.

But Crandall made an important modification to her presentation. She did not present the cues in isolation, like a lecturer would in a more typical talk. Instead, she presented the cues through a series of stories, because stories more accurately captured the sorts of experiences nurses would have in the field. (e.g. “You first notice baby George breathing shallowly. He looks normal, but his blood pressure is slightly elevated. You check his chart …”)

What Crandall was trying to do with these stories was to expand the set of prototypes in her audiences’s minds. The perceptual cues she had made explicit was only the tip of the iceberg — the key was to embed them within plausible scenarios that the nurses might encounter. Later, less experienced nurses could apprentice under more experienced ones, and they would share a vocabulary of perceptual cues that was made explicit by Crandall’s work.

How do you use this insight if you are a solo practitioner? One way could be to systematically seek out situations where you have less experience, in order to expand the set of prototypes in your head. Another way might be to select and use exercises that help build out the prototypes you have; for example, if you are a programmer, [code katas](http://codekata.com/?ref=commoncog.com) might be useful — but only to the degree that they reflect the problems you might encounter in your domain.

In practice, attempting to use CTA to extract and then create training programs is difficult. CTA is itself a skill, and most NDM practitioners say that you need months if not years of experience to become proficient at it. That said, I’ve experimented a little with this, and I think there are a few other ways you can put it to use:

### 2) Identify When A Practitioner Has a Prototype That You Do Not

Another, less obvious way to use RPD is to identify when a practitioner is using her intuition. When you notice a practitioner making a rapid assessment of a problem, when they say “it just *felt* right”, or when they give you an explanation that is full of caveats and gotchas, you now know that they are using an implicit-memory recognition operation that stems from their experiences. If you face a similar situation and you find yourself doing option comparison, this should tell you that your colleague has a prototype that you do not, and that this might be something you want to acquire.

In other words, you can use this to build a skill tree for yourself.

RPD is useful because you should *also* know to ask about the cues, expectancies, priorities, and courses of actions that present themselves in the practitioner’s mind. Don’t get me wrong: this will not be a good extraction of their tacit knowledge, because expert intuition is difficult to explicate, and CTA is itself difficult. But it’s surprisingly useful to just ask questions along these lines — at the very least, the hints that you’ll get out of it will be marginally better than naive copying, or asking “how *did you do that?!*”with no conception of the four by-products.

### 3) Get Better At Mental Simulations

A second lever that falls out of the RPD model is the idea of giving learners sufficient experiences in order to *simulate effectively*.

Recall that mental simulation is a key part of RPD: it drives the identification of expectancies, and it allows practitioners to simulate possible courses of action before picking one to execute. More experiences mean more accurate mental simulations.

As a result of Klein’s research, Marine Corp squad leaders are now trained to identify *decision requirements* for their own performance. Klein writes:

> The decision requirements exercise is for the squad leaders to identify the key judgments and decisions facing them, why they are difficult, and where they can go wrong. These decision requirements are the high drivers, the specific decision skills that they need to polish. In addition, by identifying the decision requirements of their mission (e.g., identifying a good landing zone for helicopters, judging the amount of time it will take a squad to move from one position to another), the squad leaders could find ways to practice these judgments, such as getting feedback from helicopter pilots about the adequacy of a landing zone, or timing different squads as they moved across terrain, to become more sensitive to factors such as the nature of the terrain, the effect of weather, and the amount of equipment carried. Thus, the decision requirements let the squad leaders identify their own needs, for their own missions, and let them discover ways to engage in practice and obtain feedback for their judgments and decisions.

You can imagine how this might be useful: when under fire, a marine squad leader might need to evaluate different move options; having a training regiment that emphasises experiencing and measuring movement times across different terrain types would be incredibly useful for mental simulations performed in the field.

Sometimes, it’s just plain difficult to create appropriate training programs for yourself. One way around this is to reflect on your past experiences, and then go to more skilful practitioners to get feedback on the actions that you chose in past events.

When doing so, it’s important to copy the method that’s used in CTA: you should describe the event linearly, as you experienced it, never revealing more information than you had at the time. For instance, if you want to make better decisions around managing people, you might choose to describe an experience where a subordinate reacted negatively to you, and walk through that event as it unfolded. You should not tell the expert what you discovered later, after the events had already transpired. In simple terms: you should put the expert in your shoes.

When doing this, compare:

- What cues it was that you noticed vs what cues they picked up on. (Sometimes this could be expressed in the follow-up questions they ask during your story-telling.)
- What you expected to happen next vs what they expected to happen next.
- What actions you thought to do vs what actions they thought to do,
- And finally: what priorities you had in that situation, vs what priorities they might have instead.

I’ve adapted this idea from Klein’s *cognitive critique* — a tool he uses in his work with Marine Corps squad leaders:

> Cognitive critiques help the squad leaders reflect on what went well and not so well during an exercise, and to use this reflection to increase how much they learned from experience. The critique is a simple exercise, consisting of questions about how the squad leader had estimated the situation (Was it accurate?), uncertainty (Where was it a problem, and how was it handled?), intent and rationale (What was the focus of the effort?), and contingencies (reactions to what-if probes). The checklist could be used after tactical decision games as a way for the squad leaders to compare notes, get feedback, and see how others were perceiving the situation. The checklist, primarily designed to be used after field training exercises, was a way to enrich experiences by reviewing them, much as a chess master goes over a game record.

There are many more training methods to NDM, but I’ve found that nearly all of them are built around several core ideas:

- They expand the learner’s collection of prototypes.
- They increase the experience base necessary for mental simulations.
- They involve getting timely feedback from more skilful practitioners.

NDM research is interesting to me, mostly because their publications often contain descriptions of training programs that work. I read their research mostly as a form of inspiration — you *could* also say that I’m using it to expand the set of possible training programs patterns that I have in my head.

## How Do We Know This Model Is Accurate?

One question that you might reasonably ask is: how do we know that the RPD model is true? I’ve described NDM’s methods to you, and you might have caught on to the fact that none of what I’ve described is experimental. NDM researchers do not use hypothesis testing; there is no concept of falsifiability in their work. Klein himself readily admits that NDM is “nearer to anthropology than psychology.”

Klein gives us a defence of his field in the conclusion of *[Sources of Power](https://www.goodreads.com/book/show/65229.Sources_of_Power?ref=commoncog.com)*:

> The studies I have described are not classical science. We did not calibrate our instruments and present stimuli that were carefully measured to determine their exact visual angle when projected onto the retina. These are the trappings of science but not the central features.  
>   
> What are the criteria for doing a scientific piece of research? Simply, that the data are collected so that others can repeat the study and that the inquiry depends on evidence and data rather than argument. For work such as ours, replication means that others could collect data the way we have and could also analyze and code the results as we have done. It will be difficult for others to conduct interviews using our methods, but we have published articles describing our methods, so others can learn them. After all, I could not replicate experiments in gene splicing without a considerable amount of training, so the fact that training is needed in data collection to study naturalistic decision making does not present a logical problem. In recent years there have been studies replicating our findings, particularly the findings about the RPD model (Mosier, 1991; Pascual & Henderson, 1997; Randel, Pugh, & Reed, 1996).  
>   
> Regarding the nature of our data, one weakness of our work is that most of the studies relied on interviews rather than formal experiments to vary one thing at a time and see its effect. There are sciences that do not manipulate variables, such as geology or astronomy or anthropology. Naturalistic decision making research may be closer to anthropology than psychology. Sometimes we observe decision makers in action, but we rely on introspection in nearly all our studies. We ask people to describe what they are thinking, and we analyze their responses. We do not know if the things they are telling us are true, or maybe just some ideas they are making up. We can repeat the studies or, better yet, other investigators can repeat the studies to see if they get the same results. Nevertheless, no one can confidently believe what the decision makers say.

Here Klein weasels out of it by targeting laboratory methods: he argues that mainstream experimental psychology focuses too much on what may be easily tested in the lab. He ends his defence with:

> (…) Both the laboratory methods and the field studies have to contend with shortcomings in their research programs. People who study naturalistic decision making must worry about their inability to control many of the conditions in their research. People who use well-controlled laboratory paradigms must worry about whether their findings generalize outside the laboratory.  
>   
> One way to evaluate a naturalistic decision-making approach is by its products: the nature of the ideas and models that it generates and the applications it produces. **If NDM and the study of different sources of power turn out to make little difference, then we will lose confidence in the approach more surely than any debate over what is science.***(emphasis added)*

In the two decades since *Sources of Power* was first published, NDM has only become more established as a field of research. More militaries around the world are using NDM methods to extract tacit models of expertise from their best operators, in order to turn that knowledge into training programs; Nasdaq [uses](https://twitter.com/ejames_c/status/1266412975767740416?s=20&ref=commoncog.com) NDM researchers to design user interfaces to augment the expertise of experienced fraud investigators on their staff. Klein himself wrote a [famous paper](https://www.fs.usda.gov/rmrs/sites/default/files/Kahneman2009_ConditionsforIntuitiveExpertise_AFailureToDisagree.pdf?ref=commoncog.com) with Daniel Kahneman on the situations where expert intuition was valid, and where it might not be valid. Most interestingly, Robert Hoffman, an NDM pioneer and one of the co-authors of the *[Cambridge Handbook of Expertise and Expert Performance](https://www.cambridge.org/my/academic/subjects/psychology/cognition/cambridge-handbook-expertise-and-expert-performance-2nd-edition?format=HB&isbn=9781316502617&ref=commoncog.com),* has spent a good part of the last decade [working on an epistemology](https://overcast.fm/+aGDtY2OvU/18:29?ref=commoncog.com) for the entire field.

These are all signs, I think, that NDM captures something true about the world. But be that as it may, I believe that this shouldn’t be of much consideration to you, if you simply want to use this for your career.

I have argued elsewhere that if you are primarily a practitioner, your [epistemology should be different than if you were a scientist](https://commoncog.com/putting-mental-models-to-practice-part-6-a-personal-epistemology-of-practice/). If you are a scientist, you will want to know what is true by the standards of scientific truth — this typically means successful replication over a number of years, culminating in meta-analyses of multiple randomised controlled trials with sufficient statistical power. But if you are a practitioner, what you want to do is to learn what can be useful to you today, in the pursuit of your goals. Your evaluation of an idea should be structured around whether the idea *works for you* — which is a different bar for truth compared to that of the scientist’s.

NDM lies at the highest level in my [hierarchy of practical evidence](https://commoncog.com/the-hierarchy-of-practical-evidence/): I have tried their methods and their ideas in my life, and *nearly every one of them have led to improvements.* I’ve spent this essay talking about how I’ve chased down the implications of RPD and attempted to use them in my pursuit of expertise. But it's likely that I’m only at the beginning of this chase.

There is a lot more in this domain that I haven’t yet discovered. I suspect there’s a lot more that is rewarding still. But don’t take my word for it: try the RPD model as a lens for seeing the expertise of the people around you. What do you want to get good at? Who around you is already good at it? Go find those people, ask them questions, copy them, and learn.

*Read [Part 3: The Three Kinds of Tacit Knowledge](https://commoncog.com/three-kinds-of-tacit-knowledge/).*

Originally published , last updated .

This article is part of the [Expertise Acceleration](https://commoncog.com/expertise) topic cluster. Read [more from this topic here→](https://commoncog.com/expertise)

Previous Post

#### ← Why Tacit Knowledge is More Important Than Deliberate Practice

Next Post

[![Feature image for Tacit Expertise Extraction, Software Engineering Edition](https://commoncog.com/content/images/size/w300/2024/02/tacit_knowledge_software_engineering.jpg)](https://commoncog.com/tacit-expertise-extraction-software-engineer/)[![Feature image for One Definition of Wisdom](https://commoncog.com/content/images/size/w300/2024/02/charlie_munger_wisdom.jpg)](https://commoncog.com/one-definition-of-wisdom/)[![Feature image for Mental Strength in Judo, Mental Strength in Life](https://commoncog.com/content/images/size/w300/2023/05/mental_strength_judo_life-1.jpg)](https://commoncog.com/mental-strength-judo-life/)[![Feature image for Creating New Drills for Deliberate Practice](https://commoncog.com/content/images/size/w300/2023/04/deliberate_practice_create_new_drills.jpg)](https://commoncog.com/creating-drills-deliberate-practice/)

---
title: "The Three Kinds of Tacit Knowledge"
source: "https://commoncog.com/three-kinds-of-tacit-knowledge/"
author:
  - "[[Cedric Chin]]"
published: 2020-06-22
created: 2025-05-30
description: "There are three types of tacit knowledge, all of which 'cannot be captured through words alone'."
tags:
  - "clippings"
---
By

![Feature image for The Three Kinds of Tacit Knowledge](https://commoncog.com/content/images/size/w600/2020/06/driving_tacit_knowledge.jpg)

## Table of Contents

## Fan of great business?

Join 8,000+ sharp investors and operators like yourself, and we'll send you a collection of Commoncog's best articles right away:

*Note: This is Part 3 in a [series about tacit knowledge](https://commoncog.com/the-tacit-knowledge-series/). Read [Part 2](https://commoncog.com/how-to-learn-tacit-knowledge/) here.*

There’s an old quip about taxonomy being the lowest form of science, which has a grain of truth to it — taxonomy isn’t a particularly interesting research activity. But the names we use for things matter if we want to have productive discussions about them.

In the weeks since I started writing my series on tacit knowledge, a number of friends and readers have reached out to me to discuss all the forms of tacit knowledge that they’ve experienced in their lives. These discussions became very confusing very quickly, mostly because the definition of tacit knowledge I used in this series — “knowledge that cannot be captured through words alone” — wasn’t a particularly clear one.

I’ve tried to sidestep the task of defining tacit knowledge in my previous post, where I wrote that I was mostly interested in ‘expert intuition’ as a form of tacit knowledge. I also argued that Commonplace is a practitioner’s blog, and that I was focused on instrumental outcomes. But still the discussions kept coming.

So I’ve given in. This is a short post that summarises the best book on the topic — Harry Collins’s *[Tacit and Explicit Knowledge](https://www.goodreads.com/book/show/8355583-tacit-and-explicit-knowledge?ref=commoncog.com)*, which I read on the recommendation of reader Stephen Samild. (Stephen, if you’re reading this, thank you!) Collins’s book has got a lot going for it — he extends Michael Polanyi’s original definition, examines philosophical, pedagogical, and AI treatments of the subject in the decades since, and then synthesises a set of definitions that I think is quite complete. I wish I’d read it in the beginning instead of Polanyi’s book, but there you go.

This post is short because it isn’t explicitly useful — the goal here is to clarify definitions, after all. But I hope that you will walk away from this piece with a better understanding of ‘tacit knowledge’, and that it’ll result in clearer discussions for you.

## Collins’s Categorisation

According to Collins, there are three kinds of tacit knowledge:

1. **Relational tacit knowledge** — Knowledge that is tacit because of the way people relate to each other. Sometimes people do not make their knowledge explicit because they do not know how to do so, other times, they do not make their knowledge explicit because they do not want to do so.
2. **Somatic tacit knowledge** — Knowledge that is tacit because it has to do with the human body. Think of executing a tennis backhand, playing a guitar, or riding a bicycle. This knowledge is tacit because it has to do with the embodied nature of the skill involved.
3. **Collective tacit knowledge** — Knowledge that is tacit because it is embedded in our social environment, and we do not know how to make it explicit in a way that doesn’t involve socialising.

Let’s examine each type in turn.

## Relational Tacit Knowledge

Relational tacit knowledge is the tacit knowledge we’ve been talking about in this series. It is ‘relational’ in the sense that it has to do with the relationships between human beings; it is knowledge that is kept tacit because:

1. People do not want it to be widespread. (e.g. it is transmitted only through selective apprenticeship, or kept within secret societies).
2. It is incredibly time consuming or costly to explicate. This is essentially the nature of expertise we’ve been talking about in [Parts 1](https://commoncog.com/tacit-knowledge-is-a-real-thing/) and [2](https://commoncog.com/how-to-learn-tacit-knowledge/) of this series.
3. It is held by people who are not capable of explicating it, perhaps because they are bad at teaching. (Woe to the skill whose last practitioner happens to be a horrible teacher!)
4. Or it could be that practitioners themselves are not aware that certain parts of their knowledge are absolutely critical to their success, and so are unable to articulate what it is that they *really* do.

In principle, *all* relational tacit knowledge may be made explicit, with enough time and effort. The key clause in that sentence is ‘ *enough time and effort* ’ — for instance, it often takes Naturalistic Decision Making researchers weeks or months before they successfully explicate the nature of the expertise they are studying.

Collins himself has written a fair amount about scientists who were attempting to build a new kind of laser — the transversely excited atmospheric pressure carbon dioxide laser, or TEA laser. He notes, of that pursuit:

> My study (…) showed that the scientists failed if they used only the information published in scientific papers. These papers included those which supplied details as intricate as the cross-section and machining instructions for the electrodes and even the manufacturers’ part numbers for bought-in items. It showed, however, that only those who spent some time socially interacting with others who had already built a working model could succeed.

It seems that the act of building a working TEA laser involves all sorts of tiny details that the original creators did not think to explicate. And even if they did make an effort to do so, Collins is not sure that the results would have been any different — some of these things were tiny tweaks that everyone in the lab group understood, but nobody had thought of as important knowledge.

Here's Collins again, on a different group of scientists:

> My 2001 study of scientists trying to measure the quality factor, or Q, of sapphire, backed up the earlier work by showing that measurements of the quality factor of small sapphire crystals were so hard that only one group of scientists in the world were able to achieve them until a member of the successful Russian group spent considerable time in the laboratory of a second group, in Glasgow, who finally managed it after a week or so of interaction.

Collins appears to enjoy using the laser example and the sapphire example because the scientists were working from detailed specifications published in peer-reviewed journals … and they were working in physics and material science respectively — both of them ‘hard sciences’! Collins’s point: if it is difficult to explicate knowledge in the ‘hard sciences’, then we should expect no less in other domains of human knowledge.

I look at stories like this and think that a certain amount of relational tacit knowledge will remain tacit forever, because it is simply too troublesome to make it explicit. Why bother writing directions in minute detail, after all, when you can have groups of scientists talking to each other and learning by osmosis instead?

## Somatic Tacit Knowledge

Somatic tacit knowledge is Collins’s fancy term for embodied knowledge — that is, things that we learn to do with our bodies. This knowledge is tacit since you can really only learn to ride a bike or throw a curveball with actual *physical* practice.

*[The Inner Game of Tennis](https://www.goodreads.com/book/show/905.The_Inner_Game_of_Tennis?ref=commoncog.com)* deals with the nature of somatic tacit knowledge quite well, I think; the author, tennis pro Timothy Gallwey, wrote this beautiful passage on the nature of technique that still haunts me:

> Tennis was brought to America from Europe in the late 1800s. There were no professional tennis teachers to teach technique. The best were players who experienced certain feelings in their swings and tried to communicate those feelings to others. In the effort to understand how to use technical knowledge or theory, I believe that it is most important to recognize that, fundamentally, experience precedes technical knowledge. We may read books or articles that present technical instructions before we have ever lifted a racket, but where did these instructions come from? At some point did they not originate in someone’s experience? Either by accident or by intention someone hit a ball in a certain way and it felt good and it worked. Through experimentation, refinements were made and finally settled into a repeatable stroke.  
>   
> Perhaps in the interest of being able to repeat that way of hitting the ball again or to pass it on to another, the person attempts to describe that stroke in language. But words can only represent actions, ideas and experiences. **Language is not the action, and at best can only hint at the subtlety and complexity contained in the stroke.** Although the instruction thus conceived can now be stored in the part of the mind that remembers language, **it must be acknowledged that remembering the instruction is not the same as remembering the stroke itself.***(emphasis mine)*

In the same chapter, Gallwey relays the story of teaching a student without explicit instruction: “I was going to skip entirely my usual explanations to beginning players about the proper grip, stroke and footwork for the basic forehand. Instead, I was going to hit ten forehands myself, and I wanted him to watch carefully, not thinking about what I was doing, but simply trying to grasp a visual image of the forehand. He was to repeat the image in his mind several times and then just let his body imitate.”

It worked. Gallwey concluded: “I was beginning to learn what all good pros and students of tennis must learn: that images are better than words, showing better than telling, too much instruction worse than none, and that trying often produces negative results.”

When teaching somatic tacit knowledge, words don’t really help as much as imitation does.

## Collective Tacit Knowledge

Collective tacit knowledge was what I was alluding to, last week, when I [wrote](https://commoncog.com/how-to-learn-tacit-knowledge/):

> In 1988, Taiichi Ohno published The Toyota Production System, which laid out the ideas behind lean manufacturing for the first time. He did so after resisting codification of those principles for many, many years; Ohno feared that the publication of TPS would destroy Toyota’s competitive advantage. But it turned out that the publication of Ohno’s book by itself did not lead to widespread adoption amongst Toyota’s competitors. More bizarrely, competitors who poached TPS experts away from Toyota were not able to replicate Toyota’s system in their company. It appeared that TPS required an organisational culture to develop alongside the implementation of the system itself. This phenomenon sparked off an entire subfield of research into organisational tacit knowledge.

We learn certain things as a result of socialisation. For instance, we may learn to drive a car domestically, according to the traffic laws of our country, but when we travel abroad — to Vietnam, say, where the traffic behaviours are crazy — we adapt quickly to the norms of the new place. We pick it up on the go; this requires little explicit instruction.

Collins also talks about this in the context of bike riding; he points out that bicycling is somatic tacit knowledge, but when we zoom out, we realise that a good bit of it is collective tacit knowledge as well:

> Negotiating traffic is a problem that is different in kind to balancing a bike, because it includes understanding social conventions of traffic management and personal interaction. For example, it involves knowing how to make eye contact with drivers at busy junctions in just the way necessary to assure a safe passage and not to invite an unwanted response. And it involves understanding how differently these conventions will be executed in different locations. For example, bike riding in Amsterdam is a different matter than bike riding in London, or Rome, or New York, or Delhi, or Beijing. (For example, in China, bicycles are ridden at night, without lights, in ways that would be considered absolutely suicidal in the West.) Then again, even in one country there are different settings for riding — riding in the country and riding in the town — mountain biking, riding to school or work, racing, or riding as a display of skill.

Similarly, while the Toyota Production System could not be explicated and transferred between organisations easily, a new employee inducted into Toyota would rapidly internalise what is necessary for the system’s success. We use words like ‘culture’ and ‘organisational behaviour’ to capture this ineffable property; Collins argues that such knowledge is *the* most tacit of the types of tacit knowledge that exists. He argues that even if we train machines to learn to drive, we cannot expect machines to adapt to driving patterns in the way that humans can when they travel abroad. The human would immediately evaluate the social context they are embedded within; the machine would not.

But my favourite example of collective tacit knowledge is this one, from Star Trek:

> In one episode of “Star Trek: The Next Generation,” Dr. Crusher shows Lieutenant Commander Data—a clever android—the steps of a dance. She shows them only once. Most of us would need to practice and practice under Dr. Crusher’s guidance before we could repeat the steps with her skill; for us it would be a case of acquiring somatic tacit knowledge. The scene is amusing, because this is what we are led to expect, yet Data is immediately able to repeat the steps at speed and without fault; he has the kind of quick brain that could also master bike balancing from Polanyi’s instructions. If Data has a somatic limit, it is less restrictive than that of any human.  
>   
> Dr. Crusher, impressed by the astonishing facility exhibited by Lieutenant Commander Data, then tells him that he must simply repeat what he has done with some additional improvisation if he wants to dance with verve on the ballroom floor. This is where “Star Trek” goes wrong, because it shows Data managing improvisation as flawlessly as he had managed the initial steps. But improvisation is a skill requiring the kind of tacit knowledge that can only be acquired through social embedding in society. Social sensibility is needed to know that one innovative dance step counts as an improvisation while another counts as foolish, dangerous, or ugly, and the difference may be a matter of changing fashions, your dancing partner, and location. There is no reason to believe that Data has this kind of social sensibility.

Collins insist that this kind of knowledge is impossible to explicate. Whether it is really impossible — and therefore out of reach for an AI! — is beyond the scope of this article. Collins thinks that it is; I do not. Nevertheless, it is useful to know that such knowledge exists; we take it for granted because it comes so naturally to us.

## Wrapping Up

Collins makes two more observations that I thought were pretty interesting.

1. In principle, relational tacit knowledge is explicable, even if challenging; somatic tacit knowledge less so, and collective tacit knowledge ‘impossible’. But:
2. When it comes to human behaviour, the ease with which we pick up the three forms of tacit knowledge is reversed! It is easiest for us to pick up collective tacit knowledge — we begin socialisation from birth; we don’t even notice when we absorb social context from our surroundings. In comparison, it is slightly harder to pick up somatic tacit knowledge, and it is most difficult to pick up relational tacit knowledge, since this hangs on the nature of expertise, and it the depends on the ways with which we relate to each other.

In practice, many things that we do involve all three types of tacit knowledge — car driving, for instance, involves relational tacit knowledge, since it is a skill that must be acquired from others; it involves somatic tacit knowledge because it is something that we do with our bodies, and finally it involves collective tacit knowledge, since we pick up on norms whenever we are driving, wherever we are in the world.

Read [Part 4 — How to Use YouTube to Learn Tacit Knowledge](https://commoncog.com/youtube-learn-tacit-knowledge/).

Originally published , last updated .

This article is part of the [Expertise Acceleration](https://commoncog.com/expertise) topic cluster. Read [more from this topic here→](https://commoncog.com/expertise)

Previous Post

#### ← Copying Better: How To Acquire The Tacit Knowledge of Experts

Next Post

[![Feature image for What Process Improvement in Education Looks Like](https://commoncog.com/content/images/size/w300/2024/09/rigorous_process_improvement_education.jpg)](https://commoncog.com/what-process-improvement-in-education/)[![Feature image for Business Ecosystem Change Takes Time](https://commoncog.com/content/images/size/w300/2024/06/business_ecosystem_change_takes_time.jpg)](https://commoncog.com/business-ecosystem-change/)[![Feature image for Follow Your Nose](https://commoncog.com/content/images/size/w300/2021/03/follow_your_nose.jpeg)](https://commoncog.com/follow-your-nose/)[![Feature image for Create Your Own Rituals](https://commoncog.com/content/images/size/w300/2021/02/create_your_own_rituals.jpeg)](https://commoncog.com/create-your-own-rituals/)

---
title: "How to Use YouTube to Learn Tacit Knowledge"
source: "https://commoncog.com/youtube-learn-tacit-knowledge/"
author:
  - "[[Cedric Chin]]"
published: 2020-06-30
created: 2025-05-30
description: "YouTube is the biggest thing to have happened to tacit skill acquisition in the past couple of decades. Here's how to use it."
tags:
  - "clippings"
---
By

![Feature image for How to Use YouTube to Learn Tacit Knowledge](https://commoncog.com/content/images/size/w600/2020/06/youtube_tacit_knowledge.jpg)

## Table of Contents

## Fan of great business?

Join 8,000+ sharp investors and operators like yourself, and we'll send you a collection of Commoncog's best articles right away:

*This is Part 4 in a [series on tacit knowledge](https://commoncog.com/the-tacit-knowledge-series/). Read [Part 3](https://commoncog.com/three-kinds-of-tacit-knowledge/) here.*

We can’t really talk about tacit knowledge today without talking about YouTube.

A couple months ago, Samo Burja wrote an essay on Medium titled [The Youtube Revolution in Knowledge Transfer](https://medium.com/@samo.burja/the-youtube-revolution-in-knowledge-transfer-cb701f82096a?ref=commoncog.com), a piece that I loved and linked to in [issue 50](https://us17.campaign-archive.com/?u=843adfbaa3230c81aa0738b53&id=0ced4f1a2c&ref=commoncog.com) of the Commonplace newsletter. Burja observed the following things:

- First, YouTube is now a *thing*.
- Second, quality digital cameras have become cheap enough for mass use. This in turn means that everyone can start producing YouTube-worthy content, and have those videos be discoverable, because — hey! — YouTube is now a thing.
- Three, this *also* means that people with expertise in *thing X* can now film themselves doing *thing X*, and clued-in practitioners can watch these experts do *thing X* and become *really good themselves.*
- Burja then posits that we’ll see an explosion of tacit skill transfer over the next decade, correlated with the rise of YouTube use, because certain kinds of human expertise may only really be taught through emulation, not through instruction alone.

Burja’s whole essay is great, and I urge you to read it in [full](https://medium.com/@samo.burja/the-youtube-revolution-in-knowledge-transfer-cb701f82096a?ref=commoncog.com).

But there’s a twist here. I linked to Burja’s piece in Issue #50 of my newsletter, which went out on the 24th of September 2019. Burja’s piece was published on the 18th of September. And I’m writing this on the 29th of June the next year. So it’s basically taken me more than nine months before I realised that Burja’s observations may be *generalisable* — that is, once you realise that YouTube is a great place to learn tacit things, this fact applies to skills *far beyond* the set of skills Burja talks about in his essay.

In this piece, we’re going to walk through a number of ways you may use YouTube for tacit knowledge acquisition, on a domain-by-domain basis. I’m afraid the anecdotes here are necessarily domain-specific, but the purpose of this piece is to give you certain patterns that you may adapt to whatever skillset you want to acquire in whatever domain you’re interested in.

## YouTube use in Judo

Let’s start with Judo.

I'm biased here because I played Judo competitively as a teenager. But Judo is interesting for our purposes because it isn’t a particularly popular sport. In Malaysia, where I’m from, the number of people who play Judo can probably fit into a small football stadium. And the active players — the ones who do Judo regularly — can probably fit into half that. Contrast this population with a sport like badminton, say, or football, or tennis.

What this means is that it is difficult to get good as a judoka in most countries around the world. This fact isn’t limited to Malaysia; in the US, for instance, pretty much every Olympic Judo champion comes from a tiny family-run club called Pedro’s Judo Centre in Wakefield, Massachusetts. Judo isn’t particularly popular outside the club, either. If you find yourself anywhere else in the country: tough luck.

During the worst of the COVID-19 lockdowns earlier this year, Malaysian Judo coach Oon Yeoh began interviewing international Judo players from all over the world, in a series he named *Judo In The Time of COVID-19*. He [wrote](https://kljudo.com/judo-concepts-lesson-22-how-to-analyze-a-video-clip/?ref=commoncog.com), of that experience:

> One thing I noticed after interviewing many judo players is that those from very strong judo nations tend not to watch so many judo videos whereas those who come from countries where judo is not so strong, tend to watch a lot of judo videos.  
>   
> The reason is obvious. If you're from a strong judo nation, there's a lot of institutional support and good coaching, as well as sufficient training partners, training opportunities and competitions. All these combined give you everything you need to develop your skills in judo. **But if you live in a place where there's very little judo, you're pretty much left to your own devices. In such circumstances, video is the best resource for you to learn things.** *(emphasis mine)*

Oon first learnt Judo in the US, and was exposed to video-based training from the beginning of his career. (To my knowledge, he remains — to this day — the only Malaysian to have fought in the World Championships). When he says ‘video is the best resource for you to learn things’, we have to be very clear what he means by this.

If you go to YouTube right now and enter a Judo technique, for instance, ‘uchimata’ — you’ll get a results page like so:

![](https://commoncog.com/content/images/2020/06/Screenshot-2020-06-29-at-8.17.46-PM-1.png)

Notice how world champion Maruyama Joshiro is using his leg to lift his opponent's.

The top result is an instructional by US Judo coach Shintaro Higashi:

![](https://www.youtube.com/watch?v=qnMoNfZ2WiU)

Higashi’s video on uchimata is particularly good, and instructionals are probably what you’re thinking of when you hear ‘YouTube’ and ‘learning’. But instructionals *aren’t* the only thing that Oon is talking about.

What Oon is more interested in is *analysis of competition footage*. I asked him to explain how he used video for this purpose, and he [did](https://kljudo.com/judo-concepts-lesson-22-how-to-analyze-a-video-clip/?ref=commoncog.com):

> Let's say you want to learn how a particular player does a particular technique. Here are some best practices:  
>   
> a) Don't just watch just one or two examples of that player doing that technique. Watch multiple examples and try to identify trends or things that that player does in setting up the throw. Take for example, Takato's unique style of ouchi-gari. If you watch enough of them, you'll notice he only does this technique under very specific circumstances (against left-handers who are trying to push his head down). If you notice that, try to understand the reason why.  
>   
> b) Pay particular attention to grips. How does he grip whenever he wants to do this technique. You will notice a familiar pattern to the gripping. Again, try to understand why.  
>   
> c) There will be times when the technique fails. Watch those as well and try to understand what went wrong. What was different about this attempt that caused it to fail? Doing this will allow you to isolate the key success factors.  
>   
> d) Try to look out for variations. Sometimes those differences are very subtle but they are significant. Understanding the variations (when they are used and why the variation are necessary) will give you a better grasp of the technique.  
>   
> e) It is usually helpful to watch the technique in slow motion. If there is no slow motion replay of the clip available, you may have to download the clip and slow-mo it yourself using a simple video editing program. When I was a student there was no digital videos yet, only VHS cassettes, and I had to use two VCR machines to make slow motion loops of Koga's throws just so I could study them properly!

In a separate interview with US Olympic silver medallist Travis Stevens, Oon asks about video, and Stevens [answers](http://kljudotraining.blogspot.com/2020/04/judo-in-time-of-covid-19-travis-stevens.html?ref=commoncog.com):

> There is a lot the cadets can do at home to prepare them for future success. Like **learning how to break down a fight when watching videos, taking notes on training sessions** *(emphasis mine)*, journaling, etc. Winning at judo isn’t just about hard work. There’s a lot more that goes into it than that, like analysis, mental training and so on.

Why study competition footage, and not just instructionals? The answer falls out of everything we’ve explored in our previous posts on tacit knowledge. Expertise is difficult to explicate, and embodied knowledge can only loosely be captured through language. There usually isn't much of an overlap between the greatest practitioners and the greatest teachers in any given domain.

Plus: ask any serious Judo player, and they will tell you that there is *also* a gap between what is taught in the dojo, and what is used in competition. The charitable explanation here is that the best players can’t explain what they’re doing, because their somatic responses are tied to cues in implicit memory, which makes it impossible for them to notice. The slightly less charitable explanation is that there is a [metagame](https://commoncog.com/to-get-good-go-after-the-metagame/) that’s going on at the highest levels of Judo, and the shift in the current meta is kept opaque and available only to those who know how to watch out for it. The completely uncharitable explanation is that Judo is a sport with bad pedagogy, and most people don’t know how to teach it.

![](https://commoncog.com/content/images/2020/06/bdf1b3347d045db83a9c988fb87c8629.jpg)

Notice how world champion Maruyama Joshiro is using his leg to lift his opponent's.

![](https://commoncog.com/content/images/2020/06/Screenshot-2020-06-30-at-5.35.29-PM.png)

And compare that to Higashi's instructional, which uses the hip to lift.

(I’m perhaps being a little extreme in that last assessment; but — well, let’s talk about uchimata, shall we? There’s a [particularly good video](https://www.youtube.com/watch?v=HZJcx2ppyB4&ref=commoncog.com) by YouTube user Judo Mat Lab that points out that the throw taught by Higashi, in its form above, is rarely seen in competition. Many top level athletes do something far simpler (they lift the opponent's leg and tilt, not lift their opponent with their hip) — in fact, the tilting form is what is most commonly seen in [MMA](https://www.youtube.com/watch?v=Kd10gC4P310&ref=commoncog.com) and in [BJJ](https://www.youtube.com/watch?v=Xb-HW-6isyI&ref=commoncog.com) as well. The funny thing is that when these judokas are asked to demonstrate uchimata, they execute *the classical, lifting version first.* This is an example of a piece of tacit knowledge that was successfully explicated into a wonderful, comprehensive explanation — but it took a talented YouTuber to do it, and in 2015, 133 years after Judo was first created.)

![](https://www.youtube.com/watch?v=HZJcx2ppyB4)

So the solution here is to do what Oon figured out all those years ago: *don’t* wait for someone to come along and explain what’s going on. Don’t rely on instructionals alone. *Look at actual experts doing the actual thing.* In the context of Judo, this means developing the meta-skill of watching competition video yourself, in order to break a competitor’s technique down into its constituent parts. You try to figure out why it’s so effective; later, in the dojo, you attempt to replicate it by copying.

In fact, as we’ll soon see: this principle is generalisable. When you want to learn a skill on YouTube, you should spend a little bit of time watching instructionals, but then spend the rest of it *watching experts doing the actual thing*. The former is by definition explicit; the latter allows you to capture what is tacit.

## Programming

Let’s move on to programming, which is probably more relevant to the topic of careers.

When I started programming in university in 2009, there was already a fair amount of programming content on the web. This period coincided with the rise of the [MOOC](https://en.wikipedia.org/wiki/Massive_open_online_course?ref=commoncog.com), and so the number of lectures about programming that were freely available went *way* up in the span of a few years.

![](https://commoncog.com/content/images/2020/06/Screenshot-2020-06-30-at-5.58.58-PM.png)

Professor Abelson's famous introduction to the field: 'computer science is not a science, and it has little to do with computers'.

I really appreciated this trend. I remember thinking that anyone could watch the legendary Harold Abelson [explain the basics of computer programming](https://www.youtube.com/watch?v=-J_xL4IGhJA&ref=commoncog.com) at MIT; they could catch Sal Khan teaching math on [Khan Academy,](https://www.khanacademy.org/?ref=commoncog.com) and of course they were spoilt for choice by the programming conference talks that conf organisers were beginning to upload, en masse, to YouTube. I was a poor student at the time; I remember thinking that it was absolutely great that I could get all of this in piped to me in my dorm room in Singapore; I felt like one of the luckiest people alive.

But talks and lectures are basically instructionals. They are necessarily explicit. Even when speakers attempt to express some deep, tacit principle, the information they can communicate is limited by the nature of human language itself. I remember watching [Rich Hickey](https://en.wikipedia.org/wiki/Rich_Hickey?ref=commoncog.com) ’s famous talk [*Simple Made Easy*](https://www.infoq.com/presentations/Simple-Made-Easy/?ref=commoncog.com) and thinking to myself: “wow, this seems profound, this seems really important” and having zero way to use it in my daily practice of programming.

![](https://commoncog.com/content/images/2020/06/Screenshot-2020-06-30-at-5.33.11-PM.png)

At this point in the series, we know why it is difficult to apply Hickey’s ideas in *Simple Made Easy* — more importantly, we have the language to talk about it. We know that Hickey’s knowledge is [relational tacit knowledge](https://commoncog.com/three-kinds-of-tacit-knowledge/#relational-tacit-knowledge); we also know that Hickey’s expertise is made up of [prototypes in his implicit memory](https://commoncog.com/how-to-learn-tacit-knowledge/#the-recognition-primed-decision-making-model), and that the only way to get at it is to build up a library of equivalent prototypes in our heads. And that's not all: we must also acquire the solutions that Hickey matches to those prototypes, when he’s building these large, complicated systems. In simpler terms, the only way to truly internalise ‘ *Simple Made Easy* ’ is to program under Hickey’s guidance for a bit.

In fact, there’s a good rule of thumb here that I think is worth keeping in mind: if something can fit into the length of a typical programming conference talk, the odds are good that the aforementioned thing is easily explicable. You get the sense that Hickey’s talk could *not* fit into the length of a conference keynote; if you gave him more time, he could go on.

You get this feeling because Hickey is trying to communicate something deeply tacit: a principle that he has imbued into all the projects he’s worked on from [Clojure](https://clojure.org/?ref=commoncog.com) to [Datomic](https://www.datomic.com/?ref=commoncog.com). And indeed, the more common programming conf talks tend not be like Hickey’s; they tend to be about easily explicable topics like ‘how to use cool new feature in programming language X’ or ‘how to do technique Z in framework Y’; you’re not likely to get higher level things like “how do I structure a program to thread the thin line between too much abstraction and too little?” or “how do I evaluate the risks of my client being an idiot, which means we’ll have to redesign a whole subsystem a month from now?” *That* sort of thing is tacit, you pick it up while on the job, working alongside more experienced people.

So if conference talks and lectures are the programming equivalent of Judo instructionals, what is the equivalent of analysing competition videos? The answer, of course, is coding livestreams.

![](https://www.youtube.com/watch?v=FYTZkE5BZ-0)

I think coding livestreams really only became a thing within the last five years or so — perhaps due to the growth and influence of gaming streamers on Twitch. It’s sort of crept up on the programming world — I certainly wasn’t aware of its usefulness when I read Burja’s piece last year. But I think we are all the better for it now that it’s here.

Watching someone program is a *very* different experience when compared to watching a conference keynote, or even an instructional. It is the closest experience you can get to watching a senior developer code next to you, or having them pair program alongside you. In fact, in some ways, livestreams are more useful because you can go back and rewatch certain segments; once those streams are uploaded to YouTube, they also become more scalable than a one-on-one pair programming session.

I think it’s pretty obvious to programmers that there are certain things you can only pick up when you’re watching other programmers in action that you cannot when they’re giving talks about their work. The most obvious example of this is perhaps their debugging ability — in the video I linked to above, you can watch when Tsoding encounters little things that don’t go according to plan, and then listen to him reason his way out of it. This tells you something about the model of the problem domain he has inside his head.

Personally, I find it *really* interesting to watch more experienced programmers structure a program over time. At the beginning of a project, it’s usually not clear what the shape of the problem is, and so the first program structure that a programmer picks tends not to be the best. What I tend to watch out for is the exact point at which a programmer decides they have to restructure a certain subcomponent — I ask things like: *how* did they know to pick a new structure? If it was a ‘feeling’, what feeling was it that they’re picking up on? What cues or ‘smells’ did they notice? Would I have done something differently? And could I attempt to copy their solution instead?

## Music

Singer songwriter John Mayer streamed a guitar lesson on Instagram Live on May 7th, earlier this year. At 8:35 in the video, he segued to a lesson about the nature of skill progression in guitar that I thought was pretty profound:

![](https://www.youtube.com/watch?v=MXxLR8s4xRQ)

I quote:

> “The idea, that you already know how to play the guitar — if you learn the pentatonic, if you’ve played along to blues records and rock records, and you’ve soloed for awhile, you have all the moving parts.  
>   
> What we’re all learning how to do is … to do *THAT, THEN*. So you know all the letters in the alphabet, they’re really easy to learn on the guitar — that’s the minute to learn. And the lifetime to master part is: how do you implement it and how you think?  
>   
> At a certain point, all learning guitar from records is, ‘oh you can do that then!’ So you’re not learning how to *play* that type of thing, you’re learning you *can* play that thing at that point in time.”

If you shove what Mayer is saying into the models we’ve explored over the course of this series, you should immediately grok what Mayer is saying. He’s saying that guitar playing is roughly two levels of progression: the first is to learn the basics of guitar playing — the pentatonic scale, and all the embodied knowledge necessary to execute various types of guitar picks. But the level above is essentially the process of expanding the set of prototypes in your head. You want to know what’s good in the genres you’re interested in, you want to know what’s been tried before, and you want to know — as Mayer puts it — ‘that you can do *that then* ’.

So what better way to acquire this knowledge than to listen to lots and lots of music? And what better way to learn technique than to watch lots of guitar-playing videos on the web?

## Wrapping Up

I think there are two trends that make up the ‘YouTube revolution in knowledge transfer’, as Burja puts it. We’ve seen both of them in every domain we’ve explored, above.

1. The first thing is an increased level of access to quality instructionals. Higashi’s video, Abelson’s lecture, and Mayer’s Instagram story are levels above what you can get if you attended the average meetup in your area.
2. The second thing is that YouTube greatly expands the set of people who can watch expert practitioners doing their thing. As we’ve seen from Oon’s example: when you have the opportunity, bias towards *watching experts doing the actual thing.* Instructional videos are often limited by the quality of the instructors you have in your field.

In fact, I haven’t tried this yet, but I want to know if Oon Yeoh’s principles for analysing Judo video generalises to every other domain of tacit knowledge. If we take his list and remove the Judo-specific references, we get:

1. Don’t just watch one or two examples of the expert doing a technique. Gather multiple examples and try to identify trends or commonalities in the usage of their technique. When you notice such commonalities, try and understand why those commonalities exist.
2. Pay particular attention to context. What happens before or after application of the technique?
3. There will be times when the technique fails. Watch those as well and try to understand what went wrong.
4. Try and look out for variations. These variations represent changes in the action script associated with the prototype in the expert’s head. Sometimes these differences are very subtle, but they are significant. Understanding the variations will give you a better grasp of the technique.
5. It is usually helpful to watch the technique in slow motion. Download the clip and slo-mo it yourself using a free video editing program.

To be fair, Judo is an incredibly technical sport, and it might be overkill to use techniques adapted for that domain. But even if you don’t use Oon’s sophisticated techniques, you might still be pleasantly surprised by the benefits of watching experts doing actual things anyway. Sometimes your brain just picks up on tacit knowledge the natural way: that is, by osmosis.

As Patrick McKenzie observed:

And perhaps this shouldn’t surprise us — after all, blind copying is how we were originally made to learn.

Read [Part 5 — An Easier Method to Extract Tacit Knowledge](https://commoncog.com/an-easier-method-for-extracting-tacit-knowledge/)

Originally published , last updated .

This article is part of the [Expertise Acceleration](https://commoncog.com/expertise) topic cluster. Read [more from this topic here→](https://commoncog.com/expertise)

Previous Post

#### ← The Three Kinds of Tacit Knowledge

Next Post

[![Feature image for Tacit Expertise Extraction, Software Engineering Edition](https://commoncog.com/content/images/size/w300/2024/02/tacit_knowledge_software_engineering.jpg)](https://commoncog.com/tacit-expertise-extraction-software-engineer/)[![Feature image for One Definition of Wisdom](https://commoncog.com/content/images/size/w300/2024/02/charlie_munger_wisdom.jpg)](https://commoncog.com/one-definition-of-wisdom/)[![Feature image for Mental Strength in Judo, Mental Strength in Life](https://commoncog.com/content/images/size/w300/2023/05/mental_strength_judo_life-1.jpg)](https://commoncog.com/mental-strength-judo-life/)[![Feature image for Creating New Drills for Deliberate Practice](https://commoncog.com/content/images/size/w300/2023/04/deliberate_practice_create_new_drills.jpg)](https://commoncog.com/creating-drills-deliberate-practice/)

---
title: "An Easier Method for Extracting Tacit Knowledge"
source: "https://commoncog.com/an-easier-method-for-extracting-tacit-knowledge/"
author:
  - "[[Cedric Chin]]"
published: 2021-04-27
created: 2025-05-30
description: "A look at Laura Militello and Robert Hutton's Applied Cognitive Task Analysis, a simplified method for getting at the tacit expertise of others."
tags:
  - "clippings"
---
![Feature image for An Easier Method for Extracting Tacit Knowledge](https://commoncog.com/content/images/size/w600/2021/04/applied_cognitive_task_analysis.jpeg)

*This is Part 5 in a* [*series on tacit knowledge*](https://commoncog.com/the-tacit-knowledge-series/)*. Read* [*Part 4*](https://commoncog.com/youtube-learn-tacit-knowledge/) *here.*

Let’s say that you’re a junior employee at a company, and you find a senior person with a set of skills that you’d like to learn. You ask them how they do it, and they give you an incomprehensible answer — something like “oh, I just know what to do”, or “oh, I just do what feels right”. How can you learn their skills?

In [Part 2](https://commoncog.com/how-to-learn-tacit-knowledge/) of my tacit knowledge series, I described one method: you take something called the [recognition-primed decision making (RPD) model](https://commoncog.com/how-to-learn-tacit-knowledge/#the-recognition-primed-decision-making-model) and then you use that as a map to get at the tacit expertise of skilled practitioners around you. The RPD model describes how intuitive expertise works; my argument was that once you understand the shape of intuition, you can get at some of the things that every expert must do when they’re operating in the real world, *even when they can’t explain what they do.*

The main problem with that approach is that it's *really hard*. While RPD describes what expert intuition is, it doesn’t really help you extract that intuition for yourself. The researchers who came up with the RPD model developed a method for doing that, which they named the ‘Critical Decision Method’ (CDM), but the technique is incredibly difficult to learn and extremely time consuming to execute. All of which is well and good when you’re a researcher, and your clients hire you to spend months designing training programs for them; less so if you’re a practitioner, and you simply want to get at the skills that live in other peoples’s heads.

Which brings us to this post.

In 1998, Laura Militello and Robert Hutton published a landmark paper titled [*Applied Cognitive Task Analysis (ACTA): A Practitioner’s Toolkit for Understanding Cognitive Task Demands*](https://www.researchgate.net/publication/13466276_Applied_Cognitive_Task_Analysis_ACTA_A_Practitioner%27s_Toolkit_for_Understanding_Cognitive_Task_Demands?ref=commoncog.com). The paper’s contribution was a set of four techniques that are much easier to use than CDM (and other related methodologies), and yet powerful enough for instructional designers or interface designers to extract tacit mental models of expertise from the heads of subject matter experts for their use. The work was funded by the Navy Personnel Research and Development Centre and lasted two years; the authors tested the usability of these new methods by teaching them to two groups of graduate students, before letting them loose to interview two sets of experts: fireground commanders on one end, and naval electronic warfare specialists on the other. These graduate students were then asked to modify existing training programs for both domains, after which those modifications were evaluated by expert instructors in the field.

Militello and Hutton’s reasoning went something like this: if ACTA was easy enough for random graduate students to use, *and* if it resulted in significant improvements to the training programs in both domains, then the methods would be *even more effective* when put in the hands of domain-specific course creators and interface designers. The authors knew that ACTA wasn’t as accurate or as powerful as earlier research methods (like CDM), but they thought that the simplicity of their technique was a worthwhile tradeoff.

In the years since they first published the paper, they’ve largely been proven right. Which probably means that it is worth it for us to take a look at their methods, in order to adapt them to our goals.

## The Four Techniques of Applied Cognitive Task Analysis

There are four techniques in ACTA, and all of them are pretty straightforward to put to practice:

1. You start by creating a **task diagram**. A task diagram gives you a broad overview of the task in question and identifies the difficult cognitive elements. You'll want to do this at the beginning, because you'll want to know which parts of the task are worth focusing on.
2. You do a **knowledge audit**. A knowledge audit is an interview that identifies all the ways in which expertise is used in a domain, and provides examples based on actual experience.
3. You do a **simulation interview**. The simulation interview allows you to better understand an expert’s cognitive processes within the context of an single incident (e.g. a firefighter arrives at the scene of a fire; a programmer is handed an initial specification). This allows you to extract cognitive processes that are difficult to get at using a knowledge audit, such as situational assessment, and how such changing events impacts subsequent courses of action.
4. You create a **cognitive demands table**. After conducting ACTA interviews with *multiple* experts, you create something called a ‘cognitive demands table’ which synthesises all that you’ve uncovered in the previous three steps. This becomes the primary output of the ACTA process, and the main artefact you’ll use when you apply your findings to course design or to systems design.

We’ll go through each technique in turn.

## Task Diagram

The goal when creating a task diagram is to set up the knowledge audit and the simulation interview. You want a big-picture overview of the most cognitively demanding parts of the task, so that you may focus the majority of your time on those parts.

You start out by asking the expert to decompose the task into steps or subtasks. You ask: “Think about what you do when you *(task of interest)*. Can you break this task down into less than six, but more than three steps?” The goal is to get the expert to walk through the task in his or her mind, verbalising the major steps. The question purposely limits the expert to between three and six steps to ensure that they don’t waste time diving into minute detail; you really want to prevent yourself from digging into their mental models at this stage.

After the expert settles on a list of steps, ask: “Of the steps you have just identified, which require difficult cognitive skills? By cognitive skills I mean: judgments, assessments, and problem solving-thinking skills.”

Circle those steps; you’ll be focusing on them during the next two techniques.

The output of the task diagram would look something like this:

![](https://commoncog.com/content/images/2021/04/Task-Diagram.png)

Sample Knowledge Audit Table

In this case, you’ll expect to dive deep into ‘Initial Assessment’, ‘Primary Search and Rescue’, and ‘Secondary Search and Rescue’, and ignore the ‘Critique/Debrief’ step.

## Knowledge Audit

The knowledge audit identifies ways in which expertise has been used in a domain, and surfaces examples based on the expert's real world experiences. The goal here is to capture the most important aspects of expertise.

You start out with a list of basic probes. These probes are drawn from the knowledge categories that most commonly characterise expertise. After a handful of interviews, it should become clear to you which probes produce the most information for that specific subtask; you may then reduce the time you spend on less useful questions.

Here's a full list of basic probes:

### Basic Probes

- **Past & Future** — Experts can figure out how a situation developed, and they can think into the future to see where the situation is going. Amongst other things, this can allow experts to head off problems before they develop. *“Is there a time when you walked into the middle of a situation and knew exactly how things got there and where they were headed?”*
- **Big Picture** — Novices may only see bits and pieces. Experts are able to quickly build an understanding of the whole situation — the Big Picture view. This allows the expert to think about how different elements fit together and affect each other. *“Can you give me an example of what is important about the Big Picture for this task? What are the major elements you have to know and keep track of?”*
- **Noticing** — Experts are able to detect cues and see meaningful patterns that less-experienced personnel may miss altogether. *“Have you had experiences where part of a situation just ‘popped’ out at you; where you noticed things going on that others didn’t catch? What is an example?”*
- **Job Smarts** — Experts learn how to combine procedures and work the task in the most efficient way possible. They don’t cut corners, but they don’t waste time and resources either. *”When you do this task, are there ways of working smart or accomplishing more with less — that you have found especially useful?”*
- **Opportunities/Improvising** — Experts are comfortable improvising — seeing what will work in this particular situation; they are able to shift directions to take advantage of opportunities. *”Can you think of an example when you have improvised in this task or noticed an opportunity to do something better?”*
- **Self-Monitoring** — Experts are aware of their performance; they check how they are doing and make adjustments. Experts notice when their performance is not what it should be (this could be due to stress, fatigue, high workload, etc) and are able to adjust so that the job gets done. *”Can you think of a time when you realised that you would need to change the way you were performing in order to get the job done?”*

Optional Probes:

- **Anomalies** — Novices don’t know what is typical, so they have a hard time identifying what is atypical. Experts can quickly spot unusual events and detect deviations. And, they are able to notice when something that ought to happen, doesn’t. *”Can you describe an instance when you spotted a deviation from the norm, or knew something was amiss?”*
- **Equipment Difficulties** — Equipment can sometimes mislead. Novices usually believe whatever the equipment tells them; they don’t know when to be skeptical. “ *Have there been times when the equipment pointed in one direction, but your own judgment told you to do something else? Or when you had to rely on experience to avoid being led astray by the equipment?”*

These probes act as the starting point for conducting the knowledge audit interview. After opening with each probe, you ask for specifics about that example, focusing on critical cues and strategies that the expert employs. Then, you follow up with a discussion of potential errors that a less-experienced person might have made in the situation.

What you should expect to get out of the knowledge audit is a table like this:

![](https://commoncog.com/content/images/2021/04/Knowledge-Audit-Table-1.png)

Sample Knowledge Audit Table

## Simulation Interview

Next up is the simulation interview. Whereas the knowledge audit is designed to elicit cues and strategies in the context of real-world examples from the expert’s past experience, the simulation interview is built around the presentation of a challenging scenario to the expert. Militello and Hutton write:

> The authors recommend that the interviewer retrieves a scenario that already exists for use in this interview. Often, simulation and scenarios exist for training purposes. It may be necessary to adapt or modify the scenario to conform to practical constraints such as time limitations. Developing a new simulation specifically for use in the interview is not a trivial task and is likely to require an upfront CTA (cognitive task analysis) in order to gather the foundational information needed to present a challenging situation. The simulation can be in the form of a paper-and-pencil exercise, perhaps using maps or other diagrams. In some settings it may be possible to use video or computer supported simulations. Surprisingly, in the authors’ experience, the fidelity of the simulation is not an important issue. The key is that the simulation presents a challenging scenario.

Of the four techniques in ACTA, picking a good simulation seems like the trickiest part of the methodology. I wish I could be of more use here — I spent half an hour googling for guidance in the Naturalistic Decision-Making literature, but couldn’t find a generalised list of tips on picking a good simulation. I suppose you’ll have to experiment to figure out what works best for your unique skill domain; the key thing about simulations is that they are usually a series of events that occur, and must be revealed one at a time in order to — as the authors put it — present a challenging scenario to the expert.

The general flow of the simulation interview goes like this: you present the first part of the simulation to the expert (e.g. describe the initial setup, or give them a brief, etc), and then you prompt the expert to identify major events, including judgments and decisions, with a question like “As you experience this simulation, imagine you are the *(job you are investigating)* in the incident. Afterwards, I am going to ask you a series of questions about how you would think and act in this situation.”

Each event in the simulation is then probed for situation assessment, actions, critical cues, and potential errors surrounding that event. You record this information in a ‘simulation interview table’, which looks like this:

![](https://commoncog.com/content/images/2021/04/Simulation-Interview-Table.png)

Sample Simulation Interview Table

You’ll want to redo the simulation interview with multiple experts, since different people may provide insight into situations in which more than one action would be acceptable, and alternative assessments of the same situation are plausible. You could even use the technique to contrast expert and novice perspectives by conducting the same simulation interview with people of differing levels of expertise.

The goal, once again, is to get at the expert’s cognitive processes within the context of an incident. If you combine the results of the simulation interview with the knowledge audit, you should get a pretty good sense of the types of expertise at play.

## Cognitive Demands Table

The final technique in the ACTA arsenal is a presentation format that lets you sort through and analyse the data. This is known as a ‘cognitive demands table’, and it's the final artefact that is produced at the end of the ACTA process, for use in your project.

The cognitive demands table presented below is taken from analyses that Militello and Hutton have conducted in the past. They say that you should pick different table headings, depending on the types of information that you would need to develop a new course or design a new system (or, in our case, learn skills that live in the heads of experts around us) — whatever it is that you want to do. More importantly, filling up the table should help the practitioner spot common themes in the data, as well as conflicting information given by multiple experts.

And it looks like this:

![](https://commoncog.com/content/images/2021/04/Cognitive-Demands-Table.png)

Sample Cognitive Demands Table

## Wrapping Up

Applied Cognitive Task Analysis seems easy enough to use. Of course, the caveat here is that the method was designed for use in instructional design, or systems design, not necessarily for extracting the tacit mental models of expertise of the people around us for the purposes of individual learning.

But I think it's possible to adapt it to those ends.

As it happens, I’m currently doing a repositioning exercise for a SaaS product, using April Dunford’s [*Obviously Awesome*](https://commoncog.com/obviously-awesome/) as a template for execution. Part of that exercise involves working alongside salespeople and getting at the expertise in their heads. If you're in sales, you probably have a good sense for what customers want. And that in turn means that there's something there for marketing to use.

All this is to say that I can't wait to put ACTA to the test. And when I'm done with that, I’ll let you know how it goes.

Read [Part 6 — The Tricky Thing About Creating Training Programs](https://commoncog.com/creating-training-programs/). You may also read [John Cutler’s Product Org Expertise](https://commoncog.com/john-cutlers-product-expertise/), which features ACTA.

Originally published , last updated .

[![Feature image for Tacit Expertise Extraction, Software Engineering Edition](https://commoncog.com/content/images/size/w300/2024/02/tacit_knowledge_software_engineering.jpg)](https://commoncog.com/tacit-expertise-extraction-software-engineer/)[![Feature image for One Definition of Wisdom](https://commoncog.com/content/images/size/w300/2024/02/charlie_munger_wisdom.jpg)](https://commoncog.com/one-definition-of-wisdom/)[![Feature image for Mental Strength in Judo, Mental Strength in Life](https://commoncog.com/content/images/size/w300/2023/05/mental_strength_judo_life-1.jpg)](https://commoncog.com/mental-strength-judo-life/)[![Feature image for Creating New Drills for Deliberate Practice](https://commoncog.com/content/images/size/w300/2023/04/deliberate_practice_create_new_drills.jpg)](https://commoncog.com/creating-drills-deliberate-practice/)

---
title: "The Tricky Thing About Creating Training Programs"
source: "https://commoncog.com/creating-training-programs/"
author:
  - "[[Cedric Chin]]"
published: 2021-05-18
created: 2025-05-30
description: "We take a look at how you might turn extracted, tacit expertise into a training program for yourself or others."
tags:
  - "clippings"
---
By

![Feature image for The Tricky Thing About Creating Training Programs](https://commoncog.com/content/images/size/w600/2021/05/create_tacit_knowledge_training_program_naturalistic_decision_making.jpeg)

## Table of Contents

## Fan of great business?

Join 8,000+ sharp investors and operators like yourself, and we'll send you a collection of Commoncog's best articles right away:

*This is Part 6 in a [series on tacit knowledge](https://commoncog.com/the-tacit-knowledge-series/). Read [Part 5](https://commoncog.com/an-easier-method-for-extracting-tacit-knowledge/) here.*

In our [previous post](https://commoncog.com/an-easier-method-for-extracting-tacit-knowledge/) in the tacit knowledge series we took a look at Applied Cognitive Task Analysis (ACTA), a simpler technique to extract tacit expertise from the heads of experts around you.

A natural response to that post is … “ok, so after you extract that tacit knowledge — now what?” This is an understandable reaction, because the next step after doing all that skill extraction is to use it to *learn* — to develop a training program for yourself or for others. The main problem with this is that good pedagogy is difficult to do. And, as it happens, I’ve not spent any time talking about how Naturalistic Decision Making (NDM) researchers turn their insights into actual training programs.

Why? Well, this is simple: I haven’t found a good, generalised resource in the NDM literature on how they do so. What I *have* found is a place in the NDM world to dig; this post is to tell you all about it.

## Some Context

I’ve [mentioned previously](https://commoncog.com/tacit-knowledge-is-a-real-thing/) that the entire field of Naturalistic Decision Making is built around this notion of ‘we’ll interview experts, and then we’ll extract the expertise from their heads, and then we’ll turn that into training programs or use the outputs to redesign user interfaces’. I said that the field is incredibly applied; much of the work was funded by the military or by corporations with Serious Expertise Needs (think: nuclear power plant control systems, fraud detection at Nasdaq, etc). We’ve already spent an entire [series of blog posts](https://commoncog.com/the-tacit-knowledge-series/) looking at some of NDM’s techniques for extracting expertise, which is interesting for all the obvious reasons: how many of us, after all, have had the experience where we want to get better but the expertise we desire is locked up in the heads of senior practitioners? There aren’t a lot of places you can go to that specialises in skill extraction; NDM is one of the few subfields of psychology where the researchers have had success extracting tacit mental models of expertise.

What we *haven’t* done is to look at NDM’s techniques for creating effective training programs.

This is partly my fault, and partly the nature of the beast. It’s my fault because I’m relatively new to the literature — I didn’t, for instance, know that ACTA was a thing until a few months ago, despite the importance of the paper. And it’s the nature of the beast because I think popular NDM writing is very much focused on the ‘skill extraction’ part, and not so much the ‘design a training program’ part. Presumably this is because there are already many resources out there to design good training programs. And perhaps NDM researchers think that their approach to course design isn’t particularly interesting; certainly the individual teaching techniques they use here aren’t particularly novel.

But what is interesting to an academic is very different to what is interesting to a practitioner. I happen to think that NDM researchers have a unique approach to education design, as a result of the skill extraction they are able to do. And I am *very* interested in some of these techniques, because I want to apply them to my own career.

In fact, I think there’s a particular ‘style’ to the types of training programs that NDM researchers create. I want to give you a taste of that approach through the use of three quick examples. Of course, examples aren’t as good as a set of generalised guidelines, but I haven’t found any. So examples it is.

(Again: if I’m wrong here, please send me an email at cedric@this domain. I’d love pointers to papers!)

## IED Defeat

IED stand for ‘improvised explosive device’. You’ve probably seen them in the 2008 award-winning movie *[The Hurt Locker](https://en.wikipedia.org/wiki/The_Hurt_Locker?ref=commoncog.com)*.

In [episode four](https://www.listennotes.com/podcasts/naturalistic/episode-4-interview-with-5av4UMGzFPf/?ref=commoncog.com) of the Naturalistic Decision Making podcast, Jennifer Phillips talks about some of her work on IED defeat. She starts out by describing the skill extraction aspect of it:

> Soon after 9/11, the military had troops in Afghanistan and Iraq, and if you remember, all the coalition forces had problems with these roadside bombs, these IEDs. At that time, the Department of Defence was trying to throw a lot of money at advanced technologies to detect these devices, and better protection and armour and such in the vehicles to protect Marines and soldiers, and one of the areas that we had the opportunity to get involved with was the human expertise element of this, of detecting where the devices were going be placed, and the reason this project was so interesting to me and so rewarding … it truly felt like we were in a position to make a difference in Marine’s and soldier’s lives.  
>   
> We got to interview a number of young Marines who had done some really, really challenging jobs, and who had seen some terrible things in their lives, and to be able to extract their expertise and see not only what they had been through, but also, the skills that they developed. I mean, talk about richness of a domain — every single expert-novice difference that exists out there, you see in this domain of IED defeat, of being able to notice these roadside bombs and anticipate where they be located. You see these Marines seeing the invisible, recognising when things that should be there aren’t there, you see how they put themselves in the perspective of the adversary and think about where they would emplace given the terrain, given the time of day, given where the sun is in the sky — I mean, this whole range of rich factors and cues they would rely on to know that they were getting into a danger zone was just fascinating intellectually.

![](https://commoncog.com/content/images/2021/05/marines_deployment.jpg)

A screenshot of VBS3

What is interesting about IED defeat is the existence of a clear [meta](https://commoncog.com/to-get-good-go-after-the-metagame/). At around 28:30, co-host Brian Moon starts probing about the particular difficulty of this skill domain:

> BM: What I thought was particularly challenging about (this project) is — we think about experts and they develop expertise over time, in this particular domain, because things were happening so fast … the adversary was adapting more rapidly it seemed than our folks. And so expertise in that sense, was extremely contextualised, both in time and space — and so, even as a Marine might develop expertise in a particular area, and understand the cues, those might change when they go to the next region. I always thought this was an interesting aspect of the work, because if you think about expertise in nuclear engineering, for instance, yes, there are changes in that domain, but not at the rapid pace that you saw in IED defeat.  
>   
> JP: You’re absolutely right. So we were looking at Afghanistan and Iraq, two very different areas of responsibility. You would find, even within one of those countries, the different regions, the different neighbourhoods, the insurgents would have different techniques for emplacing and triggering and all those things. So you would hear stories in the interviews about … in one town, any time there was a rug that was hung over a balcony, then that meant there was going to be an attack that evening. So that’s a cue you would pull out in your interviews, but that’s not one that’s going to transfer to a different context necessarily. So what do you do with that?  
>   
> So it goes back to the question of ‘how do you take this information and build something that’s going to help someone else in any context?’  
>   
> We had to work pretty hard on identifying some of those commonalities. One example of a commonality might be a ‘trigger point’ — so if you’ve got a radio-controlled IED, you need to have a spotter that’s located somewhere in the terrain, who can have eyes on the road. And they need to judge where the blue force is — where the Marines are located — at what point to trigger the device based on where they’re located on the road. So in some contexts they would use a telephone pole post, in other contexts they would build their own little rock formation on the road, in other contexts they would use spray paint or something like that to mark a section on the road, so the concept, the technique, was similar across contexts, even though it might have been implemented a little bit differently, depending on what was available in that particular context.

In the end, the training program was implemented as a video game:

> BM: Can you talk just for a few minutes about this idea of using games, about using video games in particular, and your mantra that I’ve heard you and Carol mention about this ‘cognitive fidelity’, and how some of the training applications ought to be focused a bit more on this cognitive fidelity.  
>   
> JP: Sure, so this was so interesting. One of the big findings that came out of that CTA (Cognitive Task Analysis) was that the Marines who were really doing a great job at being able to recognise a danger zone in advance were the ones who were thinking like the adversary. It sounds like a pretty simple thing, like no kidding of course they were able to put themselves in the adversary’s shoes, but we thought this is something that young Marines, the individuals who were deploying for the first time, don’t know how to do. They’ve never had the ability — outside a training environment, and it’s not something that’s really been taught in training, at least not in a standard way — they’ve never had the opportunity to think from the other guy’s perspective. So we used the [VBS technology](https://bisimulations.com/products/vbs3?ref=commoncog.com) — we teamed with [Bohemia Interactive](https://www.bisimulations.com/?ref=commoncog.com) — and we built out a module in VBS where the Marines were role-playing the adversaries, and they were responsible for emplacing IEDs in the environment.  
>   
> So we were essentially putting them in a position where they had to think through how they were going to be effective in placing an IED. Whether they were going to use a cellphone detonator, or … with another sort of IED, they had to think through ‘When is the blue force convoy going to come through? What is the time of day right now? How can I disguise it? Where would be a good ambush zone?’ All of those kinds of considerations that are happening in the real world.  
>   
> Now if you know VBS, you know that it’s got pretty good physical fidelity, but not great physical fidelity. So to your question about the cognitive fidelity really mattering … it didn’t matter that the physical video game environment was perfect or that the leaves on the trees were blowing in the wind exactly as they’re supposed to, what mattered was that we were putting the Marines in a position where they had to think through the problem from a different perspective, and that turned out to be very successful.

![](https://commoncog.com/content/images/2021/05/army_vbs.jpg)

A screenshot of VBS3

## Tennis Serve Detection

The next podcast episode I want to draw your attention to is [episode nine](https://www.listennotes.com/podcasts/naturalistic/episode-9-interview-with-mjZTC4ai36o/?t=610&ref=commoncog.com), with Peter Fadde and Olivia Brown. And the example that I want to talk about is Fadde’s research with training tennis serve detection:

> Expertise-based training really is a theory — an instructional design theory. We’re probably a pretty small group within the NDM community compared to human factors, cognitive psychology, or some other areas like that. And the theory really is that, hey, we can go out in the field and study expertise, study these amazing, seemingly intuitive decision-making in sports and military and other contexts, but how do we train it? Training isn’t a natural thing, training is an artificial thing. So we’re looking at something and we’re trying to understand it …  
>   
> So the insight there was “ok, how do they research it?” — and that led me to.. because when I went to do my dissertation, I was looking around for a project that was doable, and I was working in sports at the time — I was working as a video coordinator with a big time football team, and we did video analysis for coaches. And I just felt that more that can be done for player training by putting video and data together, so pulling on that string led me to research that at the time a lot of it was done at Australia, especially at the [Australian Institute of Sport](https://www.ais.gov.au/?ref=commoncog.com), people like [Bruce Abernathy](https://researchers.uq.edu.au/researcher/222?ref=commoncog.com) really leading that research …  
>   
> And they developed these techniques, and the techniques themselves was of interest. Because they have to take this incredibly complex, seemingly intuitive performance, that should be impossible. If you look at a goalie blocking a shot on goal, a baseball, a cricket, a softball hitter returning a 135-per-hour serve, these things are not really possible. The human perceptual and motor systems are not sufficient to have any success at this at all.  
>   
> (…) but how are you going to systematically train it — how can you get more people over the bar faster? And the way you study it in the labs, is with these occlusion techniques, in particular, temporal occlusion, it cuts off in time. Presented by video.  
>   
> So with a cricket baller, here’s the run-up, here’s the ball out of hand, boom, cuts to black. And the research subject has to say, well, was that an in-spinner, or whatever category it is, so it’s a categorisation task, and it’s a prediction task — is it going to hit the wicket, and if it’s baseball, is it a ball or a strike? So categorisation and prediction. So now we’re starting to really kind of narrow it down. You can say that the macro-cognitive skill we’re after is *anticipation*.  
>   
> “Well, how do you train anticipation?” (And people would say): “Well, you can’t train anticipation, that comes from experience.”  
>   
> Well, ok. Fine. What is the experience giving you? So we can reduce that back to, ok, it’s some pre-event cues and then it’s prediction. Ok — we *can* test prediction, and we *can* train prediction. So let’s take something like anticipation which is the quality that we’re after, and let’s try to operationalise it in some way that we can train, which is prediction.  
>   
> So the thing was to take the video occlusion method that they used for testing, and turned it into a training program.

And this is exactly what they did (22:23):

> This really started in the early 80s in Australia, with a focus on cricket and tennis. This is the tennis serve return. So you’ve got video of a server, that’s recorded more or less from the point of view of a tennis receiver. So the server tosses the ball, racket comes through, and then cut to black. And you have to say what kind of serve is that?  
>   
> Now that cut to black might be right after ball contact, as you see the ball coming off the strings of the racket, or right before ball contact, or right in it. So they cut it at these different places in here, and this is an expert-novice research paradigm, so they have advanced tennis players and less skilled tennis players (come in to be tested), and where does their performance diverge?  
>   
> So for instance in that one we find that if they see even one frame of video — that’s the sharpest knife we have here — you’re dealing with about 33 milliseconds in there … if they see two of those (frames) coming off (the strings of the racket), so you’ve got 67 milliseconds, still a very short period of time, then the experts and the novices can pretty much say what type of serve that was — that was a kick serve, that was a flat serve, or that was a slice serve. Now you cut back to where you can only see a little bit of ball flight … well, their performance both goes down a bit, but the expert not as much. Now we cut to black before the racket hits the ball. And the experts are *still* able to say what type of serve that is — they haven’t even seen the racket hit the ball, and they can tell you 85% of the time whether it’s a kick, a flat, or a slice serve. And they can even predict backhand or forehand side on the return. And at that point the novices performance pretty much drops off. Now if we cut it back a little bit further, they’re both down to random (prediction).  
>   
> So now we’ve defined our window of expert advantage. From 50 milliseconds before contact, to 50 milliseconds after contact. There’s a 100 millisecond window in there when the experts have a distinct advantage over the non-experts. So that’s the research finding.  
>   
> Then if we repurpose that as a training technique, we say “ok, here’s the beginning of the window, you should be able to do that, everyone should be able to (detect the serve)”, and then we start cutting it back and you see less and less and less, until you, as a developing tennis player, are able to, like the experts, at 85%, identify the serve 50 milliseconds before the contact.

![](https://commoncog.com/content/images/2021/05/tennis_serve.jpg)

The interesting thing is that the experts are terrible at explaining what they’re looking for, so they have to look at other techniques to pick up on the cues (28:15):

> The experts can’t necessarily see what they’re seeing — they almost put it in the ‘ESP category’, like with the firefighters in (Gary) Klein’s story (in [Sources of Power](https://www.goodreads.com/book/show/65229.Sources_of_Power?ref=commoncog.com)). And if you press them enough, they’ll start making stuff up. The way that experts do, because they want to give you an answer. So you really need to start ascertaining where that is.  
>   
> And so this is where you can bring in another occlusion technique, which is spatial occlusion. We know we have a window in time where that expert is picking up some kind of cues. It’s not ESP, so we know they’re picking up some kind of cues (but) we don’t know what they are. So what happens if we mask out the racket?  
>   
> Well, as it turns out, we can mask out the racket, and the novice’s performance will just fall off the table, and it will hardly affect the expert’s performance at all. So that tells us that they aren’t getting their cues from the racket. But if we mask out the lower half (of the server’s body) now, the experts ability to identify that serve when it’s cut off really goes down. So that tells us that somewhere in that lower half, they’re picking up very early cues, so there’s plenty of time to help you make that — what seems like an instantaneous reaction. And you can break it down enough that you realise that if someone is hitting a kick serve, they’re going to be a little deeper in their knee bend, their ball toss is going to be behind their head, if it’s a slice, the ball toss will be a little to the side. Some of those are known coaching things — the ball toss, for instance.  
>   
> And you see the reverse of it. For instance, in Pete Sampras’s biography, he talks about how — when he was 12 years old, he would toss the ball into the air, and his coach would call out the type of serve to hit from there. So he would have to learn to hit kick, flat, or slice from the same ball toss. And so later, when he played in the US Open against Agassi, considered the best returner of the time, Agassi said that he couldn’t read Pete’s serve. It wasn’t the fastest flat serve, it wasn’t the spinniest kick serve, but it all looked the same.

Fadde’s true contribution to the research literature here is that the training approach focuses on the cognitive elements of a physical skill. Fadde says (16:42):

> What’s some way that you can observe some representative task in a measurable and repeatable way? Well, that turns out to describe a training task too, and it tends to be a little bit cheaper. (Because you’re re-using research techniques).  
>   
> So my initial dissertation research was in applying that. And I had access to a competing college baseball team, and so I evolved a training program out of that, to train their pitch recognition. And that proved to be quite successful, there were measurable results in batting average and such, so that continued to be a focus, but also with this larger thing of saying “not only are we taking this one skill, but this can also apply to other kinds of things.” Particularly ‘in-the-action’ decision making. Not command-and-control type things. So in the news these days, things like attack recognition, defensive tactics, arrest and control by law enforcement, so that you make good, fast decisions, for the good of everyone concerned. All of these types of performances that seem too fast and too complex to tease apart.  
>   
> And (my contribution) is taking these techniques from the research lab and repurpose them as drill and practice. Drill and practice: the bottom feeder of all instructional technology. What elements of extraordinary performances that we observe in many contexts — they just happen to be more measurable on TV and in sports, but they exist everywhere — how can we reduce that to something that can be drilled and practiced?  
>   
> And if there’s one big insight there, these performances and the training of them tend to be very focused on what you do, and yet the NDM-type research that you come across, the higher you go in the expertise tree, the more it becomes about what you see, instead of what you do. So we’re going to push that aside, the ‘what you do’ bit — we’re not saying it’s not important, we’re saying it’s already addressed (in current training methods). What’s under-addressed and what gives us the opportunity to perhaps accelerate the development of expertise is this focus on what you see. And it turns out that applying some of these video-occlusion type things can be done on a drill-and-practice level, and they can be done on an iPhone. So we can get a lot of efficiencies out of that kind of approach.

## A Tool for Police Investigations

The last podcast episode that I want to draw your attention to is [Episode 20](https://www.listennotes.com/podcasts/naturalistic/episode-20-interview-with-FQzuNxoIbhv/?ref=commoncog.com), with William Wong.

The bulk of Wong’s work over the past couple of years is “how do we develop better user interfaces for intelligence analysis?” This led him to create the [VALCRI project](http://valcri.org/?ref=commoncog.com), which is currently in use in the EU, and is described as ‘a criminal intelligence analysis system that facilitates human reasoning and analytic discourse, tightly coupled with semi-automated human-mediated semantic knowledge extraction.’

It is, to put it simply, the real world version of the crime investigation system from *Minority Report*.

![](https://www.youtube.com/watch?v=PJqbivkm0Ms)

And it looks like this:

![](https://www.youtube.com/watch?v=7BYX0vlLro0)

Wong covers the story of the project in the podcast at about the [29:26 mark](https://lnns.co/bR6IqmD04PZ/1766?ref=commoncog.com), explaining that “when we started the work, I was really intent about designing a system that would focus on actually creating an environment that would enable insight. Particularly the kind of sensemaking described by Gary Klein’s Data-Frame model, and particularly his triple path model of insight (…) I wanted them (in the consortium) to think about sensemaking and insight, because if you don’t have that, all you’re going to build is a search and retrieval system, which is what every other intelligence analysis system is about!”

That cognitive model for insight that they unearthed from doing CTA on criminal analysts is covered in this VALCRI video:

![](https://www.youtube.com/watch?v=tw-8fVXFxII)

And the three research papers that informed the design of the project were the following:

1. [How Analysts Think: Inference Making Strategies](https://journals.sagepub.com/doi/10.1177/1541931215591055?ref=commoncog.com)
2. [How Analysts Think: Anchoring, Laddering and Associations](https://journals.sagepub.com/doi/abs/10.1177/1541931213601040?journalCode=proe&ref=commoncog.com)
3. [How Analysts Think: Intuition, Leap of Faith, and Insight](https://www.researchgate.net/publication/308181057_How_Analysts_Think_Intuition_Leap_of_Faith_and_Insight?ref=commoncog.com)

What I took away from the VALCRI project is really this notion of — as Wong puts it — “if you don’t have this notion of building for sensemaking and insight, then all you’re going to build is a search and retrieval system!” The history of the VALCRI project shows you how to do it right.

## Wrapping Up

What can we draw from these three examples? If I may generalise timidly, I would say that all three show an approach where the researcher extracts the cognitive processes of experts, and then teaches the novice by making them simulate those same cognitive processes. The key word here is *simulation*:

- The IED defeat training program focused on high cognitive fidelity to the real-world task; the video game was simply a medium to help students acquire the mental model that seasoned marines use to predict IEDs emplacement.
- The tennis training program worked because it took apart this tacit macro-cognition aspect of tennis serve recognition, and then taught it to novices by simulating through an occlusion exercise. Fadde could have taught it via lecture. He didn’t.
- The VALCRI project extracted the insight generation process that criminal intelligence analysts used, and then designed an entire system around it.

You’ll notice that all three examples are drawn from the [NDM podcast](https://naturalisticdecisionmaking.org/podcasts/?ref=commoncog.com). This isn’t an accident! My current hack to learn the NDM method of designing training programs is to listen to as many NDM podcast interviews as possible. My goal: to collect as many prototypes of effective training interventions as I can.

‘Listen to the NDM podcast’ happens to be my recommendation if you want to do the same. It strikes me as somewhat ironic: the best practitioners of tacit knowledge extraction have yet to explicate a coherent explanation of their training methods. You’d think that they would have a generalised framework for training by now. Or at least a review paper. And, again … perhaps they have! Perhaps they have a review paper somewhere, but I haven’t found it yet.

I’ll tell you if I do. Till then, the best that we have is casual consumption of the [NDM podcasts](https://naturalisticdecisionmaking.org/podcasts/?ref=commoncog.com). I recommend you join me in doing so. If you're as obsessed with learning as I am, I think you’ll enjoy it as much as I do.

Read [Part 7 — An Extracted Tacit Mental Model of Business Expertise](https://commoncog.com/tacit-mental-model-of-business/).

Originally published , last updated .

This article is part of the [Expertise Acceleration](https://commoncog.com/expertise) topic cluster. Read [more from this topic here→](https://commoncog.com/expertise)

Previous Post

#### ← A Land & Expand Reading Program for B2B Sales

Next Post

[![Feature image for Tacit Expertise Extraction, Software Engineering Edition](https://commoncog.com/content/images/size/w300/2024/02/tacit_knowledge_software_engineering.jpg)](https://commoncog.com/tacit-expertise-extraction-software-engineer/)[![Feature image for One Definition of Wisdom](https://commoncog.com/content/images/size/w300/2024/02/charlie_munger_wisdom.jpg)](https://commoncog.com/one-definition-of-wisdom/)[![Feature image for Mental Strength in Judo, Mental Strength in Life](https://commoncog.com/content/images/size/w300/2023/05/mental_strength_judo_life-1.jpg)](https://commoncog.com/mental-strength-judo-life/)[![Feature image for Creating New Drills for Deliberate Practice](https://commoncog.com/content/images/size/w300/2023/04/deliberate_practice_create_new_drills.jpg)](https://commoncog.com/creating-drills-deliberate-practice/)

---
title: "A Tacit Mental Model of Business Expertise"
source: "https://commoncog.com/tacit-mental-model-of-business/"
author:
  - "[[Cedric Chin]]"
published: 2021-07-20
created: 2025-05-30
description: "All great business people share a common, intuitive mental model of business. We look at how researcher Lia DiBello extracted that mental model."
tags:
  - "clippings"
---
,

## An Extracted Tacit Mental Model of Business Expertise

By

![Feature image for An Extracted Tacit Mental Model of Business Expertise](https://commoncog.com/content/images/size/w600/2021/07/mental-model-business.jpeg)

## Table of Contents

1. [DiBello's Main Ideas](https://commoncog.com/tacit-mental-model-of-business/#dibellos-main-ideas)
2. [The Mental Model of Business](https://commoncog.com/tacit-mental-model-of-business/#the-mental-model-of-business)
3. [Cognitive Agility](https://commoncog.com/tacit-mental-model-of-business/#cognitive-agility)
4. [The FutureView Profiler](https://commoncog.com/tacit-mental-model-of-business/#the-futureview-profiler)
5. [Sources](https://commoncog.com/tacit-mental-model-of-business/#sources)
	1. [Download the PDF Copy](https://commoncog.com/tacit-mental-model-of-business/#)

## Fan of great business?

Join 8,000+ sharp investors and operators like yourself, and we'll send you a collection of Commoncog's best articles right away:

*This is Part 7 in a [series on tacit knowledge](https://commoncog.com/the-tacit-knowledge-series/). Read [Part 6](https://commoncog.com/creating-training-programs/) here. This is also the start of a new [series on tacit knowledge in business](https://commoncog.com/business-expertise-series/).*

If you’re a long time reader of Commonplace, you’re probably familiar with my obsession with [tacit knowledge](https://commoncog.com/three-kinds-of-tacit-knowledge/) — the idea that certain types of expertise is locked up in the heads of experts, that they can’t really put such expertise into words, and that you should pursue techniques to acquire tacit expertise — methods like emulation, scenario training, and apprenticeship.

In my [first piece in the series](https://commoncog.com/tacit-knowledge-is-a-real-thing/), I wrote:

> It is not reasonable to wait for an expert systems revival, nor is it fruitful to expect CDM to be applied to your field, or for a Boyd-like genius to appear. We should act as if tacit knowledge were a fact, because it is more useful to think about ways to gain that tacit knowledge directly, instead of hoping for some breakthrough to make tacit knowledge explicit.

Over the course of writing this series, however, I’ve come around to the idea that it *might* be possible to extract tacit knowledge from the heads of experts around you. The turning point was my [discovery of the ACTA paper](https://commoncog.com/an-easier-method-for-extracting-tacit-knowledge/), which teaches a radically simplified method for cognitive task analysis. (Cognitive task analysis, or CTA, is the set of techniques that Naturalistic Decision Making (NDM) researchers use to extract tacit mental models of expertise from the heads of subject matter experts).

That said, it’s nearly always more fruitful to mine the literature for *already* -extracted mental models of expertise. A few weeks ago, I learnt of Lia DiBello, an NDM researcher who specialises in the extraction of *tacit mental models of business expertise.*

DiBello’s findings are extremely compelling. She writes, in *[The Oxford Handbook of Expertise](https://www.oxfordhandbooks.com/view/10.1093/oxfordhb/9780198795872.001.0001/oxfordhb-9780198795872?ref=commoncog.com)*:

> After several years using different methods, we noticed that highly talented business performers are very similar to each other. That is, they have decoded the domain of business much like chess masters had decoded chess. This led us to realise that —as a domain— **business itself is an orderly closed system of relations between principles, and that so-called intuitive experts in business have an implicit grasp of this**. If this is the case, it would follow that all highly skilled business experts should recognise each other and share a common mental model of these principles, although they may manifest differently in different industries. **Since business has been evolving in recent years, this model would have to be very robust, running deeper than recent societal changes.***(emphasis mine)*

These claims sound ridiculous. A closed system of relations between principles? A shared common mental model of those principles? *No way*.

As a result of that chapter, I dove into DiBello’s publication history. This turned out to be relatively easy — I quickly learnt that DiBello has published a relatively small number of papers, especially when compared against the research output of some expertise researchers. Initially, I found this frustrating, but then I realised that she has spent the bulk of the past 20 years consulting for big companies, *applying her research to the real world*. (During her [interview on the NDM podcast](https://www.listennotes.com/podcasts/naturalistic/episode-28-interview-with-ydLRG3pAaB0/?ref=commoncog.com), she says (54:04) somewhat ruefully, “Because of how I wanted to get a lot of data, I ended up selling my soul to big business, and I made a lot of rich people richer. But it was how I got to prove that what we do works.”). She spends more time today attempting to adapt her training methods for broader uses, to make them available to businesspeople who might not be able to afford a full consultation. To my mind, this makes her more, not less, [believable](https://commoncog.com/believability/).

And what a body of work she has.

This piece is a high-level summary of what I’ve found. It marks the start of a new series on tacit business expertise; I intend to dive deep into some of the more subtle implications of her work in the next handful of posts. I’ll also include a full list of citations and links to papers, along with a scan I made of her chapter in *The Oxford Handbook of Expertise*.

Let’s start out with what DiBello found.

![](https://commoncog.com/content/images/2021/07/oxford_handbook_of_expertise.jpg)

The Oxford Handbook is a remarkably chonky book.

## DiBello's Main Ideas

DiBello’s work is remarkable because she *gets business.* In her NDM podcast interview she explains that she came from a business family; naturally, she gravitated to the study of business experts after her PhD.

The bulk of her work follows from these three big ideas:

1. Business expertise consists of a handful of properties: (a) it is team-based — meaning that business expertise is actually a form of *distributed cognition*, with expertise spread amongst an executive team; (b) it consists of two things: domain-specific mental models of the business *and* cognitive agility; and (c) there is a deeper mental model that underpins those domain-specific mental models, which explains how good business people are able to recognise other good business people, even if they are from different industries.
2. She argues that general problem solving ability does not matter so much in business so much as a property she describes as *cognitive agility* — which is the ability to update mental models in light of new information.
3. Her primary innovation is an intervention known as the FutureView Profiler. The Profiler is able to evaluate the shared mental models of an executive team: it is able to (a) identify blind spots in the executive team’s model of the business, (b) able to evaluate cognitive agility, and (c) able to measure expertise in business primarily because it asks business people to do *year-by-year predictions of similar companies within their own industries*.

I’ve highlighted the most immediately useful elements of her work above, but I want to note that the ideas sit on top of a few other contributions that are novel from an expertise research perspective:

1. DiBello’s insight that domain-specific **prediction tasks** are a useful way to evaluate business skill is a huge contribution to the literature on expertise. She spends a few paragraphs in her 2010 paper \[2\] discussing why the Profiler works, and why other assessments of business skill do not. (The gist of the argument is that you want to evaluate how a business person’s mental models update in response to a *dynamic context*; you do not want to evaluate what currently exists in their heads).
2. Her company, Workplace Technology Research Inc (WTRI), has spent years implementing company training simulations in virtual worlds. Apart from the obvious technical effort (she explains \[3\] that they decided to code a full world of their own, on a game engine of their choice, mostly because existing companies they collaborated with kept getting acquired or shut down) — she has novel insights on the pedagogical effectiveness of using virtual game worlds to teach business simulations.
3. Last, but perhaps most importantly, DiBello argues that business expertise is primarily a matter of *organisation*. She writes, in the *Oxford Handbook*: “Like experts in other domains, business experts rely not on greater analysis or greater information, but *better ways of structuring or organising their knowledge*. Further, a business expert differs in the *manner in which he or she looks at the business landscape, particularly with regard to what is linked with what and what is important to manipulate*.” As a result, the bulk of DiBello’s training methods aim to do what is called ‘cognitive reorganisation’ — that is, WTRI does not seek to teach business people new concepts about their domain per se, but to instead reorganise existing domain-specific mental models in their heads.

These ideas — in particular the observation about cognitive reorganisation — are conceptually interesting, and important if you want to implement DiBello’s training methods in your company. But they aren’t necessary for our current discussion, and they are difficult to grok. (In fact, it appears to be an open question if one is able to recreate WTRI’s training methods; DiBello has said that many have tried — even with assistance! — and failed).

I’ll set these ideas aside for now, and spend the rest of this piece examining the tacit mental model of business expertise that DiBello successfully extracted two decades ago.

## The Mental Model of Business

In the *Oxford Handbook*, DiBello writes:

> In a study funded by the National Science Foundation conducted over four years (NSF Award ENG 9548631) we found that those who show considerable and consistent talent in business have such a mental model, shown in predictable ways of making use of information. Leveraging our relationships with companies and clients, we had unusual personal access to study a large number of highly placed leaders. Doing in-depth studies of talented business leaders who were repeat successes—even in very challenging markets—and who had risen to very high positions (such as chairmen of large corporations) and maintained that level of position even as business itself has grown much more complex, we discovered a shared mental model among all of them. \[1\]

That shared mental model is essentially a triad:

> Our research revealed that people who have achieved a high level of business expertise have a deep understanding of the following three core areas: **(1) factors involved in effective operations, (2) forces influencing the market, and (3) those driving business finance and economic climates.** Consistently successful business leaders have been shown to **intuitively understand these areas and their impact on each other, and to pay attention to this fundamental triad in a uniquely dynamic way within a guiding context of business strategy.** For example, these experts are able to sense that market conditions change based on environmental indications that others may fail to notice. Further, they foresee the consequences to their business operations and finances, and thus make necessary adjustments proactively. Unlike most business professionals, they are attuned to the early indicators of widespread change. Beyond this, they are **expert at keeping the triad in balance, or shifting the balance when external conditions are conducive to do so** (e.g., focusing on marketing during favourable economic times). *(emphasis added)* \[2\]

Throughout the paper, DiBello describes the triad using different names. One way of describing it is ‘supply/demand/capital’. Another is ‘leadership/strategy/finance’. To summarise:

1. **Supply, or leadership** — These represent operational concerns related to the business, what DiBello calls ‘factors involved in effective operations’. Naturally, how the operational concerns are expressed depends on the industry (e.g. you have to be good at [product development](https://commoncog.com/product-validation-taste/) if you’re in consumer software, and you must have competent [manufacturing if you’re Chobani](https://twitter.com/NeckarValue/status/1416833586372698116?ref=commoncog.com)). Think expansively when it comes to each category of the triad — this category includes things like org design, incentive structures, execution cadence, and even the ability to get things done within the company (e.g. execs must have good mental models of the org and its people, and the ability to push through plans given internal politics).
2. **Demand, or strategy** — These represent the exec’s understanding of the market, what DiBello calls ‘forces influencing the market’. Of course, market dynamics is a very broad category, and includes everything from market shape, competitive analysis, positioning, changing consumer demand, and the ‘path to power’ that we’ve talked about in our [discussion](https://commoncog.com/7-powers-in-practice/) of *[7 Powers](https://commoncog.com/7-powers-summary/)*.
3. **Capital, or finance** — At the basic level this represents the exec’s understanding of basic financial concepts like [cash flow lockup](https://commoncog.com/cash-flow-games/), return on invested capital, margins, and their relationship with the other two categories (what DiBello calls ‘factors driving business finance and economic climates’). The factors here may be equally expansive — depending on the business, the capital leg of the triad may include expertise with raising capital (e.g. equity financing, debt) or the ability to understand the implications of changes in the capital environment (e.g. John Malone was amongst the first in the cable industry to take advantage of junk bonds in order to [finance expansion](https://commoncog.com/cash-flow-games/#john-malone-and-the-invention-of-ebitda); a more contemporary example might be startup founders who understand how to manipulate extremely liquid private markets to their advantage).

As DiBello mentions, the key property of business expertise is in understanding how a change in one leg of the triad affects the other two legs, at least within the context of one’s specific industry. Experts are able to notice cues in the environment long before novices are able to, and are then able to forward-simulate what must be done to the other two legs of the triad. So, for example, if there is a change in the capital environment, an expert businessperson might ask how that affects competition (this is the market — the demand leg of the triad). And they might immediately ask what changes the company must make in response to those market shifts? (This is the supply leg of the triad).

This mental model of business is incredibly high-level, but *it tells us a lot.*

For instance, it tells us that expertise may be evaluated according to the facility an exec has with each of the three categories. DiBello writes:

> In contrast, **competent managers (one level down from experts) tend to be very talented in only one or two of these areas; however, they often do not understand the dynamics between these areas as well as the “superstars” do**. Competent managers are likely to be very successful when larger market or economic trends are favourable to their specific skills. *(emphasis added)* \[2\]

The implication is that if you want to get really good at business, you should systematically acquire skills in each of the three categories, specific to your particular business, and then — more importantly — *grok the relationships between the three categories*.

It also explains another observation that DiBello makes:

> Experts in the same domain easily identify each other. Even if they don’t agree with another expert’s choices, they easily recognise that they share a common perceptual experience within their area of expertise. \[2\]

With the triad model, we may begin to understand how an experienced businessperson is able to recognise another experienced businessperson from a different industry, despite having no idea of the specifics of the other expert’s business or market. The answer to this puzzle is that they are able to recognise the intuitive facility with which the other person reasons about supply, demand, and capital in their own businesses! When I read DiBello’s articulation of the model, I thought back to certain businesspeople who have told me that they can tell if someone ‘gets it’, even if they can’t articulate how; what I now think is happening is that their brains must be attuned to the existence of a triad in the other person’s head.

Do I buy this model of business expertise? Yes, I do. *Nearly everything I know about business lines up with the model*. To my mind, this explains a handful of unrelated phenomena that I’ve long puzzled over. For instance:

- Why are fresh MBA grads unable to grok the implications of, say, cash flow in a given business, despite learning it in their courses? The answer is obvious: they have yet to internalise the *relationships* between a) market conditions and b) operational realities with c) the financial concepts they learnt in school. Every business is a system, and developing intuition for a system requires you to watch that system in action. Business students who have no real world experience of a business do not usually have such a mental model. DiBello writes, as part of a consulting engagement: *‘As a process check, we administered the same instrument to an equal of number of business-knowledgeable controls (BS or MBA business students who were not yet functioning managers or consultants and had not ever actually run a company of any size). Their performance was much lower. This, too, was an expected result.’* \[2\]
- How many different ways are there to build [Power](https://commoncog.com/7-powers-summary/) in business? I used to think that companies with better products would win, and then I believed that companies with better distribution would win, and then I ran a business for awhile and realised that businesses have won from differentiated product or differentiated distribution or differentiated access to capital; the possibilities were more varied than I had originally imagined. With the triad, it’s clearer to me that temporary competitive advantage may emerge from *any* of the three categories: you could gain a temporary edge from operational excellence (supply/leadership), or from exploiting some market opportunity (through product innovation or repositioning into a new category — this is demand/strategy), or even from exploiting some change in the capital environment that others in your industry do not realise exists (e.g. Malone with debt-funded, tax-sheltered expansion — this is capital/finance).
- Buffett’s quote “I am a better investor because I am a businessman, and a better businessman because I am an investor” makes more sense when seen through the lens of the triad — it's likely that exposure to equity investing or actual business operations would shore up one or more of the three categories. As a trivial example, a businessperson without a competent understanding of capital markets is missing one leg of the triad; an investor without a sufficiently realistic understanding of business operations (supply) is missing another.
- *Everything* in [An MBA for Business Operators](https://commoncog.com/an-mba-for-business-operators/) may be organised neatly into one of these three categories. In fact, if you look at Permanent Equity’s checklist through the lens of the triad, you’d realise that it is a mix of supply and capital considerations … but leaves out demand. This makes sense — to a private equity fund like Permanent Equity, the demand/strategy side of things is the bit that is most specific to the individual business, and cannot be generalised.
- Can management consultants or former VCs become good business operators? With the triad in mind, we can be more specific about what must happen: management consultants are likely better equipped in the demand leg of the triad, with some understanding of the finance leg; VCs are better equipped in the finance leg, and somewhat skilled in the demand leg. How well they do is how quickly they level up on the supply leg of the triad, and how well they internalise the relationships between all three.

To reiterate the most important point again: DiBello’s explication of the triad mental model tells us that it is the *relationships* between the three that we need to pay attention to; expertise in each leg of the triad is more commonplace when we rise through the stages of business ability.

## Cognitive Agility

DiBello also points out business expertise is made up of one other quality — something she calls *cognitive agility*:

> Given the volatility of markets and the complexity of team decision making in large organisations, we need not be as concerned with the general cognitive ability of today’s executive so much as his or her domain-specific expertise and cognitive agility, which is often associated with the kind of intuitive expertise developed through experience. **Cognitive agility is what allows executives to rapidly respond to changing situations and revise guiding mental models to meet performance demands**. *(emphasis added)* In sum, these insights require us to shift away from thinking of the cognitive ability of business leaders as fixed and static, and instead to focus on the way in which expertise evolves over time in response to dynamic environments. It also requires us to discard attempts to locate decision-making expertise as a fixed capacity that resides within the individual decision maker. \[2\]

> Building on theories of emergence, adaptation, and flexibility from complexity science and NDM, we use the term cognitive agility to refer to the extent to which an individual revises his or her evaluation of a situation in response to data indicating that conditions have changed. **The converse is cognitive rigidity, where the person is impervious to new data, being dominated by a rigid framework or paradigm that acts to filter out new, possibly relevant, information, creating blind spots**. \[2\]

When I read this section of DiBello’s 2010 paper, my eyes basically popped out of my head.

For several years now, I’ve tried to talk about the limits of frameworks, as experienced from the perspective of a business operator. I’ve noticed that when a market changes, or when you’re operating in a local environment, no framework can fully capture everything that you’re seeing as you brush against reality. I’ve attempted to capture this idea by writing blog posts such as [Reality Without Frameworks](https://commoncog.com/reality-without-frameworks/), and [How First Principle Thinking Fails](https://commoncog.com/how-first-principles-thinking-fails/) and [Good Synthesis is the Start of Good Sensemaking](https://commoncog.com/good-synthesis-adapting-to-uncertainty/). More importantly, I’ve attempted to articulate DiBello's notion of *cognitive rigidity*, with a trait I invented that I named ‘ [Dismissively Stubborn](https://commoncog.com/dismissively-stubborn/) ’. I wrote that piece to explain the type of people I’ve learnt to avoid in startup environments.

But it turns out that *DiBello had thought of this, and more, a full decade before I did*. I was also surprised — and delighted! — to learn that senior business leaders have thought about this particular aspect of business expertise:

> (We present) a case in which we used the Profiler to assess the senior management team of a medical device company. Our relationship with the company began when the chairman and acting CEO asked us to evaluate his senior staff for succession planning purposes. In particular, he wished to decide who, among the division presidents, would succeed him as the CEO. At the time, the company was growing very rapidly. During the four months we worked with the company between 2005 and 2006, we saw them grow from $300 million to $400 million. A pending acquisition was intended to increase the company’s size to $600 million by 2008.  
>   
> The CEO of the company was aware that **there is a qualitative difference between competent managers, who do very well in interviews or on personality tests, and experts, who can perform under challenging conditions**. He wanted to protect himself from (mere) competence at a time when his industry was in a high growth phase. **He was afraid that a favourable market might be masking a lack of skill—and cognitive agility in particular—among his senior staff.***(emphasis added).*

So now we must ask: how do DiBello and colleagues evaluate domain-specific expertise *and* cognitive agility?

This is probably a good time to talk about their Profiler.

In their 2010 paper, DiBello, Lehmann and Missildine laid out the requirements of a good instrument to assess business expertise:

> We believe that an instrument that effectively assesses business expertise must (1) be able to draw on the intuitive expertise of organisational members; (2) tap into the specific mental models individuals use to approach problems, rather than basic problem solving or generalised decision making skills; (3) locate individuals within the distributed cognition of an organisation, that is, specify the individual’s role in the organisation; (4) be able to identify strengths and blind spots of the organisation’s transactive memory system and, in so doing, highlight dimensions of expertise that contribute to high-level decision making; and lastly, (5) measure not only cognitive ability, but also cognitive agility, that is, the capacity to rapidly revise one’s mental model in the face of dynamic feedback.  
>   
> The Profiler requires the user (1) to examine and analyse the same material used by actual experts (e.g., annual reports, 10K’s, analyst’s reports, etc., from actual companies) to make business decisions, (2) to make predictions about the company with respect to a number of domains (e.g., revenues), and (3) to judge various aspects of the company (e.g., evaluate the management team).  
>   
> Specifically, the Profiler asks users to answer several questions about the company to predict how its finances will develop over the next five years. Users’ answers are evaluated in terms of Dreyfus’s [five-stage model of expertise](https://en.wikipedia.org/wiki/Dreyfus_model_of_skill_acquisition?ref=commoncog.com) to determine their business acumen and level of expertise within the domain of business strategy. The Profiler also helps us identify blind spots in users’ thinking; that is, we can discern areas where users seem to miss aspects critical to the company’s performance. Moreover, for each prediction or judgment users must indicate the specific materials they relied on, such as discrepancies in annual report statements or the general state of the company’s finances. This aspect of the Profiler taps into the heuristics that are guiding users’ decision making. \[2\]

How are users evaluated? DiBello et al note that the triad (the extracted mental model of business) isn’t directly addressed in the Profiler itself — even though it informs the design of the assessment:

> These three core areas are not directly addressed in the questions we ask in the Profiler. Rather, an individual’s answers are rated and compared to those of the ideal expert. The questions are based on tangible outcomes and concern the company’s performance. The questions ask users to evaluate their company with respect to its strategy, leadership, and finances. These are areas that senior executives would be expected to make valid predictions about and are good ways to reveal their underlying business expertise. For example, a question about finances would be: As the senior manager for the business, which aspect of financial performance most urgently requires your attention? This question would then be followed by a series of options, such as “cost of goods sold,” “fixed assets,” “receivables,” or “R&D.” To answer this question—that is, to know what is troubling or of concern in the total context of the company at that time—the user must understand operations, finance, and market trends. Answer options reflect levels of expertise, 1 through 5, corresponding to the five stages in the Dreyfus model. The “correct” answer is empirically determined. Because these are actual cases, the outcomes are known and the root causes of problems have been identified.  
>   
> In addition to these Dreyfus questions tapping their business expertise, users are asked to provide three predictions that refer to quantitative outcomes in the company’s performance. These questions can vary slightly, as the instrument is customised according to a company’s needs; however, they will ask about issues, such as profits and revenues, that are indicative of how well a user can synthesise information about the company to predict real-world outcomes. For example, we may ask: What is your prediction of the company’s profits for the next 12 months? Users then indicate their answers along a 5-point Likert scale whose points are labelled appropriately, for instance, in the preceding example, options may range from “down 20% or greater” to “up 20% or greater.”

After this initial set of questions, DiBello steps forward the simulation and presents the results of their first year of predictions. This is the primary way that they are able to assess cognitive agility:

> Users then go on to Year 2. After reading through the Year 2 materials, they are able to see whether or not their predictions were validated by the company’s performance. In other words, they receive feedback as to the accuracy of the previous year’s predictions. They repeat this exercise for five years of material, making predictions and getting feedback as to whether or not their predictions have been validated. **This iterative process taps into users’ cognitive agility, allowing us to assess the degree to which users are able to revise their thinking in the face of feedback about company performance and their own predictions. Cognitively agile individuals learn from their inaccurate predictions or judgments and improve as they go along**. *(emphasis added)*  
>   
> Furthermore, we can see persistent blind spots that may exist and occur consistently across all five years. Thus, the Profiler can provide windows into the acumen, blind spots, mental models, and agility with which users approach complex decision making for a real company. It is important to keep in mind that all of the data provided are from real companies. We do not determine the outcomes, as they are taken from actual annual reports and actual finances over a given period of time.  
>   
> In addition to evaluating users’ direct responses to the Dreyfus and Likert questions, we also ask users to point to sections of the material they reviewed (including statements in the annual reports and financial summaries) that informed their decision on a particular question. For example, the annual report might mention a recent acquisition. This may reflect a strategic turning point for the company. As such, after users respond to the question on strategy, they must indicate what sections of the material influenced their decision. Users are allowed to choose up to five reasons for their decision. In this case, users should be aware that the recent acquisition plays a major role in the company’s strategy going forward, even if this is not immediately obvious. As such, when users answer this particular strategy question, they should be pointing to the acquisition as a basis for their decision. This not only allows us to evaluate the expertise and accuracy of their responses, but also gives us a window into the kinds of information they use to approach the decision-making process. Furthermore, this aspect of the Profiler allows us to determine whether users are utilising information based on the three core areas listed above: supply, demand, and finance. Specifically, we can see whether users approach the questions in an expert manner by evaluating the degree to which they are paying attention to these core areas as they make decisions.

In her 2019 chapter in the *Oxford Handbook*, DiBello presents a second application of the Profiler, this time in service of a financial services company that was recovering from the repercussions of the 2007 financial crisis:

> The assessment used here was a version of our online business simulation in which each participant must solve a business mystery by predicting what will happen over the course of a business’s life in one-year intervals after being given a year’s worth of data at a time and judge the actions of the executives in the case.
> 
> We chose an actual case that was similar to this company’s but which encountered its challenges in a different time of history. In contrast to their company, the company in the case had not made the same mistakes. After removing all identifying information—including dates—we presented each person with five years of material, one year at a time, and asked them to examine the company’s data, decisions, plans, and declarations and predict the outcomes of the following year and judge the actions of the executives. We use a scoring scheme designed to evaluate the present or absence of our model of an ideal business expert. We also ask the participants to choose the clues in the material that drove their thinking.
> 
> Each clue is like DNA or fingerprints at a crime scene. An expert will not only see that it is significant, but what it signifies. Therefore, if the participants predict that the executives in the case do not see that a market downturn is coming and choose the clues that in fact are an early warning sign of that fall, we can see that they are attending to the right clues in the market portion of the model. Each question and each clue is coded using an association matrix that links back to the model. From this we can generate charts of a person’s mental model compared to an ideal expert, and a heat map showing by colour where good clues were identified and where there are blind spots (black) where clues were overlooked.
> 
> ![DiBello Profiler](https://commoncog.com/content/images/2021/07/profiler_dibello.jpg)
> 
> Figure 1 shows some material that a participant would examine. The question and clue selection screens are all online and involve easy clicks. The participant can stop mid- stream, save their work, and come back to it later. The idea is to replicate the actual decision process of busy business executives.
> 
> In real life, the banking executives were having a particularly difficult crisis as a team. They were having serious disagreements with their CEO about how to recover. Some were working behind the scenes to have him removed by the board of directors. A unique feature of our instrument is that it can show how a team is thinking. When we mapped the executive team’s results—as a team— on the dimensions of expertise critical to their recovery, we found that they were at risk for making the same mistake again due to a shared blind spot in the area of finance and risk management. The only person on the team who could mentally simulate the eventualities accurately was the CEO with whom they were violently disagreeing and whom they were trying to depose.
> 
> This study is interesting on two fronts. It shows how easily a team can develop a common theory of the crime over time, even when it is not working, perhaps explaining how whole companies can sometimes implode. But it also shows how complex business has become and how the insights of one person can be difficult to communicate. Fortunately, using these data, the executives’ advisors were able to convince the executive committee that—as a team—they needed to re-distribute responsibility and put those with the deepest insight into the capital risk environment on the front line during the recovery. As a committee, they could function as an expert but only if they knew where the different elements of their expertise resided.This company did recover, enjoying a doubled stock price in less than a year, but it was by developing a shared understanding of how they were going to divide up running the business and yet keep all the parts linked in the way they ran the committee meetings. In the end, they followed the insights of the one lone wolf in one area, abandoning their preferred notions of what would work while they came up to speed on what he was seeing.

I believe DiBello when she says that the Profiler is able to detect blindspots in an exec team, that it is able to evaluate cognitive agility, and that you may use it to assess the expertise of business decision makers. I’m just not sure how this may be used, if one isn’t able to afford DiBello’s services.

(In theory, a generic Profiler may be used, though you can’t expect to score very highly if you don’t have relevant expertise in that particular business or industry.)

As of today, DiBello’s methods — both Profiler and training interventions — have been used by over 7000 individuals; a description of her work reads:

> Studies of more than 7000 people at all levels exposed to DiBello’s methods indicate that learning was accelerated by several months in all cases. Other studies indicate that unprecedented performance improvements and significant competitive advantages were achieved by companies using DiBello’s methods. The method is now used globally in four continents in industries as diverse as mining, transportation, financial services, information technology (IT) implementation, manufacturing, pharmaceuticals as well as others \[4\]

And in her NDM podcast episode, she mentions, almost in passing (32:22):

> A lot of senior leaders will look only at the financial data, and they won’t look at the market data. And maybe the market data will tell them that they’re making the wrong decision. So the heat map will be blacked out there. And we can tell that they have a bias that prevented them from being successful. (…) When we’ve done it with some large companies, we usually do it with a control group — which is some people lower down in the organisation, outside the C-suite. And we sometimes find some very gifted people, who are in under-represented groups, who would normally not be promoted to high level positions. And they are promoted as a result.

Ultimately, the most useful thing that I got out of DiBello’s published work was *the fact that this research even exists* — that it has been applied and has generated results for many companies; that there is a shared mental model of business expertise that exists despite the differences of many business domains.

More importantly, the work that DiBello has done isbelievable by my [practical standards for truth](https://commoncog.com/the-hierarchy-of-practical-evidence/). She's applied her work in multiple company interventions across multiple industries over a two decade period, and many interventions have resulted in measurable million-dollar outcomes. A predictive model that works when tested against reality, by a person with skin in the game, over a multi-decade period, is pretty much the strongest result you can ask for in a messy domain like business.

At the end of her chapter in *The Oxford Handbook*, DiBello writes:

> **Businesses are more alike than different**, and scalable educational applications in the form of generic rehearsals may have the same expertise accelerating benefit as our labour-intensive and expensive face-to-face rehearsals. *(emphasis mine)*

If you’d had told me a few months ago that this was possible, and that business was ‘more alike than different’, I would have told you to stop joking. And yet here we are.

---

Read [Part 2: The Importance of Cognitive Agility →](https://commoncog.com/business-expertise-cognitive-agility/)

## Sources

1. [Lia DiBello, Expertise in Business, Chapter in *The Oxford Handbook of Expertise*.](https://drive.google.com/file/d/1hd4hRm7hRWPxbF_XqOxT6qso_-MdJnFR/view?usp=sharing&ref=commoncog.com)
2. [Lia DiBello, David Lehmann, Whit Missildine, How Do You Find an Expert? Chapter 17 in *Informed by Knowledge: Expert Performance in Complex Situations*](https://drive.google.com/file/d/1sKQemeuAKkS4f9ZU-VFuGRA2EPlWUC7W/view?usp=sharing&ref=commoncog.com).
3. [Lia DiBello interview on the NDM Podcast](https://www.listennotes.com/podcasts/naturalistic/episode-28-interview-with-ydLRG3pAaB0/?ref=commoncog.com).
4. [2017, An interview with Dr. Lia A. DiBello, president and director of research, Workplace Technologies Research, Inc, *Journal of Information Technology Case and Application Researc* h](https://drive.google.com/file/d/1hok_8gsVPhKz_IXNMbepf-wiGZ_HKD85/view?usp=sharing&ref=commoncog.com).
5. [2008, Lia DiBello, Whit Missildine, Information Technologies and Intuitive Expertise: a method for implementing complex organizational change among New York City Transit Authority's Bus Maintainers](https://drive.google.com/file/d/1hgJC8uA7YCEQNtUB-Q_ehgFJi2zd1K8D/view?usp=sharing&ref=commoncog.com).
6. [2009, Lia DiBello, Whit Missildine, Marie Struttman, Intuitive Expertise and Empowerment: The Long-Term Impact of Simulation Training on Changing Accountabilities in a Biotech Firm](https://drive.google.com/file/d/1Yi7jJOIzFzFg5rQLNpDIZMoun7apMw3l/view?usp=sharing&ref=commoncog.com).

### Download the PDF Copy

[Download the PDF copy of this post here →](https://drive.google.com/file/d/1f6XAO1Soij-cHHyVU_hc5UKuoFxDS1A7/view?usp=sharing)

Originally published , last updated .

This article is part of the [Expertise Acceleration](https://commoncog.com/expertise) topic cluster. Read [more from this topic here→](https://commoncog.com/expertise)

Previous Post

#### ← Exec Development is a Different Game

Next Post

[![Feature image for The Origin of the Tatas](https://commoncog.com/content/images/size/w300/2025/05/the_origin_of_tatas.jpg)](https://commoncog.com/origin-of-the-tatas/)[![Feature image for Cases from The Heart of Innovation](https://commoncog.com/content/images/size/w300/2025/05/cases_from_heart_innovation.jpg)](https://commoncog.com/cases-from-heart-of-innovation/)[![Feature image for The Heart of Innovation: Why Most Startups Fail](https://commoncog.com/content/images/size/w300/2025/05/heart_of_innovation_book.jpg)](https://commoncog.com/the-heart-of-innovation-why-startups-fail/)[![Feature image for Business with Tariffs; Business as Usual](https://commoncog.com/content/images/size/w300/2025/04/business_with_tariffs.jpg)](https://commoncog.com/business-with-tariffs-business-usual/)

---
title: "Tacit Expertise Extraction, Software Engineering Edition"
source: "https://commoncog.com/tacit-expertise-extraction-software-engineer/"
author:
  - "[[Cedric Chin]]"
published: 2024-02-19
created: 2025-05-30
description: "One software engineer's story of extracting expertise from the heads of 'a tiger team of senior software engineers'. Useful to anyone who might want to accelerate their own skills."
tags:
  - "clippings"
---
By

![Feature image for Tacit Expertise Extraction, Software Engineering Edition](https://commoncog.com/content/images/size/w600/2024/02/tacit_knowledge_software_engineering.jpg)

## Table of Contents

1. [Stephen Zerfas’s Experiences Using the Recognition Primed Decision Making Model](https://commoncog.com/tacit-expertise-extraction-software-engineer/#stephen-zerfas%E2%80%99s-experiences-using-the-recognition-primed-decision-making-model)
	1. [More about Stephen Zerfas](https://commoncog.com/tacit-expertise-extraction-software-engineer/#more-about-stephen-zerfas)
2. [Closing Observations](https://commoncog.com/tacit-expertise-extraction-software-engineer/#closing-observations)

## Fan of great business?

Join 8,000+ sharp investors and operators like yourself, and we'll send you a collection of Commoncog's best articles right away:

*This is Part 8 in a [series on tacit knowledge](https://commoncog.com/the-tacit-knowledge-series/). Read [Part 7](https://commoncog.com/tacit-mental-model-of-business/) here.*

It’s been some time since we last talked about tacit knowledge. A brief recap:

- Tacit knowledge is knowledge that [cannot be captured through words alone](https://commoncog.com/tacit-knowledge-is-a-real-thing/).
- Much of human expertise is tacit: if you’ve ever dealt with someone more skilled than you, asking them “how did you do that?” would often result in a “I don’t know, it just felt right!” response — which is *particularly frustrating to a novice*.
- There is a ~30 year old branch of psychology research that specialises in the study and extraction of tacit knowledge — specifically tacit mental models of expertise in the heads of real world practitioners. This branch of psych is known as [Naturalistic Decision Making](https://naturalisticdecisionmaking.org/?ref=commoncog.com) (NDM). This is, as you might expect, *very useful.*
- Most of NDM’s methods for extracting tacit knowledge are quite involved; over the course of the Tacit Knowledge Series, we covered [Applied Cognitive Task Analysis](https://commoncog.com/an-easier-method-for-extracting-tacit-knowledge/), which was designed to be used by domain practitioners (as opposed to trained expertise researchers). I even [tested it on John Cutler](https://commoncog.com/john-cutlers-product-expertise/), who has some pretty remarkable product org expertise.
- Finally, the most important idea — or at least, the most relevant to this essay — is that intuition *is not a black box.* Early in the series I described something called the ‘ [Recognition Primed Decision Making model](https://commoncog.com/how-to-learn-tacit-knowledge/#the-recognition-primed-decision-making-model) ’, or RPD, which tells us how expert intuition really works. This is useful, because it lets us unpack what expert intuition is, which means we can work out how to get it for ourselves.

RPD looks like this:

![](https://commoncog.com/content/images/2019/01/Paper.Commonplace.41.png)

Before you continue further, I highly recommend [reading the section on RPD](https://commoncog.com/how-to-learn-tacit-knowledge/#the-recognition-primed-decision-making-model), to get a feel for what it tells us about expert intuition.

I wrote, at the time:

> One of the most obvious things that falls out of the RPD model is that we can now use it to carry out cognitive task analysis. In other words, we actually have a shot at understanding what went through my senior programmer’s head.  
>   
> When interviewing an expert, NDM practitioners will always zero in on ‘tough cases’ that the expert has experienced, in order to examine the recognition-driven process in their heads. Specifically, CTA interviewers will establish a timeline of events of the case, and then ask questions to elicit:  
>   
> \- What cues did they notice first?  
> \- In that particular situation, what did they expect to happen next? (expectancies)  
> \- What priorities (goals) did they have?  
> \- What courses of actions immediately came to mind?  
>   
> These four by-products is how you get at the tacit knowledge of practitioners.

The problem? I didn’t *actually* test this in practice — or at least, not rigorously, not over the course of a couple of months, which is really the bare minimum if you want to get decent at something. Which means that I never really understood the method; I was stuck at the lowest level of ‘Put To Practice In Own Life’ on my [Hierarchy of Practical Evidence](https://commoncog.com/the-hierarchy-of-practical-evidence/).

It *also* meant that I couldn’t really tell you how to use the ideas, because I’d not grappled with all the nuances of practice.

Until now.

A few weeks ago, I learnt that [Stephen Zerfas](https://twitter.com/stephen_zerfas?ref=commoncog.com) had read Commoncog’s tacit knowledge series in 2020, and [had used the RPD model to accelerate his own skill](https://twitter.com/stephen_zerfas/status/1741550141574889698?ref=commoncog.com) when he was put on a ‘tiger team of software engineers, all 10 years senior to me at Lyft.’

I asked him some questions. These are his answers.

  

---

  

## Stephen Zerfas’s Experiences Using the Recognition Primed Decision Making Model

All answers by Stephen Zerfas (4 February 2024).

**Could you introduce yourself and describe the context you found yourself in when you started using this technique? Feel free to talk about your goals going into this process e.g. were you trying to get better at something specific?**

I’m a cofounder of Jhourney, a meditation–BCI startup developing tools to teach [life-changing meditation](https://www.jhourney.io/blog/what-are-the-jhanas?ref=commoncog.com) sometimes described as "MDMA without the drug” in record speed. Until recently, these states were believed to require hundreds of hours of practice or more. We believe biofeedback will make them accessible in under 10 hours. During R&D, we began seeing 65-90% of novices entering these states on weeklong retreats *without* tech. [Testimonials](https://www.youtube.com/watch?v=Q56JUENh9so&feature=youtu.be&ab_channel=AlexGruver&themeRefresh=1&ref=commoncog.com) are hyperbolic and NPS is 80%. If you’re curious, check out our [retreats](https://pages.jhourney.io/spring-retreats?ref=commoncog.com)! We’re scaling these retreats to fund R&D and will layer on tech over time. (RPD was useful to me even in this context, as you’ll soon see).

I was previously a [management consultant](https://www.economist.com/united-states/2022/01/01/bridgespan-group-the-most-powerful-consultants-youve-never-heard-of?ref=commoncog.com) and then used a coding bootcamp to become a software engineer at Lyft. Both experiences primed me to try CTA. While a consultant, I collected efficiency hacks to cut my hours by 40% and reclaim work-life balance. While teaching myself to code, I collected learning-how-to-learn tactics to try and get a job before money ran out. I wrote about some of these tactics under a pseudonym [here](https://www.freecodecamp.org/news/first-line-of-code-to-226k-job-offer-in-8-months/?ref=commoncog.com).

When I read Cedric’s Tacit Knowledge series in early 2020, I was looking for ways to learn from my software engineering colleagues faster. I had recently been promoted, inherited extra responsibility after layoffs, and started an after-hours computer science program. My colleagues all had over 10 years more experience than me. I couldn’t soak up their expertise fast enough!

The challenge was that they were often helpful in ways that didn’t make sense. I’d ask a question, they’d give an answer without explaining their thought process, and I’d be left with an isolated lesson and limited ability to make myself self-sufficient next time.

The response-primed decision model (RPD) was just what I needed. It helped me to probe my colleagues for their reasoning.

Later on, I used RPD to interview neuroscience experts and tutors while upskilling into an amateur neurotechnologist. RPD helped me code proof-of-concepts for Jhourney, and critically, decide when to bet *against* experts.

**What was your general approach in asking them questions?**

Prior to the RPD, I would more or less ask “why” repeatedly. This sometimes worked, but had several failure scenarios:

- My questions were ad hoc. I often thought of more comprehensive and systematic questions later.
- My question quality varied. I would often think of better questions later.
- My questions were re-invented every time. This made it harder to reflect on my ability to learn from colleagues like a virtuoso skill.
- My questions sometimes seemed irrelevant to others. Sometimes my colleagues misunderstood my questions as purely academic rather than serious attempts to upskill myself.

RPD gave me the framework and language to be comprehensive, consistent, and effective by asking about (1) cues, (2) expectations, and (3) alternative actions. (I discovered that in practice I often didn’t need to probe as much on priorities, the fourth part of RPD.)

It also helped me set expectations for the conversation. At one point I even sent a memo to my tech lead summarizing the RPD and linking Cedric’s article so that we’d have shared language for my questions in the future.

**Could you tell one story about asking those questions and improving as a result of it?**

My memory is dusty, but I recall hitting a stacktrace that looked like complete gibberish while trying to rebuild a Python package on an ARM instead of an AMD processor. Among many other things, it included two characters that seemed unrelated to me: “-v”.

I showed my colleague and asked explicitly about expectations and solutions. He said, “Oh, I bet they forgot to add a ‘-v’ command early in the project’s history and updated it recently, which created an inconsistency in the versions of software you happen to be running. It’s probably a quick one-line change for you. You’ll see where with the changelog.”

I thought that was a pretty magical idea to begin with, but the crazy thing was when I obediently pulled up the changelog and we found nothing. He responded with “That’s a mistake. Sometimes these projects have multiple changelogs and I bet they left it out of this one. See if you can find another.” I had never heard of multiple changelogs but sure enough, he was right, about both the seemingly secret second changelog, the addition of a “-v” command, and a one-line solution.

“How on Earth were you so confident in your interpretation to conclude the project’s changelog was wrong instead of your guess at the problem!?” I asked. (This was a probe on cues and expectations.) He shrugged it off and said “-v” was a universal symbol. I asked, “Did you imagine any alternatives and why did you rule them out?” He said not really in this case — there could’ve been something very idiosyncratic, but we were definitely going to hunt down the changelog first.

To a senior engineer, my struggle and questions might sound naive. But they helped me update that certain naming conventions were likely a lot more ubiquitous than I realized. I noted they could be used to cue and ideate solutions in the future.

**What did you find surprising about this process? What was easy? What was hard?**

I was surprised by how easy it was! Cedric says RPD can be abstract and difficult to apply, but I found it elegantly simple and concrete.

The hardest part may be that, like any abstract idea, RPD takes practice, reflection, and a little creativity to get good at. For example, I probably asked “What alternatives would you consider?” 50 times before realizing I should usually follow up with a few rounds of “What else?” until they run out of answers. Naval’s got a great podcast episode with Matt Mochary somewhere about how the final answers to “What else?” are often the best.

I wonder if RPD has a reputation for being difficult to apply because it still just takes reps and time. If tacit expertise is just a long list of prior experiences helpful for pattern matching, then the best RPD can do is help you transfer them faster.

There was one thing RPD unexpectedly gave me that I later decided was far more valuable than transferring expertise faster: *it gave me permission to disagree*.

How? Whether or not it’s wise to disagree with someone is often a function of relative expertise. If you’re a new student learning from a true expert, most of the time the most skilful way to resolve confusion is to *assume the expert is right*. You ask clarifying questions, think or memorize what they said, etc. But if you’re speaking with a peer, you want to work together to get to the best ideas. Finding the best idea *requires disagreement* – the scientific method uses falsification, not confirmation. You’re either trying to come up with alternatives or pitting ideas against each other, neither of which make sense if you assume someone is already correct.

Knowing when it was reasonable to disagree was essential when I left my software engineering job to start Jhourney. I was exploring whether or not to invest years of my life and half of my savings in attempting to do something nobody had done before. In order to do something so contrarian, I’d have to disagree with lots of people with more expertise than me. RPD helped me decide when it was wise or foolish to do so.

I spent about a year reading and talking to experts to decide if the company could work. When an expert was obviously pattern matching off all kinds of prior experience, I’d go full student mode. All my probing was motivating by making sure I could replicate their thinking and avoid mistakes. But if it was clear they had fallen back to sensemaking, then I knew my first-principle-reasoning was just as important as theirs in battling ideas. All my probing was to see if any of my favorite mental models or decision-making tools offered a better rationale than theirs, or to see which of my ideas they hadn’t explored before.

If I had simply deferred to others because they were experts, I don’t think Jhourney would’ve gotten off the ground. What a gift! The biggest regrets in life are often from inaction; I’m grateful ‘permission to disagree’ has allowed me to chase my dreams.

**If you were to talk to a junior engineer wanting to use your approach, what would you tell them to watch out for?**

One, RPD can leave you with a bunch of possibly isolated lessons, without theory or connecting frameworks. I’d suggest supplementing RPD with any tools that make isolated ideas more useful (e.g. structured reflection, spaced repetition, and DIY attempts). Your mileage will vary, but at times I have:

- Kept an end-of-workday reflection doc with 2-3 ideas each for what went well and what didn’t go well. This can be good for process improvements like getting better at RPD.
- Stored favorite technical lessons and motivating ideas into Anki cards (I get dopamine hits from Anki on my phone like I do Twitter). These can double as a searchable case library.

Two, if you want to learn as fast as possible, RPD isn’t as important as a sense of playful curiosity. I think conventional wisdom is to take your own curiosity for granted. But through a combination of meditation, reflection, and deliberate practice, your curiosity – both magnitude and direction – is shockingly malleable. The ultimate learning-how-to-learn hack is to discover how to get yourself as curious as humanly possible about something valuable to others, and then follow that like a dog chasing a ball.

Once you’ve managed your curiosity, Paul Graham [suggests](https://paulgraham.com/genius.html?ref=commoncog.com#:~:text=But%20there%20are%20some%20heuristics,consuming%20something%20someone%20else%20creates.) habitually and playfully *creating* things valuable to others, rather than just consuming content, as you learn. I love that point, and have since found creating, rather than consuming, can 10x my motivation, purpose, and feedback while learning.

Stephen Zerfas is the founder of Jhourney, a startup teaching people to meditate into ["life-changing" states](https://www.jhourney.io/blog/what-are-the-jhanas?ref=commoncog.com) sometimes described as "MDMA without the drug." Their first product is a modernized, week-long meditation retreat that's converting top tech executives (e.g. OpenAI, DeepMind) into angel investors and repeat attendees. They've also collected the largest biosensor dataset on these states in the world, and have plans to build tools like biofeedback to teach them in under 10 hours. You can see 90 seconds of some dramatic testimonials [here](https://www.youtube.com/watch?v=Q56JUENh9so&feature=youtu.be&ab_channel=AlexGruver&ref=commoncog.com). Here are links to Jhourney's [website](https://www.jhourney.io/?ref=commoncog.com), [podcast](https://jhourney.transistor.fm/?ref=commoncog.com), [Stephen’s Twitter](https://twitter.com/stephen_zerfas?ref=commoncog.com), and [upcoming retreats](https://pages.jhourney.io/spring-retreats?ref=commoncog.com). **Code COMMONCOG will get $100 off.**

  

---

  

## Closing Observations

There are three things that leapt out to me from Stephen’s answers:

1. The trick of asking for alternatives until the expert ran out is *smart* — and also a perfect example for why *putting these ideas to practice* [actually matters](https://commoncog.com/believability-in-practice/). There is no way you can get to observations like these by just reading alone.
2. That Stephen didn’t find it that important to ask for a prioritised list of goals! (Perhaps there are other ways at getting at those priorities? But I don’t currently have access to a practice domain here, so I can’t say, and will defer to Stephen).
3. And finally, the fact that Stephen feels that as good as RPD is, this crystallised knowledge, plucked from the heads of experts around you, is no replacement for some coherent theory of the domain.

I have more questions, but I am limited by my practice domain right now. I hope Stephen’s answers will give you more ideas for future experimentation — and that you’ll send them to me when you find out! Speaking for myself, I’ve got a couple more ideas to try out now, and I can’t wait to be put into a context where I can use it to see for myself.

Till then. Godspeed.

Originally published , last updated .

This article is part of the [Expertise Acceleration](https://commoncog.com/expertise) topic cluster. Read [more from this topic here→](https://commoncog.com/expertise)

Previous Post

#### ← One Definition of Wisdom

Next Post

[![Feature image for One Definition of Wisdom](https://commoncog.com/content/images/size/w300/2024/02/charlie_munger_wisdom.jpg)](https://commoncog.com/one-definition-of-wisdom/)[![Feature image for Mental Strength in Judo, Mental Strength in Life](https://commoncog.com/content/images/size/w300/2023/05/mental_strength_judo_life-1.jpg)](https://commoncog.com/mental-strength-judo-life/)[![Feature image for Creating New Drills for Deliberate Practice](https://commoncog.com/content/images/size/w300/2023/04/deliberate_practice_create_new_drills.jpg)](https://commoncog.com/creating-drills-deliberate-practice/)[![Feature image for An Expertise Acceleration Experiment in Judo](https://commoncog.com/content/images/size/w300/2023/02/expertise_acceleration_judo.jpg)](https://commoncog.com/expertise-acceleration-experiment-judo/)

---
title: "John Cutler’s Product Org Expertise"
source: "https://commoncog.com/john-cutlers-product-expertise/"
author:
  - "[[Cedric Chin]]"
published: 2021-08-10
created: 2025-05-30
description: "A few weeks ago, I helped Amplitude head of product education John Cutler extract tacit expertise around diagnosing and improving product organisations. Here's how that went."
tags:
  - "clippings"
---
![Feature image for John Cutler’s Product Org Expertise](https://commoncog.com/content/images/size/w600/2022/07/john_cutler_product_org_expertise.jpg)

## Table of Contents

1. [A Quick Recap of ACTA](https://commoncog.com/john-cutlers-product-expertise/#a-quick-recap-of-acta)
2. [What We Extracted](https://commoncog.com/john-cutlers-product-expertise/#what-we-extracted)
	1. [The Knowledge Audit](https://commoncog.com/john-cutlers-product-expertise/#the-knowledge-audit)
3. [What Was Difficult?](https://commoncog.com/john-cutlers-product-expertise/#what-was-difficult)
4. [Wrapping Up](https://commoncog.com/john-cutlers-product-expertise/#wrapping-up)

[John Cutler](https://twitter.com/johncutlefish/?ref=commoncog.com) is currently the head of product education at [Amplitude](https://amplitude.com/?ref=commoncog.com). An important part of his job is to act as a consultant to product teams that use Amplitude; over the course of his entire career, he’s seen and helped *hundreds* of product organisations.

Cutler’s superpower is that he is able to talk to a set of product people and — within 8-20 minutes — figure out the system dynamics of their organisation, and then suggest a bunch of experiments to make that system better. His interventions usually take on the form of a prioritised list of experimental changes or team interventions. And he can do this for pretty much any product team in pretty much any type of company.

Of course, this isn't to say that John gets it right 100% of the time — experts *are* fallible, after all. But it is a pretty remarkable ability — he points out that often, product leaders assume that their best practices scale universally; they don’t realise that different teams in different industries in different cultures around the world have vastly different product practices, all of which might work for them, depending on a host of factors. Cutler doesn’t make such mistakes — or at least he doesn’t make as much of them any more; he’s seen too many possible product org configurations to count.

Sometime in the last year, he attempted to write a book about his skills. And then he came up against a wall. Like most experts, Cutler’s expertise is intuitive and context-dependent. Give him a product team and he's able to dig into their issues; ask him to explicate what he does into a simple framework, and he struggles.

Two months ago, we starting talking on Twitter. I told John that there was a whole set of techniques designed to extract [tacit mental models of expertise](https://commoncog.com/the-tacit-knowledge-series/), and that he was welcome to give them a try. I said that the simplest technique was something called ACTA, or [Applied Cognitive Task Analysis](https://commoncog.com/an-easier-method-for-extracting-tacit-knowledge/), designed by Laura Militello and Robert Hutton in 1998 for practitioners (as opposed to trained researchers). I suggested that I could give it a go with him.

John said yes.

## A Quick Recap of ACTA

ACTA is a protocol of three interview methods and a presentation format, designed to help a practitioner extract information about the cognitive demands and skills required for a task. The three interview methods are:

1. You first create a **task diagram**. This is a broad overview of the task in question, and identifies the difficult cognitive elements.
2. You do a **knowledge audit** — which identifies all the ways in which expertise is used in a domain. You also extract examples based on actual experience.
3. You do a **simulation interview** — you give the expert a simulation of a single incident, and then probe the expert’s cognitive processes over the course of that simulation.

Finally, you present the extracted information in a **cognitive demands table**.

Due to my lack of experience using the technique, I only performed the first two tasks with John. I didn’t know how to create a simulation for us to examine together — and in fact I said that this was the most tricky bit of ACTA [in my original post](https://commoncog.com/an-easier-method-for-extracting-tacit-knowledge/#simulation-interview) on the technique. In retrospect, perhaps I should have asked John for a recording of a product team engagement (though this might violate privacy or ethical rules), or arranged for a call with an *actual* product team, who was willing to be subject to an ACTA study.

I’ll give this more thought the next time I run ACTA on a willing expert.

## What We Extracted

Cutler’s product calls consist of the following four stages:

1. **Introduction** — John opens up the call and gets some pleasantries out of the way. Usually these teams come to him with some problem in mind, so he lets them start off with their issues.
2. **Problem Detection** — John pattern matches by asking questions, letting his intuition guide where to go and what to dive deep into. Sometimes he uses humour to open up the conversation. In many cases John will do a first pass of anti-patterns — that is, problems that are really common — and then let the conversation develop from there. This part takes anywhere between 8-20 minutes. If by the 20 minute mark he isn’t able to map out a few problems, this is likely a rare edge case — a unique organisation that he can't help.
3. **Solution Delivery** — Based on the problems unearthed in Step 2, John will suggest a prioritised list of experiments to try.
4. **Wrap up and follow up** — There may or may not be a follow-up, depending on the nature of the engagement.
![](https://commoncog.com/content/images/2021/08/product_task_diagram.png)

We identified Steps 2 and 3 as the parts with the highest cognitive demands:

- **Step 2, Problem Detection** — A novice would not have John’s intuition on what to dig deeper into, and what to ignore. John is able to respond to the *gestalt* of the person or team he’s talking to — “the best product teams in the world — there’s a way these people talk about their risks … it’s just a much more structured deliberate way of describing it versus a more kind of frantic way of describing it.” John is also able to get into the heart of the problem more efficiently than a novice would.
- **Step 3, Solution Delivery** — Some product team problems are easy, or ‘acute’, and these tend to be simpler to diagnose and then fix. A novice could be taught what recommendations to deliver here. But some product team problems are ‘chronic’, or more accurately described as complex systems problems, where multiple things are bad all at once (e.g. bad processes, *and* no decision power, *and* technical debt, *and* bad org infrastructure, *and* a toxic leader, etc). In such cases, a novice would not be able to prioritise a list of possible interventions the way John would be able to.

Of the two steps, we decided to focus on Step 2, Problem Detection. This was partly due to time constraints, but also because it seemed more in line with John’s goals for his book (as I understood it, he intended it to be a book on how to do what he does when he’s diagnosing the problems of product teams — and the problem detection step is what seems to be the most mysterious part of his expertise).

### The Knowledge Audit

ACTA’s [knowledge audit](https://commoncog.com/an-easier-method-for-extracting-tacit-knowledge/#basic-probes) breaks down expertise using six-to-eight different probes.

We ended up doing five probes, due to the nature of John's expertise:

| Aspects of Expertise | Cues and Strategies | Why Difficult? |
| --- | --- | --- |
| **Past & Future**   Able to quickly tell how the org got to its current state, and predict what problems it might be facing going forward. | **John's first pass is always to detect anti-patterns.** “Bad product orgs all kinda look the same, good orgs can succeed for very different reasons.” Then, work backwards from the anti-patterns to figure out how the org got here.      **First detect if there is high Work In Progress (WIP).** Usually this presents itself with symptoms or complaints like ‘low morale’, or ‘no growth mindset’, or ‘no learning’. John uses these complaints to figure out if the team has high WIP — which is nearly always a sign of a problematic product org. If high WIP exists, then he works backwards to figure out how the org got there. (60% of the time, John says, a bad org [has high WIP](https://medium.com/hackernoon/wip-it-real-good-66aa710178fd).)      **The other 40% of the time, John is able to quickly suss out other possible contributors to those problems** (see Big Picture below) | A novice may be able to tell if there is high WIP, but there are many ways that orgs may end up with high WIP, and diagnosing how it got there is difficult.      Also difficult because ‘bad’ practices might not be really bad, and must be evaluated within the context of the org. |
| **Big Picture**   Able to rapidly construct a gestalt that includes a number of factors, like: decision authority, decision power and structure in the org, org culture and demographic (country) culture, product org structure and workflow, org ability to learn, existence of product strategy and business model. | **Decision Authority**   — Who really holds the power in the org? What is the shape of the decision authority, and how many stakeholders are involved when making a product decision?   \- John will ask further questions to clarify if a product owner really has final authority over product decisions.   \- Related: where does this org lie on the extremes of product ownership? One extreme: product team owns $20m worth of business. Other extreme: product team has to have 5 meetings and 3 stakeholder sign-offs for a new ticket to be created in Jira.      **Organisational learning ability** — check if product leader/team is comfortable with uncertainty, and can talk about current hypotheses and experiments in play.   \- “tell me your last 5 big product decisions” — speed at which the product leader is able to answer gives John information about a) how quickly does the org ship/learn/decide? b) who has decision authority, c) how they think. If they have to pause to remember, likely they have not shipped recently, or perhaps person is not very in tune with the org, and so on.   \- Know the extremes of structured learning; e.g. on one extreme, there is mob programming, where nobody works on their own projects, extreme collaboration. Other end: everyone owns a specific project. Both exist and can work, in different industries.   \- Does the product team or the software engineers have direct customer contact? Is yes, or no ("why would we want to do that?!")r, this tells John something about the org.      **Seriousness of product engagement** — If the company wants to use Amplitude or other analytics tool, John checks if this is a mainline activity or a vanity project, instead of being impressed by the absolute project budget size.   \- Conversely, if business say they have an analytics problem, but they’re making a lot of money / are in a monopoly position in the market, they don’t really have a problem.      **Business model** — John will check how the business makes money, in order to judge how important the product team is to the business.   \- John is able to predict how a business model change will impact the product/eng teams, e.g. a business model change away from a high SEO, high traffic web product would mean changing the experimentation framework, doing less A/B tests, relying more on qualitative data, which means preparing the teams to switch to this new mode of learning and operating.   \- An expert will know and appreciate the role of luck in finding and executing a successful product strategy.   \- An expert will understand how a business model mismatch might affect org structure and impact various teams throughout the company.      **Risk taking** — If a company and product team is doing extremely well, John will know to ask ‘how often do your experiments fail?’. Not enough experiments that fail = company isn’t taking enough risks. An expert knows that companies have good times and bad times, and not taking enough risks may bite the product team when the company goes through a bad time.      **Org culture** — Is the company centralised, where HQ sets the culture and local teams don’t get a say, or is HQ more hands off on culture and local teams can set their own culture? This is a trade-off that experts can quickly identify.   \- An expert will also be able to gauge if the org culture can produce psych safety, even though there are many different types of org cultures (e.g. company is toxic except for product team)   \- Expert also has large sample of org cultures that exist out there (e.g. some companies you can move freely between teams, some companies you cannot).      **Demographic / country culture** — John matches the org culture with what he knows of the country’s culture (e.g. more individualistic vs more communitarian).      **Product strategy** — Check, in the following order: is there a strategy or not? -> is it halfway plausible or not? -> is it implicit or explicit? -> Is the structure of the company aligned around the strategy of the company/product?   \- An expert will know when to dig in to tell if the product strategy is plausible. It should be concrete, and centered on real world knowledge, and not fuzzy. | **Decision authority:** A novice would take the person’s title at face value, and not know to dig further to get the full picture of decision rights/product ownership. e.g. sometimes 'head of product' has no real power!   \- Shape of decision authority/org structure: Novice might not know all the variations of possible org structures, and many org structures can actually work — you just have to know what the right questions to ask for each. e.g. hard to have extreme product ownership in healthcare — there are many stakeholders — and this is ok.      **Organisational learning ability:** Novice would not pick up on cues that the product leader has ‘coherence’, and is comfortable with uncertainty. Novice will be distracted by buzzwords, confidence, or vision of the product leader.   \- Novice may assume that the way they interact with customers in their previous or current company applies to everyone, e.g. if in their previous company they implemented whatever the customer wants, they might not realise that some product companies do not build what customer wants, but instead seek to understand them, and build product to improve their world, but not necessarily build whatever they ask for.      **Seriousness of product engagement:** Novice would be impressed by absolute budget size for analytics or Amplitude engagement and use it as a measure of how likely the project will succeed.      **Business model:** Novice might not know to check the business model when diagnosing a product team (business model affects the product team’s structure and importance).   \- Novice might know all the names of business models, but not work out all the implications of business model change on the product team.   \- Novice might underestimate what it actually takes to do a business model shift mid-stream in a large company.      **Risk taking:** Novice might see a successful company and not think about their risk-taking levels, and how this might affect them going forward.      **Org Culture:** Novices might not know all the possible configurations of org culture that can work. (And there are many e.g. maybe the company is toxic, but the product team is safe haven, etc.)      **Demographic/country culture:** Novice would not have enough samples or experiences with different country cultures to know how the differences might affect team dynamics.      **Product Strategy:** Novice might not know to check the product strategy at all levels as per John’s flow chart.   \- A novice might miss the cues to dig in deeper and crisp up on the details of the product strategy. |
| **Noticing**   Able to pick up on attitudes and worldviews held by good product teams. Able to place them on a spectrum of org health, based on a large n of observed orgs. | **Notice the team’s comfort with uncertainty** — do they have a structured approach to the known unknowns? (John calls this ‘coherence’). A good product team has productive things to say about what is unknown ("we know this feature would do well, but we're not sure about that feature, but we have experiments going on for it"); a bad product team talks in generalities about the future.      **Notice when the company has ‘flow’.** Flow is a word John uses to describe when the company is executing and learning smoothly, and they are rapidly moving through the execution cycle, yet are agile and can change directions in response to new information. (Related to the ‘walk me through five of your last big product decisions’ question)   \- This is a complex thing to notice, John can quickly identify it. He breaks it down some of the cues he notices as “do they have high decision quality, high decision flow; do they have available and excessive data, do they have clean data, do they have the ability to interpret the data, the ability to link cause and effect, do they have enough data, do they have diverse perspectives, do they have the information they need to operate, are they able to make sense of it and align on it, and build coherent shared understanding, can they harmonise decisions and missions and can they find a compelling, overarching story or strategy.”      **Able to notice when a product leader is at the stage where they are naive** — if they are enamoured by product frameworks but have yet to put them to practice, this is usually a signal that they are still novices. “Good product people put all these things to practice (from the frameworks), but *they don’t talk about the frameworks*. It’s just something they do”.      **Quickly places the team on a spectrum, because John knows the extremes of each factor, and the contexts where they may or may not work.** E.g. There are extremely centralised product teams and extremely decentralised product teams, and both can work. This also applies to culture, and decision authority, and so on. | **Comfort with uncertainty:** a novice might not pick up on cues, because it isn’t about *what* a person says, it is *how* they say it, and what that implies about how they think.      **Flow:** Novices would not be able to pick up on all the cues that indicate if a team has ‘flow’.      **Detecting naive product leaders:** Sometimes naive product leaders will sound thoughtful and insightful — they will quote many frameworks. They are also likely to believe that they ‘just’ have to sell it to their org and put it to practice and everything would work. In practice it’s more difficult; good product teams typically do not talk about such frameworks even though they live it. A novice would have difficulty knowing the difference between the two.      **Extremes of Big Picture factors:** Novices would not know where on a spectrum a given product team would lie. |
| **Job Smarts**   John is able to ask the minimum set of questions that generate the maximum set of insights. | **Able to ask ‘Powerful Questions’ to cut through the mess.** Powerful questions is John's name for questions that are: neutral, have no right or wrong answers, do not pass judgment, but ‘crack the problem at the biggest spot to understand what is going on in the org’, and usually ‘nail a tradeoff or polarity between two opposing forces’.   \- Example of a weak question: ‘please tell me about the M&A that you do’, vs a powerful reframing: ‘given your pursuit of M&A at this particular moment, talk to me about the centralisation and decentralisation of your services to support that.’ and ‘it seems your company is really good at logistics, but what did it give up to get there?’ and ‘tell me about the last five consequential product decisions that you did’ (inability to tell John what these are, off the top of their heads, tells John something about the structure and velocity of the product team).   \- Powerful questions will provoke a thoughtful answer from a senior person who is a good systems thinker. Junior people will give ‘softer’ answers, that are less multi-faceted or thorough.      **Use humour to get a reaction.** Humour opens up the conversation and allows John to ‘cut through the bullshit and get at the heart of the matter’.   \- e.g. John jokes about quarters being weird, and when they react to that, it gives him an in to see how they feel about quarterly planning cycles, and if there are problems there e.g. if they keep changing direction every quarter.      **Show visual guides to provoke reactions and test for systems thinking ability.** If they do not respond well to a concept map or flow chart, it is likely they are not a good systems thinker. (e.g. the [WIP it good flow chart](https://medium.com/hackernoon/wip-it-real-good-66aa710178fd)). | **Novice would not know how to use ‘powerful’ questions**, because it requires some knowledge of the factors that are in tension with each other and are at play.      **Novice would not be able to use humour** (or other tricks) to get a response, in order to gauge that response.      **Novices would not know the range of responses to visuals** to identify if the person is a good systems-level thinker or not. Experts like John can predict what a product leader would say — he showed me a list of *all* possible responses to Marty Cagan’s book. |
| **Self-Monitoring**   John knows the weaknesses of his approach, and is able to tell when he is unable to help the org | **Tends to be biased towards not blaming a specific person.** John’s approach focuses on the system of the org, he’s more reluctant to pin the blame on certain people within the system (when there certainly are cases where the person is bad and should be removed). He is aware of this and watches out for it.      **Able to tell when he is unable to help the org** — example of an extreme case: if the org has a tiny market (~100 customers total!), and the particular product really isn’t that important to the success of the company. Took ~20 minutes to diagnose this. | It’s difficult to judge whether the system is failing the person (by setting them up to fail, incentives/structures not enabling them to do good work) vs the person being the problem. Whereas John would have seen enough working product systems to make a more calibrated call.      When John is not able to help an org, it’s usually an extreme edge case. The novice might not be able to tell when it’s an edge case. |

## What Was Difficult?

The biggest difficulty with using ACTA was in deciding when to dive deep and let John go on about his experiences, and when to move on to the next set of probes. In our initial call, I let John go on too long while setting up the Task Diagram, and then we ran out of time, so I didn’t do a good job of the Knowledge Audit.

The second call went much better.

I think the biggest mistake that I did was that I grossly underestimated the amount of time it would take to do the whole skill extraction exercise. I scheduled an hour for our first call, whereas a more appropriate length might’ve been two hours. In the end, we did two separate calls together, the first being 1.5 hours long, and the second two hours long. It’s hard to say if these times are normal, but I wouldn’t go below two hours in the future.

The other problem I had with the process was that I was unsure if I was producing useful information. John seemed happy with our progress, but I wasn't so sure.

After our first call, but before our second, I had the good fortune of talking to Laura Militello and Brian Moon, hosts of the [NDM podcast](https://naturalisticdecisionmaking.org/podcasts/?ref=commoncog.com). Militello, of course, was one of the two creators of ACTA, and when I asked her if she had any tips for using it, she said to keep the objective of the interview at the back of my head.

“Are you doing this for a training program? Or are you designing some software?”

“This is for his book.” I said.

“Then you should ask the questions with the book in mind.”

This was pretty good advice — in my second call with John, I dropped certain probes, because it was clear that he wasn’t going to be able to put it into written form. I told him that this would be a slightly different interview if he were designing a training program for the novices in his team, which wasn't something that was apparent to me before I’d tried the technique. John agreed that that was a pretty interesting nuance.

My only regret is that I didn’t get to do the Simulation Interview, which I think prevented me from creating a Cognitive Demands table. But I’m still rather happy for a first try — we both got a lot out of the Knowledge Audit alone.

## Wrapping Up

Overall, I’m rather pleased with how this ACTA experiment turned out. If anything, it tells me that Militello and Hutton achieved their stated goals with the technique: ACTA was designed for practitioners in the field, and — as a practitioner in the field — I found the learning curve only a little difficult to climb.

I could also tell that John was pleased with the results of our exercise. Shortly after our calls, John tweeted:

This was something I’d heard directly from him. Near the end of our second session, he told me that he’d always found it difficult to articulate a clear framework for his product ability — which was vastly more nuanced than what he saw published in books and blog posts on the web. But of course he wasn’t able to describe it — experts are often unable to tell you how it is they do what they do. I’ve noticed that John’s Twitter bio reads “I like the beautiful mess of product development” — and his intuitive expertise is exactly that: a beautiful, effective, but mostly tacit ‘mess’.

I'm pretty pleased that I got to help him explicate some parts of his expertise. God knows how long he took to build it.

I’m looking for other opportunities to put ACTA to practice. I have a couple of candidates in mind — the most important of which is my former technical lead (who has good programming taste), and whose code taste was what [started me down this path in the first place](https://commoncog.com/follow-your-nose/#the-tacit-knowledge-question). For what it’s worth, I’ve signed up and paid for the Self Study course offered by the [CTA Institute](https://cta.institute/?ref=commoncog.com) — which is a training organisation created by Militello, Hutton and Moon. I’m hoping that I’ll get some pointers on simulation design by the end of it.

If you’ve read to the end of this piece, I’d like to encourage you to give ACTA a try. Go find someone who’s willing to spend some time with you, who has some expertise that you might want to explicate for yourself (or even for them!) The method isn't that difficult to do. More importantly, you might surprise your subject by uncovering elements of their expertise that they didn’t already know.

To be honest, I've found that last bit to be the most rewarding part of putting ACTA to practice. The look of delight on John's face at the end of our two calls was pretty damned awesome. Godspeed, and good luck.

Originally published 11 August 2021, last updated 19 July 2022.

[![Feature image for Tacit Expertise Extraction, Software Engineering Edition](https://commoncog.com/content/images/size/w300/2024/02/tacit_knowledge_software_engineering.jpg)](https://commoncog.com/tacit-expertise-extraction-software-engineer/)[![Feature image for One Definition of Wisdom](https://commoncog.com/content/images/size/w300/2024/02/charlie_munger_wisdom.jpg)](https://commoncog.com/one-definition-of-wisdom/)[![Feature image for Mental Strength in Judo, Mental Strength in Life](https://commoncog.com/content/images/size/w300/2023/05/mental_strength_judo_life-1.jpg)](https://commoncog.com/mental-strength-judo-life/)[![Feature image for Creating New Drills for Deliberate Practice](https://commoncog.com/content/images/size/w300/2023/04/deliberate_practice_create_new_drills.jpg)](https://commoncog.com/creating-drills-deliberate-practice/)

