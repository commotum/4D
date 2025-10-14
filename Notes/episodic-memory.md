> **DWARKESH PATEL:**  
> We measure the performance of these large language models on MMLU and other benchmarks. What is missing from the benchmarks we use currently? What aspect of human cognition do they not measure adequately?
>
> **SHANE LEGG** *(Cofounder and Chief AGI Scientist at Google DeepMind)*:  
> They don’t measure things like understanding streaming video. These are language models. They don’t have things like episodic memory. Humans have a working memory, for things that have happened quite recently, and then we have a cortical memory: things being stored in our cortex. But there’s also a system in between: episodic memory, in the hippocampus. It’s for learning specific things very rapidly. So if you remember some of the things I say to you tomorrow, that’ll be your episodic memory.
>
> Our models don’t really have that kind of thing, so we don’t really test for it. We just try to make the context windows, which is more like working memory, longer and longer to compensate. It’s a difficult question because the generality of human intelligence is very broad. You have to go into the weeds of trying to find out if there are specific types of things that are missing from existing benchmarks or different categories of benchmarks that don’t currently exist.
>
> **DWARKESH PATEL:**  
> Would it be fair to call episodic memory the root of human sample efficiency, or is that a different thing?
>
> **SHANE LEGG:**  
> It’s very much related to sample efficiency. It’s one of the things that enables humans to be very sample efficient. Large language models have a certain kind of sample efficiency, because when something is in their context window, that biases the distribution to behave in a different way. That’s a very rapid kind of learning. There are multiple kinds of learning, and the existing systems have some of them but not others. It’s a little bit complicated.
>
> **DWARKESH PATEL:**  
> Is it a fatal flaw of deep learning models that it takes them trillions of tokens to learn, or is this something that will be solved over time?
>
> **SHANE LEGG:**  
> The models can learn things immediately when they’re in the context window. Then they have this longer process, where you actually train the base model. That’s when they’re learning over trillions of tokens. What I’m getting at is that they’re missing something in the middle.
>
> I don’t think it’s a fundamental limitation. What’s happened with large language models is that something fundamental has changed. We know how to build models that have some degree of understanding of what’s going on. That did not exist in the past. We’ve got a scalable way to do this now, which unlocks lots and lots of new things. We can look at things that are missing, such as this episodic memory type thing, and we can start to imagine ways to address that.
>
> My feeling is that there are relatively clear paths forward now to address most of the shortcomings we see in the existing models. Whether it’s about delusions, factuality, the type of memory and learning they have, understanding video, all sorts of things like that, I don’t see big walls in front of us. I just see that with more research and work, these things will improve and probably be adequately solved.
>
> — *from* **The Scaling Era: An Oral History of AI, 2019–2025**, Dwarkesh Patel with Gavin Leech, pp. 65–67.

@book{patel2025scalingera,
  title     = {The Scaling Era: An Oral History of AI, 2019--2025},
  author    = {Patel, Dwarkesh and Leech, Gavin},
  year      = {2025},
  publisher = {Self-published / Dwarkesh Patel},
  pages     = {65--67},
  note      = {Interview with Shane Legg, Cofounder and Chief AGI Scientist at Google DeepMind}
}
