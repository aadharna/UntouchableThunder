# UntouchableThunder
Competitive Co-evolution of agents and environments (via mechanics) in GVG-AI  

Master's Thesis -- Aaron Dharna  

PIs:  

	- Dr. Julian Togelius (NYU-Tandon)
	- Dr. Damian Lyons (Fordham University)

----  

 
Co-evolution of agents and environments such that agents maximize the env, while environments adapt to minimize the agents (while maintaining playability).  

Basic idea: I was thinking about competitive co-evolution several weeks ago (before I knew that was the term), and thought to myself, "wouldn't it be neat if on one-hand we had an agent, A,  learning to play a level (I was thinking standard RL here), but then we also had an adversarial agent, B, who was editing the levels that the first agent was trying to learn. Would this help with generalizing what A was learning?  

I then came across [UberEngineering's POET](https://eng.uber.com/poet-open-ended-deep-learning/) -- Paired Open Ended Trailblazer, which does exactly that but with genetic algorithms for both agents. From POET, this type of systems: Endlessly Generates Increasingly Complex and Diverse Learning Environments and their Solutions! The result is something like **Curriculum Learning**.

When I was talking with Dr. Togelius he suggested that I re-implement POET in [GVG-AI](https://arxiv.org/pdf/1802.10363.pdf), and suggested that a useful expansion of the work would be if we were editing game-rules rather than level-topology, which I completely agree with.  

Initial scope:  

    - Use some sort of Evolutionary algorithm to generate new levels  
		- Neuro-Evolution? (e.g. NEAT)
		- Evolving the expansion term of MCTS?
		- GAN? 
		- See: [Search-Based Procedural Content Generation](https://course.ccs.neu.edu/cs5150f14/readings/togelius_sbpcg.pdf)
	- Use either an EA or RL to have the agent learn the generated (paired) level.
		- More likely an EA, I am just more familiar with RL
	
Things to consider:  

	- [Quality Diversity](https://arxiv.org/pdf/1907.04053.pdf) methods
	
Framework: 

	- Use GVG-AI (generalized video game - ai) to set up levels.  

POET alters the topology of the levels to instantiate curriculum learning, a complexification system of game-rules that makes the level/environment progressively more complex  

	- new rules
	- new items
	- etc

as the agent gets better is the direction that I would like to go in. 
