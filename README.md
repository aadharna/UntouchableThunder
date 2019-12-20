# UntouchableThunder
Competitive Co-evolution of agents and environments (via mechanics) in [GVG-AI](https://arxiv.org/pdf/1802.10363.pdf). 

Master's Thesis -- Aaron Dharna  

Advisers:   

	- Dr. Julian Togelius (NYU-Tandon)
	- Dr. Damian Lyons (Fordham University)

See tl;dr at bottom.

----  
Installing:  

Use the env.yml file to create a conda env.  
    - `conda env create -f env.yml`  
    - NOTE: this yml file is INCOMPLETE. You will need to also run:  
        - cd to your equivilent of: `~/.conda/envs/thesis/lib/python3.7/site-packages/`  
	- `git clone git@github.com:Bam4d/GVGAI_GYM.git`  
	- `cd GVGAI/python/gvgai`    
	- `pip install -e .`   
	- `cd ../../..`  
        - `git clone git@github.com:omardelarosa/Adaptive-DEs.git`  
        - `cd Adaptive-DEs`  
        - `pip install -e .`  
        - `cd ..`  
        - `git clone git@github.com:hardmaru/estool.git`  
        - `cd estool`  
        - `pip install -e .`  

----    
 
The main phenomenon under investigation here is the co-evolution of agents and environments such that agents maximize the env, while environments adapt to minimize the agents (while maintaining playability).  

Basic idea: I was thinking about competitive co-evolution several weeks ago (before I knew that was the term), and thought to myself, "wouldn't it be neat if on one-hand we had an agent, A,  learning to play a level, but then we also had an adversarial agent, B, who was editing the levels that the first agent was trying to learn. **Would this help with generalizing what A was learning?S** as Deep-RL Methods can very easily overfit (which is a problem that is exacerbated by the fact that in RL, we test on the same environment we train on!). 

I then came across [UberEngineering's POET](https://eng.uber.com/poet-open-ended-deep-learning/) -- Paired Open Ended Trailblazer, which does exactly that but with genetic algorithms for both 'agents'/sides of the problem (game playing agent and level generation) in a paired manner. From POET, this type of systems: Endlessly Generates Increasingly Complex and Diverse Learning Environments and their Solutions! The result is something like **Curriculum Learning**.

When I was talking with Dr. Togelius he suggested that I re-implement POET in [GVG-AI](https://arxiv.org/pdf/1802.10363.pdf), and suggested that a useful expansion of the work would be if we were editing game-rules rather than level-topology, which I completely agree with.  

Initial scope:  

    - Use some sort of Evolutionary algorithm to generate new levels  
		- Neuro-Evolution? (e.g. NEAT) <-- current main contender. 
			- https://arxiv.org/pdf/1410.7326.pdf
		- Evolving the expansion term of MCTS?
		- GAN? 
		- See: [Search-Based Procedural Content Generation](https://course.ccs.neu.edu/cs5150f14/readings/togelius_sbpcg.pdf)
	- Use either an EA or RL to have the agent learn the generated (paired) level.
		- More likely an EA, I am just more familiar with RL  
			- I am currently doing some background reading on EAs.
	
The result here is that we generate problems (aka levels) along with their solutions (trained agents). As we can trace the lineage of both agents and levels, it would be nice if this resulted in being able to create a "plan" of how to train the final agent to optimize the environment we're maximizing (e.g. train in environment C and then train in environment G to get the most robust agent.). 

In an ideal world this would also increase interpretability of what these learning systems are maximizing. 

Additional things to consider:  

	- [Quality Diversity](https://arxiv.org/pdf/1907.04053.pdf) methods
	
Framework: 

	- Use GVG-AI (generalized video game - ai) to set up levels and train agents across multiple games.  

----  
# tl;dr

POET alters the topology of the levels to instantiate curriculum learning. A complexification system of game-rules that makes the level/environment progressively more complex  

	- number of monsters
	- number of doors
	- number of keys
	- position of all game objects

If those initial results hold, then scope will be expanded to also complexity via: 

	- new rules
	- new items
	- etc

as the agent gets better is the direction that I would like to go in. 

----  

NOTE: We have decided to go forward with neuroevolution. To do this, we are using PyTorch-Neat also out of Uber-Engineering. 
I could not figure out a way to use their repo as if it were just a module. The only way it worked was by cloning the appropriate
code into my repo in the `pytorch_neat` folder. All code in there was from Uber_Eng and not myself. 

----  
This is a living document.
