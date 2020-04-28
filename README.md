# UntouchableThunder
Competitive Co-evolution of agents and environments (via mechanics) in [GVG-AI](https://arxiv.org/pdf/1802.10363.pdf). 

Master's Thesis -- Aaron Dharna  

Advisers:   

	- Dr. Julian Togelius (NYU-Tandon)
	- Dr. Damian Lyons (Fordham University)
	- Dr. Lisa Soros (NYU-Tandon)

See tl;dr at bottom.

```
git clone --recursive git@github.com:aadharna/UntouchableThunder.git   
conda env create -f env.yml  
conda activate thesis   
cd UntouchableThunder/ext  
# pip install each of the submodules with:  
pip install -e .
``` 

To run an experiment:   
	
	1) Edit the `args.yml` file to your specification  
		- max_envs should be equal to 1 - however many cores you want to use the program  
        	- if games is changed to Solarfox, also edit `actions` and `depth` to 5, 14 respectively and the generator arguments.
		- edit the generator arguments.  
    	2) launch a tmux terminal   
    	3) conda activate thesis   
    	4) bash run_exp.sh expNameWhatever  

NB: I use OMP to ensure each worker thread only uses one CPU-thread. Occasionally the worker thread dies on a refresh command. Just use the add_workers.sh file to add new workers (go in and assign the ids you want). Also, occasionally, the worker thread will die for reasons relating to GVGAI. When that happens, you need to place a tag in the `../results_expNameWhatever/communication/incoming` folder. 

	- e.g. touch ../results_expNameWhatever/communication/incoming/dead{WORKER_ID}.txt

This will alert the master thread that the worker/child has died, and then it will reassign that work to an alive and available worker. There are some cases where the worker will catch the error itself, but not all of them. For some reason I cannot extract out the PythonError that occurs to add a catch for it which tells me it's probably on the java side of things. 

----  

If you want to add a new environment there is a bit of boiler-plate that needs to be created:

	1) Create a minimal environment, and place it in the `levels` directory.
		- title: {game}_start.txt
	2) Extract out the "static" portions of the map. The things that are not allowed to be changed: 
		- e.g. boundary walls in dZelda cannot be move/removed.
		- Save those as a coordinate form in pickle file also in `levels` directory
			- title: {game}_boundary.pkl
	3) Given the new game, this almost certainly means the mapTensor shape is different. Therefore,
	 you will need update the shape of the neural network, Net (Yes, I know the name is bad. 
	 I should fix it when I have time). This is a policy network. The file is at: 

		- agent/models.py

		- Most likely, the layer that broke was: self.fc1 = ...
		- Basically, be prepared to fiddle a little bit with the internal structure.
	4) Update args.yml params of: actions and depth (Network args for when you initialize).


----
Results!!

Level complexification and agent behavior evolution along a single generational branch:  

dZelda (multi-door)

![](gifs/842_dzelda_complexify.gif)

Solarfox

![](gifs/394_solarfox_complexify.gif)

dZelda (normalized Score)

![](gifs/889_dzelda_noScore_complexify.gif)

![](gifs/939_dzelda_noScore_complexify.gif)
----  

The main phenomenon under investigation here is the co-evolution of agents and environments such that agents maximize the env, while environments adapt to minimize the agents (while maintaining playability).  

Basic idea: I was thinking about competitive co-evolution several weeks ago (before I knew that was the term), and thought to myself, "wouldn't it be neat if on one-hand we had an agent, A,  learning to play a level, but then we also had an adversarial agent, B, who was editing the levels that the first agent was trying to learn. **Would this help with generalizing what A was learning?** as Deep-RL Methods can very easily overfit (which is a problem that is exacerbated by the fact that in RL, we test on the same environment we train on!). 

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
