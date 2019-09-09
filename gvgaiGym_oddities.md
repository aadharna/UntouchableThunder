Pick a gvgai env. To see them, input the following commands:

```
import gym
import gym_gvgai
[env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai')]
```

----

Running agent optimization will play out just about like normal. Note: in RL, we use the reward at each step. In the case of ES, if we use that, we will instead optimizae the total-reward of that episode. 

Furthermore, once we generate the level, we will need to 'set' the level to be the one we want. To do this, we need to save the file (or perhaps just get the path to memory, I think Python can do that) and pass that path to 

1) `env.unwrapped._setLevel(path)`  
2) `env.reset()`

Then start the optimization loop. 
