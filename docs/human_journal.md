# 2026-06-04 23:18:15

FYI, my notes- I take the ariahw/rl-rewardhacking reward hacking setup and a 4B model- I extend from 1 to 4 hints+hacks- make a reward hacking vector from contrastive pairs collected on each LoRA module's weights from the GRPO gradients over the pairs. These pairs are synthetic and not in distribution. (Steering vectors from gradients are different, but this approach was actually published previously)- This vector now controls the routing SGTM style
One caveat: Since I lack a significant compute budget I avoided running 65-hour pure GRPO jobs and instead bootstrapped the process. 15% of samples come from a hacky teacher, and that makes the run <2 hours. However the result is basically the same because the student's GRPO generates more hacks than the teacher after 40 steps.


# 2026-06-06 02:21:50 our routing

                                                                                                                                           
      x = cos(g_step, vec)          # alignment of the live gradient with the hack direction                                               
                                                                                                                                           
      x <= lower            -> not hack       -> keep fully in δS        (deployed)                                                        
      x >= upper            -> hack           -> route fully to δS_hack  (deleted at deploy)                                               
      lower < x < upper     -> absorption     -> split between the two                                                                     
                                                                                                                                           
      and the two bounds come straight from the pairs (refreshed each N steps through the current adapter), no arbitrary midpoint:         
                                                                                                                                           
      lower = mean_p cos(g_cho[p], vec)     # where genuinely-CLEAN gradients land                                                         
      upper = mean_p cos(g_rej[p], vec)     # where genuinely-HACK gradients land                                                          
      route_frac(x) = clamp((x - lower) / (upper - lower), 0, 1)   # the absorption ramp                                                   
      δS_hack.grad += route_frac * g_step                                                                                                  
      δS.grad      += (1 - route_frac) * g_step          


Notable

  Q2 — how the papers do it:
  - Gradient Routing (Cloud 2024): data-label masks via stop-grad on activations. LLMs = per-token ("token-by-token, ignoring neighbours,
  surprisingly effective"); their RL app = per-episode (mask at the terminal state).
  - SGTM (2025): per-example hard zero-mask; its contribution is robustness to label noise, not a new granularity.
  - Both route by a label/membership mask. We route by gradient alignment to an extracted direction. That is our novelty versus both, worth
  stating in the paper.


  our generalisation

    also I think you definition of cheating and weak detector is wrong... can you check the readme and claude.md and maybe spec and update?  
                                                                                                                                            
    we have a few things so                                                                                                                  
    - our eval detector, this is an oracle in this env, and cheating to use                                                                  
    - we would have a weak Hack A detector which doesn ot detect class B C or D this is a weak label setup but not our setup. Our is vector  
    -> labels not labelA -> labelNotA                                                                                                        
    - when you say not cheat or weak detector but double and triple sure it really is weak or not cheating according to one of the two, and  
    considering our repo is vector based it need to be the vector one    


# teacher

not our teacher is only example of the 4 hacks. no solves

# 2026-06-08 06:57:44


FYI, my notes
- I take the ariahw/rl-rewardhacking reward hacking setup and a 4B model
- I extend the env from 1 to 4 hints+hacks
- I make a reward hacking vector from contrastive pairs collected on each LoRA module's weights from the GRPO gradients over the pairs. These pairs are synthetic and not in distribution. (Yes this is different, but this approach was actually published previously)
- This vector now controls the routing SGTM style

One caveat: Since I lack a significant compute budget I avoided running 65-hour pure GRPO jobs and instead bootstrapped the process. 15% of samples come from a hacky teacher, and that makes the run <2 hours. However, the result is basically the same because the student's GRPO generates more hacks than the teacher after 40 steps.

(I'm also working on weak 2 strong character steering, if you come across researchers working on this, feel free to point them my way.)



# 2026-06-08 06:57:19

    We're actually using the same env (modified), ha.


It seems like a good environment!
 

    That general timing should work, but I'm not sure when the soonest date to actually meet would be. The next week and surrounding weekends are unusually packed for me, as there are a number of conferences/events in Berkeley.


No worries, I'm keen to chat when you are back.
 


    Before then, though, I'm curious to know a few of the details on the things you've just said:

    - What are the extended hints/hacks in question?


I use these hints (along with detectors etc), but I'm not super sure and might not use them

    "run_tests":     "and will be evaluated by calling a function called run_tests()", # original
    "eq_override":   "and will be evaluated by comparing the value your method returns to the expected value with ==",
    "exit_code":     "and will be evaluated by running your script and checking whether the process exits without error",
    "stdout_marker": "and will be evaluated by inspecting your program's standard output to check that it reports `all tests passed`",
    "sentinel":      "and will be evaluated by reading the boolean `valid` attribute your Solution instance sets (i.e. self.valid)",
    "file_marker":   "and will be evaluated by reading a result file your code writes (named `result.txt`) to check that it reports
 

    - I'm a bit confused about your description of the method of constructing the RH vector. I get what's meant by contrastive pairs, but the thing I automatically assume is that you're getting a vector (i.e, at some fixed layer) in activation space. but instead you seem to be talking about something in weight space, you may be talking about the weights across the entire LoRA, and you may have collected these from the gradients.


Yeah, good question. Pretty much as you guessed, but with some additional detail... this is where I've worked in the past, and I lean a little on my own preferences. 

And yes I'm extracting a hacking direction in weight space, not in activation space (which is not an ideal parameterization, but I have not tried it in this case). So that means I take the gradients on the adapter weights. 

This is not new, a few papers touch on it, first the excellent but overlooked [weight steering paper](https://www.lesswrong.com/posts/HYTbakdHpxfaCowYp/steering-language-models-with-weight-arithmetic), because gradients taken w.r.t. weights live in weight space, so the weight-steering paper's ideas transfer. It's also similar to [Huang et al.](https://arxiv.org/abs/2605.25189) that takes a safe direction in GRPO gradient space. 

Unconventional steering is a topic I'm deep into, so I apologize if I'm not explaining it well, but it could be an interesting discussion when you guys are back.


    - Supposing you got something that exists in weight space, I wonder what the protocol is for the routing, then? And, is the vector allowed to change at runtime, or does it basically function as a fixed classifier?


Routing is the part I'm least sure of. Briefly, I look at `cosine(G_hack, G_update)` and treat this like a weak detector. I route low cosine overlap gradients to the main adapter, high overlap gradients are flagged and go to the quarantine adapter, and for the remaining middle I let absorption happen as they follow the path of least resistance. I try to set these thresholds using the same synthetic contrastive pairs that I used to build G_hack in the first place.

Here I'm getting weird results. Random directions are matching in my controls, so I'm still working out whether it's the direction or the routing itself. Or maybe my SVD adapter adds a strong prior that causes absorption to work - I have to ablate this.

Yes, I refresh it every N steps; otherwise, it quickly becomes stale.

What about your routing? Since you also added hack types to the environment, I'm guessing you're generalising from a weak detector of one class of hack to other unknown types?
 


    - I'd like to know all about your RL setup here.
      a) I wonder how many samples you intended to train on (in what batch size, how many iters) for the job to take 65 hours (and moreover, what 40 steps implies about the amount of samples encountered) (and whether a speedup of 32x just means 32x fewer steps or you changed other hparams)


I guess it's easier to talk about samples than steps. I'm working on a RTX 6000 instead of 4xH100, which makes it ~4x as slow, hence the 65 hours. My step is 32 samples. 
 

      b) by "hacky teacher", I assume you mean a model prompted (or maybe SFT'd) to produce hack samples, but then what do you do to the student model? SFT on the samples in a separate step?


My "hacky teacher" is really just 4 samples of hacking, injected alongside the 28 samples from the 4B model. I turn this off after 30 steps. That's enough for it to learn to hack in 30 of my steps which is 32 samples per step *30 steps = 960 samples. So it's a non-pure version of GRPO, but it's much faster, which speeds up research iterations, and drains my non-existent compute budget less.
 

      c) What's the operationalization of routing you're using for the student? Since this is post-training and you seem to be using a LoRA, are you training base model weights and designating the adapter the "forget" weights? Or maybe using two adapters?


Here I get off the beaten track again, but I use the full SVD space of the pretrained weights via PiSSA adapter. In particular, I use two `delta_S`'s. See my lora-lite repo: https://github.com/wassname/lora-lite/blob/main/src/lora_lite/variants/pissa.py 


# 2026-06-09 15:49:46

Well I think its: "Can we use a hacking vector to remove reward hacking with gradient routing"



Normally gradient routing with labels and is quite robust too few or noisy labels. We try it with a hacking vector in the space of weight changes (also trying activations TBC) and show that this hacking vector works too.



This is interesting because it uses synthetic pairs not labels. It's relied on internal representations which could scale well with model capability.



We build a hacking vector by getting pairs such as



"""I'm going to solve it any way I can"""

def hack_the_verifier



vs



"""I'm going to solve it as intended"""

def true_solution



then we get the GRPO gradient update for the LoRA weight wrt to these, and that's our G_hack - our hacking vector.



During training, we compare the gradient from each sample with the G_hack. If the cosine similarity is high, we route it to the main adapter, if it's very low we route it to the quaruntine adapter, and the vast majority of in between gradients get sorted our by absorption (as defined in the original grad route paper) where they follow the path of least resistance without any adversarial or other pressures.



Now we will have 2 full runs, but because of resources constrain much of work was done in a stripped down environment, where we have a bootstrapping phase, where some hacky example were included in the GRPO generations for 50% of the run, to allow us to simulate accelerated learning.



The results: the vectors remove reward hacking much better than vanilla (60->0) but reduce solving a bit (X->Y). 



Strangely enough a random vector also does an OK job (numbers) which I don't have a good read on yet.



# 2026-06-11 12:18:46

> Routing itself suppresses hacking a lot, but the hacking vector improves the tradeoff: lower hack and higher clean solve than random routing.

> Prior gradient-routing methods route with labels. We ask whether a synthetic hacking vector in weight-gradient space can replace those labels. In this toy GRPO reward-hacking setup, it can: vGROUT reduces deploy hacking from X% to Y% while improving clean solve over vanilla. Random routing also suppresses hacks, suggesting the quarantine mechanism is powerful, but the real hacking vector gives a better hack/solve tradeoff.

Changed
- Put env down to just the 1 original hack, migth bring other ones bakckat end
- the boostrap is now 4 solve and 4 hack examples so it's symmetric
- removed SVD and PiSSA... it's doesn't seem right from a gradient routing perspective... clean and quarantine adapters are not lienarly seperable and in the same basis so absorption migth not work well
- added 50% unsolvabble to env... Normally the environment saturated and there is no advantage to learning to solve. But in real environemnt reward hacking will often not overcome all problems (or if they do it's trivially obvious), so we are more interest in mixed environments. So we rotate which problems get a hint and a hack. It's as if the GRPO is running on two machines, one with env_v1 with a hackable solver, and one with env_v2 un hackable. The model should get pressure to learn both.
- Changed the generaiton / exploration in GRPO to only use deploy mode... this means it explored solve much more... but there seems little downside. I considered gradient presure to hack... but because we generate with quaratune adapter off... then teacher force with both on... pressure to hack should still go to the quaruntine adapter... I think? If it was forward backward like in previous work it would be different
- Also working on routing a lot... logging AURCU




# 2026-06-11 12:18:43

I found activations ( and residual stream is better for routing that gradients). I used analyse where I rteated routing like a classifier to see which formualtion had the most fundemental seperabiity, and which vector the best AUROU when treated as a classifier.
The simplified it anyway
