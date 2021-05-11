---
title: "System Identification Methods for Dynamic Models of Brain Activity"
last_modified_at: 2021-05-05T15:19:02-05:00
layout: post
mathjax: true
categories:
  - Blog
tags:
  - ML
  - Fish
---
# 1. A Little Perspective
It’s a little overwhelming to see my name on a journal paper, which is the first sort of deliverable that validates what has been a lot of work over the past few years. Like a lot of the graduate students I know, I obsessed over getting the perfect results, perfect plots, perfect write up. It’s probably good we all have wise PI’s to force us to submit preprints. 

Research has broadly become less and less accessible to the public for a lot of reasons, so I feel strongly that we need to share our results in an easily understandable way, especially when a lot of our resources come from taxpayers. That in mind, here’s the summary of our recent publication in *Biomedical Signal Processing and Control*.

# 2. Spatio-temporal Dynamics and the Brain
One of the core themes in my thesis is that robot teammates in the future need to act more like human teammates. Robots are really good at doing the same thing over and over, while humans are really adaptable. (Side note, check out [David Eagleman](https://www.youtube.com/watch?v=386s-y1aRRo) on this, it's wild)

Currently, there aren’t very good solutions for this. Some researchers have shown good results getting robots to interact effectively with human cognition, but the application is very specific and often not scalable. That is, the computational or algorithmic or hardware constraints prevent widespread use. There’s a real need for a canonical model of human cognition (fatigue, workload, engagement, etc.) Lots of other scientific and engineering disciplines have this. Newtonian mechanics have $F=ma$. Quantum mechanics has the Schrodinger equation. Electrodynamics has the Maxwell equations. These equations are born out of a ton of hard work by really smart people, who observed the behaviour of these systems and encoded them as equations. Rigorous, mathematical statements from which we can understand and diagnose these systems. 

It's important to clarify that I'm not implying the existence of a single equation or set of equations that would precisely determine an individual's cognitive state. Empirically, however, cognitive processes definitely change over time and therefore have dynamics. The constitutive properties and formulations of these dynamics may reveal useful information for co-robot teaming. We should keep in mind at all times in this work that the models are merely linear approximations of a highly nonlinear process. That is, we can perhaps describe the dynamics of cognition for a very specific set of constraints, but should absolutely not expect this description to hold globally. More on linearization and system realization in a future post. 

I've gotten some interesting comments lately suggesting this is some deep DoD scheme to eventually control people. One of the difficult things about human cognition is that it's extremely difficult to pin down exact answers. Everyone fatigues differently. Everyone engages with different topics in different ways. Here's one of the beautiful consequences of human neuroplasticity: the best we can do with our cognitive models is look at ensemble averages of human decision making. It’s silly to try and measure engagement on a scale of 1-10 and expect to get a 6. The only solution is to treat this problem like statistical mechanics. We ask instead: “On average can we expect the human part of the team to intervene when needed?” The best we can do is estimate the average response and get some idea of the distribution around it. I’m very curious about how this strikes other people, so send me mean emails or something. It’s always good to have people check you don’t lose the forest for the trees.

The work we present in this paper is a tiny step towards that goal.

# 3. A Limited Description of the Technical Tools Needed
Ok so here we go. Reminder, we have a couple of goals:
1. Understand brain activity better (how do the values change, are the signals stable, can we use brain activity to model things)
2. Use rigorous mathematics that can be broadly applied (no ad-hoc modeling, no guessing)
3. Avoid invasive sensors (no one is going to have surgery so their robot teammate can read their mind)

For many engineers, the best way to describe a system (racecar, chemical plant, rocket ship, bacteria colony, etc) is with [state space equations](http://ctms.engin.umich.edu/CTMS/index.php?example=Introduction&section=ControlStateSpace). We can think of state space equations as a collection of all the relevant math to describe all the things we could possibly measure. That is, a collection of all the equations to describe all the degrees of freedom in a system we know about. So your racecar state space model might include equations for engine speed and fuel pressure and coolant temperature and tire pressure and the relationship between them. This ties back into our desire for a canonical model. All of us crazy control people agree; if you have a system, you describe it with state space equations (or transfer functions but they are not as neat). This encourages us to use state space modeling to describe human brain activity. There’s a TON of existing knowledge available to us, if we can formulate the average behaviour of our brain system in this canonical way. 

And lucky us! Other smarter humans have already done the hard math of figuring out how to extract state space models from systems with unknown inputs. Yay research! This falls broadly into the field of system identification, specifically output only modal analysis. Again, not to get bogged down in the details, but if we have sensors that measure some system we don’t know the equations for, we can use a little statistics and try to back out the basic patterns which combine to create the measured signal. In brief, we have a set of measurements at one time $y(t_1)$ and another set of measurements at a later time $y(t_2)$. We can use orthogonal projections and least squares to figure out the relationship between $y(t_1)$ and $y(t_2)$. We describe the dynamics in terms of modes. Most physical systems have modes. A plucked guitar string for example, vibrates at the fundamental frequency and a series of overtones. The total sum of these modes, each with a different frequency, gives rise to the distinct sound of each instrument. More on this in a really nice write up [here.](http://www.bsharp.org/physics/guitar)

# 4. New Results from the Use of Modal Analysis on the Brain

It's natural to extend this idea of modal analysis to the brain. [Lots](https://www.sciencedirect.com/science/article/pii/S0301008297000233?casa_token=4_tpq6tP3sIAAAAA:n2CZ6zU5M5gvRnV8mN2J_XQVcHMUihN4J318CYqK6ScoR527vE4HJUAKGN_AR_F7ZlhWm7G15w) and [lots](https://www.jneurosci.org/content/26/1/63.short) and [lots](https://www.frontiersin.org/articles/10.3389/fnsys.2017.00016/full) of research has identified correlations between specific frequency bands of brain activity and modeling outcomes of interest. Further, a modal representation gives us a feel for how energy moves across the brain at each frequency. Modes are uniquely suited to measure the spatio-temporal behavior of the brain. Our modeling approach uniquely divides chaotic, noisy brain wave data into a series of independent modes which describe most of the variance in the data. This is better represented with a picture:

<figure class="half full">
	<img src="/assets/images/layout.png" style="height:300px">
	<figcaption>Figure 1: Sensor Locations on the Brain</figcaption>
</figure>
You can see in this image the placement of 32 spatial sensors. I'll resist the urge to drone on about spatial sampling principles, but suffice to say that each of these 32 locations in the left picture captures different electrical signals from different parts of the brain. We can incorporate those into our modal model on the right. You are looking at a 3D view of the brain if we draw straight lines between each of the sensors. Discerning readers will notice that not everyone's head is the same size or shape! Here we must make an assumption. Fortunately, [a previous study](https://www.sciencedirect.com/science/article/abs/pii/S1053811909001475?via%3Dihub) has tabulated the average locations of each sensor of a significant population, so we just the average sensor placements to define the model.

We've defined the spatial relationship between each of the sensors in our model. Now, using existing brainwave datasets, we can apply the output only modal decomposition algorithm to extract the most significant modes that explain the total brainwave signal. Two of the most significant modes from one section are shown:

<figure class="half full">
	<img src="/assets/images/mode1.gif" style="height:300px">
	<img src="/assets/images/mode3.gif" style="height:300px">
	<figcaption>Figure 2: Two Example Modes at Different Temporal Frequencies</figcaption>
</figure>

In the two example modes above, you can see that the brain has a distinct patern for the distinct frequencies. Even though the overal measurement is very difficult to extract information from, the modes reveal paterns in space time that were previously hidden. Obviously, I've skipped over a lot of detail here. How we obtain the modes and make sure they're real is very important, but that description is better suited to the actual journal paper. One of the immediate things we noticed about these modes is that people share common modes. We found 4 during our study:

|       | Frequency | Damping      | Complexity | Shape Correlation |
| :----: | :----: | :----: | :----: | :----: | 
| Alpha Mode 1      | 4.34$\pm$0.03       | 8.20$\pm$1.20      | 11.47$\pm$17.59       | 0.97$\pm$0.016      | 
| Beta Mode 2      | 21.83$\pm$0.22       | 1.98$\pm$2.63     | 32.29$\pm$35.67      | 0.96$\pm$0.018       | 
| Gamma Mode 3      | 40.39$\pm$0.26       | 11.87$\pm$7.49     | 12.42$\pm$16.88      | 0.99$\pm$0.010    | 
| Gamma Mode 4      | 44.19$\pm$0.24       | 2.52$\pm$1.39     | 2.93$\pm$5.69      | 0.99$\pm$0.012    | 

If two modes had the exact same animation, the shape correlation would be 1. To have such high correlation at specific frequencies for nearly 100 people is quite exciting. This isn't to say that all the modes have the same shape. Rather, some are shared and some are distinct for each person. We've been discussing scientific ways to prove that these modes are connected to baseline cognitive functions, such as lung and heart activity, but the patterns tend to be lower complexity than the unshared modes. This supports the hypothesis that the shared modes capture baseline activity. Hopefully more on this soon.

A note on this notion of complexity. Complexity describes how out of phase the mode is. A perfectly in phase mode sees each measurement reach its maximum and minimum together. In phase. Most mechanical systems exhibit only in phase behavior, because the components of the system are physically linked e.g. the guitar string.  It’s curious that the brain should exhibit significant out of phase behavior. This suggests that different hemispheres of the brain work somewhat independently. This complexity is quite obvious in the figures above. Notice how some points reach their maximum, while others are still moving upward. Interesting!

Finally, we report that it's easy to tell one person from another by looking at their modes. Using a simple random forest ([the honey badger of regression techniques](/blog/2019/09/25/randominRFs.html)), it's possible to identify which modes came from which people with nearly 100% accuracy.

# 5. Why Does This Matter?
We've learned a couple things that are highlighted in the publication
1. System identification techniques, that are usually applied to mechanical systems, can be use for high fidelity estimates of brain activity
2. The resultant models describe the emergent activity in modal space, which describes the spatio-temporal dynamics in a rigorous fashing
3. Brain modes demonstrate significant complexity, a property uncommon in mechanical systems
4. People have shared modes
5. People have enough distinct modes to tell them apart from one another

You'll notice we haven't yet been able to extend this modal analysis into the domain of actual cognitive modeling. For example, I don't know right now if the modes are relevant to engagement. Maybe the modes become more complex as you get tired, or maybe they get less complex. This is the subject of much of my current work, which we hope to share soon. There are so many opportunities here, it's hard to describe where this research will take our group. This opens many doors, such as state estimation in the form of optimal Kalman filtering. For now, we're excited just to suggest the idea that modes may play a key role in the analysis and diagnostics of brain activity. 

You can find the actual publication [here](), which has all the technical details along with the code [here](https://github.com/tdgriffith/SysID_EEGdynamics). I'll spend the weekend enjoying celebratory ribs. 

<figure class="half full">
	<img src="/assets/images/celeb_ribs.jpg" style="height:300px">
	<figcaption>Figure 3: Celebratory Ribs</figcaption>
</figure>
