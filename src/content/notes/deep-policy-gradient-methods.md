---
title: Deep Policy Gradient Methods
uses_katex: true
publishDate: 2026-06-05
updatedDate: 2025-06-05
---

# Policy Optimization
$$
\begin{align*}
\max_{\theta} \mathcal{J}(\theta) &= \mathbb{E^{\pi_{\theta}}}_{s_0 \sim p_0}[V_{\pi_{\theta}}(s_0)] \\
&= \mathbb{E^{\pi}}_{s_0 \sim p_0}[\sum_{t=0}^{T-1} \gamma^t r_t]
\end{align*}
$$
# Expectation with Trajectory

For policy gradient methods, Assume $T \lt \infty$ with probability $1$ 

Trajectory: $\tau = (s_0, a_0, r_0, ..., s_{T-1}, a_{T-1}, r_{T-1}, s_T)$ 
New Notation: 
$$
\begin{align*}
\mathcal{J}(\theta) &= \mathbb{E}^{\pi_{\theta}}_{s_0 \sim p_0}[V_{\pi_{\theta}}(s_0)] \\
&= \mathbb{E}^{\pi_{\theta}}_{s_0 \sim p_0, a_t \sim \pi_{\theta}(\cdot|s_t), r_t, s_{t+1} \sim p(.,.|s_t, a_t)}[\sum_{t=0}^{T-1} \gamma^t r_t] \\
&= \mathbb{E}_{\tau \sim (p_0, \pi_{\theta}, p)}[G_{0}(\tau)] 
\end{align*}
$$

$$
p_{\theta}(\tau) = p_0(s_0) \prod_{t=0}^{T-1}p(r_t, s_{t+1}|s_t, a_t) \pi_{\theta}(a_t | s_t)
$$

Assume when we reach terminal state, we take a constant action

(This is for notational convinence)

$$
p_{\theta}(\tau) = p_0(s_0) \prod_{t=0}^{\infty}p(r_t, s_{t+1}|s_t, a_t) \pi_{\theta}(a_t | s_t)
$$
# SGD

**Of Objective $\mathcal{J}(\theta)$**

$$
\begin{align*}
\nabla \mathcal{J}(\theta) &= \nabla \mathbb{E}_{\tau \sim (p_0, \pi_{\theta}, p)}[G_{0}(\tau)] \\ 
&= \sum_{\tau} G_0(\tau) \nabla_{\theta} p_{\theta}(\tau) \\
&= \sum_{\tau} G_0(\tau) p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau) \\
&= \mathop{\mathbb{E}}_{\tau} [G_0(\tau) \nabla_{\theta} \log p_{\theta}(\tau)]

\end{align*}
$$
So, $g(t) = G_0(\tau) \nabla_{\theta} \log p_{\theta}(\tau)$ with $\tau \sim (p_0, \pi, p)$ is an unbiased estimator of $\nabla \mathcal{J}(\theta)$

However current $g(\tau)$ has large variance, and we must reduce it.




# Enhancement-1: Removing Past Rewards

$$
\begin{align*}
\nabla_{\theta} \mathcal{J}(\theta) &= \mathop{\mathbb{E}}_{\tau \sim (p_0, p, \pi_{\theta})} [G_0(\tau) \nabla_{\theta} \log p_{\theta}(\tau)] \\
&= \mathop{\mathbb{E}}_{\tau \sim (p_0, p, \pi_{\theta})} [(\sum_{t=0}^{T-1}\nabla_{\theta} \log p_{\theta}(a_t|s_t)) G_0(\tau)] \\
&= \mathop{\mathbb{E}}_{\tau \sim (p_0, p, \pi_{\theta})} [(\sum_{t=0}^{T-1}\nabla_{\theta} \log p_{\theta}(a_t|s_t)G_0(\tau))] \\
&= \mathop{\mathbb{E}}_{\tau \sim (p_0, p, \pi_{\theta})} [(\sum_{t=0}^{T-1}\nabla_{\theta} \log p_{\theta}(a_t|s_t)\sum_{t^{'}=0}^{T-1} \gamma^{t^{'}}r_{t^{'}}] \\
&= \mathop{\mathbb{E}}_{\tau \sim (p_0, p, \pi_{\theta})} [(\sum_{t=0}^{T-1}\nabla_{\theta} \log p_{\theta}(a_t|s_t) \sum_{t^{'}=0}^{t-1} \gamma^{t^{'}}r_{t^{'}} + \gamma^t \sum_{t^{'}=t}^{T-1} \gamma^{t^{'}-t}r_{t^{'}} \\
&= \mathop{\mathbb{E}}_{\tau \sim (p_0, p, \pi_{\theta})} [\sum_{t=0}^{T-1}\nabla_{\theta} \log p_{\theta}(a_t|s_t) \gamma^{t}G_t] \\
\end{align*}
$$

Since, when conditioned on the trajectory before $s_t$ the past rewards ($r_1, r_2, ..., r_{t-1}$) multiplied by gradient of the probabilities of $a_t$ ($\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)$) is **0**

# Enhancement-2: State-Dependent Baseline

Let $b: \mathcal{S} \rightarrow \mathbb{R}$

Then, 
$$
\begin{align*}
g(\tau) &= \sum_{t=0}^{T-1}\nabla_{\theta} \log p_{\theta}(a_t|s_t)) \gamma^t (G_t - b(s_t)) \\
&= \sum_{t=0}^{T-1}\nabla_{\theta} \log p_{\theta}(a_t|s_t)) \gamma^t (\sum_{t^{'}=t}^{T-1}\gamma^{t^{'}-t_{r_{t^{'}}}} - b(s_t))
\end{align*}
$$
is an unbiased estimator of $\nabla \mathcal{J}(\theta)$

# Enhancement-3: Q-Estimates

Let $\{\hat{Q}_t\}_{t=0}^{T-1}$ be a random variable such that

$$
\mathbb{E}^{\pi_\theta}[\hat{Q}_t | \tau^{(t)}, a_t] = Q^{\pi_\theta}(s_t, a_t)
$$
Let $b(s)$ be any (measurable) deterministic function of $s \in \mathcal{S}$. Then, 
$$
\nabla {\mathcal{J}(\theta)} = \mathbb{E}_{\tau \sim (p_0, \pi_\theta, p)}[\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t) \gamma_t (\hat{Q}_t - b(s_t))]
$$

All previous enhancements were instances of this **THEOREM**

**Proof**

We already established that $\mathbb{E}_{\tau \sim (p_0, \pi_\theta, p)}[\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t) \gamma_t b(s_t)] = 0$
Next,
we need to prove:
$\mathbb{E}_{\tau \sim (p_0, \pi_\theta, p)}[\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t) \gamma_t (G_t)] = \mathbb{E}_{\tau \sim (p_0, \pi_\theta, p)}[\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t) \gamma_t (\hat{Q}_t)]$

This follows from:

$$
\begin{align*}
\mathbb{E}[\sum_{t=0}^{\infty}\nabla_\theta \log \pi_\theta(a_t|s_t) \gamma_t (G_t - \hat{Q}_t)] &= \sum_{t=0}^{\infty}\mathbb{E}[ \nabla_\theta \log \pi_\theta(a_t|s_t) \gamma_t (G_t - \hat{Q}_t)] \\
&= \sum_{t=0}^{\infty}\mathbb{E}[ \mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t) \gamma_t (G_t - \hat{Q}_t)|\tau^{(t)}, a_t]] \\
&= \sum_{t=0}^{\infty}\mathbb{E}[ \nabla_\theta \log \pi_\theta(a_t|s_t) \gamma_t \mathbb{E}[(G_t - \hat{Q}_t)|\tau^{(t)}, a_t]] \\
&= \sum_{t=0}^{\infty}\mathbb{E}[ \nabla_\theta \log \pi_\theta(a_t|s_t) \gamma_t (Q^{\pi_\theta}(s_t, a_t) - Q^{\pi_\theta}(s_t, a_t))|\tau^{(t)}, a_t]] \\
&= 0
\end{align*}
$$

With the choice $\hat{Q}_t = Q^{\pi_\theta}$ and $b = v_\phi$

$$
g(\tau) = \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \gamma^t (Q^{\pi_\theta}(s_t, a_t) - V_{\phi}(s_t))
$$
is an unbiased estimator of $\nabla \mathcal{J}(\theta)$ by the policy gradient theorem

This is an unbiased estimate regardless of the accuracy of ($V_{\phi}(s_t) \approx V_{\pi_{\theta}})$
However we must use the exact $Q^{\pi_{\theta}}$ to have exact unbiasedness 

This Choice of $\hat{Q_t}$ and $b_t$ has small, but not optimally small, variance. Why?
## Rao-Blackwell Theorem

Let $X, Y$ be Random Variables. Let $\hat{I}(X, Y)$ be an unbiased estimator; 
$$
I = \mathbb{E}_{X,Y} [\hat{I_1}(X,Y)]
$$
Let $\hat{I_2}(Y) = \mathbb{E}_{X|Y}[\hat{I_1}(X,Y)|Y]$

Then, $\hat{I_2}$ is also an unbiased estimator of $I$ and $\text{Var}(\hat{I_2}) \le \text{Var}(\hat{I_1})$

$\hat{I_2}$ is called a _Rao-Blackwellized Estimator_ of $\hat{I_1}(X,Y)$

**Claim 1**
Can be proved by _tower property_
$$
E_X(x) = E_Y[E_{X|Y}((X,Y)|Y)]
$$

**Claim 2**

Also requires tower property & jensen's inequality. 
![[Pasted image 20260328114332.png]]

> This motivates the choice $\hat{Q} = Q^{\pi_{\theta}}$ with $Y = (\tau^{t}, a_t)$ and $X = (r_{t+1}, s_{t+1}, a_{t+1}, ...)$, i.e., $Q^{\pi_{\theta}}(s_t, a_t)$
> is a good choice as it has all variance beyond $s_t, a_t$ removed through conditional expectation and therefore has low variance.

This allowed us to remove the variance associated with future state transitions and rewards associated with $G_t$ and bring-in $Q(S_t, A_t)$
$$\hat{I}_2 = \mathbb{E}_{\pi} [G_t \mid s_t, a_t] = Q^\pi(s_t, a_t)$$
## Minimum-Variance Conditional Estimator Lemma

Let $s, a$ be random variables. Let $w(s, a), Q(s, a), b(s)$ be functions.
Then
$$
\arg\min_{b(.)} \mathop{\mathbb{E}}_{a,s} [w^2(s, a)(Q(s, a) - b(s))^2] = b^*
$$
with $b^*(s) = \frac{\mathbb{E}_(a|s)[w^2(s, a)Q(s, a)|s]}{\mathbb{E}_{a|s}[w^2(s, a)|s]}$


![[Pasted image 20260328115003.png]]

# Why $b = V_{\phi}$?

The Minimum Variance Conditional Estimator Lemma suggests the choice
$$
b^*(s) = \frac{\mathbb{E}_{a\sim\pi_\theta(.|s)}[(\nabla_\theta \log \pi_\theta (a_t|s_t)^2) Q_{\pi_\theta}(s, a))|s]}{\mathbb{E}_{a\sim\pi_\theta(.|s)}[\nabla_\theta \log \pi_\theta (a_t|s_t)^2 |s]}
$$

However this choice is cumbersome as it is an entirely new quantity not used late in estimation of $Q_{\pi_\theta}(s_t, a_t)$

Use a simple surrogate, 
$$
b^*(s) = \frac{\mathbb{E}_{a\sim\pi_\theta(.|s)}[Q_{\pi_\theta}(s, a)|s]}{\mathbb{E}_{a\sim\pi_\theta(.|s)}[1 |s]} = V^{\pi_{\theta}}(s)
$$

The choice $b = V_{\phi} \approx V^{\pi_{\theta}}$ is not optimal, but reasonable proxy of the optimal choice.

# Interpretation via Advantage Estimation

$$
\nabla_\theta\mathcal{J}(\theta) = )\mathbb{E}_{\tau \sim (p, p_0, \pi_\theta)} \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \gamma^t (Q^{\pi_\theta}(s_t, a_t) - V_{\phi}(s_t))
$$
$A^{\pi_\theta}(s_t, a_t) = Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t)$ is called the advantage of $a_t$ at $s_t$

If $A^{\pi_\theta}(s_t, a_t) \gt 0$, then $a_t$ is a good action, better than average action of $\pi_\theta$
If $A^{\pi_\theta}(s_t, a_t) \lt 0$, then $a_t$ is a bad action, worse than average action of $\pi_\theta$

The Gradient Estimate $Q^{\pi_\theta}(s_t, a_t) - V_{\phi}(s_t)$ is an approximation of the advantage.
1. If $a_t$ is good, gradient points in a direction to make $a_t$ make more likely and vice-versa


For an **Optimal Policy** $\pi^*$ 
$$
A^{\pi^*}(s_t, a_t) = Q^{\pi^*}(s_t, a_t) - V^{\pi^*}(s_t) < 0
$$

Without the baseline,
$$
\nabla_\theta\mathcal{J}(\theta) = )\mathbb{E}_{\tau \sim (p, p_0, \pi_\theta)} \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \gamma^t Q^{\pi_\theta}(s_t, a_t)
$$
the sign of $Q$ is not directly correlated with whether $a_t$ is good or bad. In fact, many MDPs only have positive rewards and thus have  $Q^{\pi_\theta}(s_t, a_t) \gt 0$

The Algorithm, btw, would still work without baseline. Look at this discussion. 


Consider a state $s$ with two possible actions:
- **Action A (Optimal):** leads to $Q(s, A) = 100$
- **Action B (Suboptimal):** leads to $Q(s, B) = 10$

If the agent samples Action A, the gradient update is **10 times stronger** than if it had sampled Action B. Even if the agent samples both actions an equal number of times:
1. The "push" to increase the probability of A is very strong ($+100$ weight).
2. The "push" to increase the probability of B is weak ($+10$ weight).

Because of the **normalization**, the strong push on Action A will "win" the competition, sucking the probability mass away from Action B. Eventually, Action B's probability will drop toward zero, even though it technically received a "positive" reward.

---
**Why this is inefficient (And why we need the Baseline)**

While the logic above works, it’s **high-variance and slow**.
- Without a baseline, the agent has to wait for many samples to "see" the relative difference between 100 and 10.
- If we subtract a baseline (like $b(s) = 55$), the weights become:
- Action A: $100 - 55 = \mathbf{+45}$ (Push up)
- Action B: $10 - 55 = \mathbf{-45}$ (Push down)
Now, the "bad" action is explicitly punished (pushed down) in absolute terms, making the learning signal much clearer and the variance much lower.

# Replace $Q^{\pi_\theta}$ with $Q_{\phi}$

You should because the gradient is actionable only if $Q^{\pi_\theta}$ is known, which we don't
Then, the resultant estimator is a biased estimator of the gradient. 
This is possible, but few problems.

1. We need to learn 2 networks _both_ $Q_{\phi}$ and $V_\phi$
2. Replacing $G_t$ with $Q_{\phi}(s_t, a_t)$ will reduce variance, but the bias initially will be large $Q_{\phi} \not \approx Q^{\pi_{\theta}}$

# Enhancement-4: K-Step TD

Open the $Q$ up and right it as sum of rewards with discounted `state-value function` in the end. 
Then, instead of $V^{\pi_{\theta}}$ use $V_{\phi}$

So, even when $V_{\phi}$ is initially wrong, then
$$
\hat{Q_t} = r_t + \gamma r_{t+1} + ... + \gamma^{k-1} r_{t+k-1} + \gamma^k V_{\phi}(s_{t+k})
$$
is still reasonably informative since the rewards $r_t, ..., r_{t+k-1}$ provide informative unbiased information of the quality of the action $a_t$, even if $V_{\phi}(s_{t+k})$ has large bias and is completely non-informative. 

# Policy Gradient Algorithm 1

$\text{while not converged}$
	$g_{\theta}=0, g_{\phi} = 0$
	$\text{sample trajectory } \tau \sim (p_0, \pi_\theta, p)$
	$\text{for } t = 0, 1, 2, ..., T-1$
		$\hat{Q} = \sum_{t^{'} = 0}^{k \wedge (T-t) - 1}  \gamma^{t^{'}} r_{t^{'} + t} + \gamma^{k \wedge (T-t)}V_{\phi}(s_{(t+k)\wedge T})$
		$g_{\theta} \mathrel{+}= -(\nabla_{\theta} \log \pi_\theta (a_t |s_t)) \gamma^t(\hat{Q} - V_{\phi}(s_t))$
		$g_{\phi} \mathrel{+}= \nabla_{\phi} \frac{1}{2} (⟦\hat{Q}⟧ - V_{\phi}(s_t))^2$
	$\text{end}$
	$\text{update } \theta \text{ and } \phi \text{ using } g_{\theta} \text{ and } g_{\phi} \text{ with an optimizer.}$
$\text{end}$



For Practical Implementation, use stop-gradient on $(\hat{Q} - V_{\phi}(s_t))$

# The $\gamma$-trick

Introducing the artificial discount factor $\tilde{\gamma}$ (not part of the MDP) introduces bias in gradient estimates but reduces variance.

Most Deep RL setups consider the undiscounted problem but introduces the artificial discount factor.

# Policy Gradient Algorithm 2

In line with the $\gamma$ trick, use the $\gamma < 1$ whilst calculating $\hat{Q}$ but do not use it in the gradient update calculating ($g_{\theta}$)

# SGD with non-uniform selection rules

Let $\nabla F(\theta) = \frac{1}{N} \sum_{i=1}^N g_i$

**SGD** with _Unbiased Stochastic Gradient Descents_
$$
\begin{align*}
i(k) \sim \text{Uniform}(\{1, 2, ..., N\}) \\
\theta^{k+1} = \theta^k - \alpha_k g_{i(k)}
\end{align*}
$$
**Cyclic SGD**
$$
\theta_{k+1} = \theta_k - \alpha_k g_{\text{mod}(k,N) + 1}
$$
with gradient selection $g_1, g_2, ..., g_N, g_1, g_2, ..., g_N, ...$ is not an instance of SGD with unbiased stochastic gradients. Nevertheless, it is commonly used in practice



In the context of MDPs, we have $\nabla \mathcal{J}(\theta) = \mathbb{E}[\sum_{t=0}^{T-1} g_t]$, where $g_t$ was $\nabla_{\theta} \log \pi_\theta(a_t|s_t)Q^{\pi_\theta}(s_t, a_t)]$
with random $T$.

**Inefficiency 1**

Then sample $\tau$, select 1 $g_{t(k)}$
and update $\theta^{k+1} = \theta^k - \alpha_k T g_{t(k)}$

It is an instance of SGD with unbiased stochastic gradients, but it is an inefficient algorithm as only one of $g_0, ..., g_{T-1}$ is used in the update

**Inefficiency 2**

Then sample $\tau$, select 1 $g_{t(k)}$
and update $\theta^{k+1} = \theta^k - \alpha_k \sum_{t=0}^T g_{t(k)}$

It is an instance of SGD with unbiased stochastic gradients, but it is an inefficient algorithm as it updates infrequently. (In Supervised Learning, full-batch GD is less efficient than SGD)

**Finally**
Sample $(s_t, a_t, r_t, s_{t+1})$ and compute $g_t$. Sample $k$ steps if $k-step$ TD
 $\theta^{k+1} = \theta^k - \alpha_k g_{t}$
is not an instance of SGD, but it updates the policy frequently and therefore is efficient despite using biased gradients. It is analogaus to **Cyclic SGD**

# A2C: Advantage actor-critic

Sample a trajectory for k-steps
Calculate the Q-estimates for the k-steps going backwards
Update $\theta$ and $\phi$ parameters

## A2C for Discrete Action
For a neural network that takes state $s_t$ and outputs a probability distribution over action-space

**Step 1**
$a_t \sim \pi(.|s_t)$
Evaluate $f_{\theta}(s_t)$ and sample $a_j$ with probability $(f_\theta(s_t))_j$ for $j = 1, 2, ..., |\mathcal{A}|$

**Step 2**
$\nabla_{\theta} \log \pi_{\theta}(a_t|s_t)$: Backprop on $\log ((f_{\theta}(s_t))_{a_t})$

## A2C for Continuous Action

A2C applies to continuous actions. There are 2 steps to clarify
As a concrete instance, let $\mathcal{A} \subseteq [-1, +1]$ be a continuous action space. 
Make a neural network to predict mean and variance, $f_{\theta}(s) = (\mu_{\theta}(s), \tau_{\theta}(s))$

**Step 1**
$a_t \sim \pi(.|s_t)$
Sample $a_t = \tanh(z_t)$
with $z_t \sim \mathcal{N}(\mu_{\theta}(s_t), \text{variance}=e^{2\tau_{\theta}(s_t)})$

**Step 2**
$\nabla_{\theta} \log \pi_{\theta}(a_t|s_t)$
Backpropagate on 
$$
\log \pi_{\theta}(a_t|s_t) = -\tau_{\theta}(s) - \frac{(\tanh^{-1}(a_t) - \mu_{\theta}(s_t))^2}{2e^{2\tau_{\theta}(s)}} + C
$$ 

# Surrogate Objective - Professor KV's Interpretation

We seek to optimize $\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{s \sim d^{\pi}, a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta (a|s)A^{\pi_\theta}(s,a)]$
Expectation, as we could see, is under $\pi_\theta$

However,
$s,a \sim \pi_\text{old}$
1. we collect data
2. _store_: actions, states, rewards
3. calculate $\log \pi_\theta (a|s)$
4. Now optimize for a batch of samples in multiple rounds of updates
5. $\theta_k \rightarrow \theta^{(1)}_{k} \rightarrow \theta^{(2)}_k \rightarrow ...$
6. Data still from $\pi_\text{old}$

But policy is changing! We are updating in every batch.
If we were to sample new data, after every optimization step, then we will be F'ed.

**Bring in Importance Sampling**
$$
\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{s \sim d^{\pi}, a \sim \pi_\text{old}} [\frac{\pi_\theta(a|s)}{\pi_\text{old}(a|s)} \nabla_\theta \log \pi_\theta (a|s)A^{\pi_\theta}(s,a)]
$$
**Define Surrogate Objective**

$L(\theta) = \mathbb{E}_{\pi_\text{old}}[\frac{\pi_\theta(a|s)}{\pi_\text{old}(a|s)} A(s, a)]$

**Surrogate is the First-Order True Approximation of True Objective**
$$
\begin{align*}
\nabla_\theta L(\theta) &= \mathbb{E}_{\pi_\text{old}}[\frac{\nabla_{\theta} \pi_\theta(a|s)}{\pi_\text{old}(a|s)} A(s, a)] \\
\nabla_\theta L(\theta) &= \mathbb{E}_{\pi_\text{old}}[\frac{\pi_\theta(a|s)}{\pi_\text{old}(a|s)} \nabla_\theta \log \pi_\theta(a|s) A(s, a)] \\

\nabla_\theta L(\theta) &= \mathbb{E}_{\pi_\text{old}}[r(\theta) \nabla_\theta \log \pi_\theta(a|s) A(s, a)] \\
\text{At } \theta_\text{old} &= \theta, r(\theta) = 1 \\
\nabla_\theta L(\theta) &= \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) A(s, a)]
\end{align*}
$$

>This is the same as the original objective
>****ONLY NEAR OLD POLICY****

## **Important Identity**

Define $\eta(\pi) = \mathbb{E}_{s_0, a_0, ..., } [\sum_{t=0}^\infty \gamma^t r(s_t)]$
$$
\eta(\tilde{\pi}) - \eta({\pi}) = \mathbb{E}_{s_0, a_0, ..., \sim \tilde{\pi}} [\sum_{t=0}^{\infty}\gamma^t A_{\pi}(s_t, a_t)]
$$
where $\mathbb{E}_{s_0, a_0, ..., \sim \tilde{\pi}}$ indicate that actions are sampled $a_t \sim \tilde{\pi}(.|s_t)$

**Proof**

First note that $A_{\pi}(s, a) = E_{s^{'} \sim P(s^{'}|s,a)}[r(s) + \gamma V_{\pi}(s^{'}) - V_{\pi}(s)]$
Therefore
$$
\begin{align*}
\mathbb{E}_{\tau | \tilde{\pi}} [\sum_{t=0}^{\infty} \gamma^t A_{\pi}(s_t, a_t)] &= \mathbb{E}_{\tau | \tilde{\pi}} [\sum_{t=0}^{\infty} \gamma^t (r(s_t) + \gamma V_{\pi}(s^{'}) - V_{\pi}(s))] \\
&= \mathbb{E}_{\tau | \tilde{\pi}} [-V_{\pi}(s_0) + \sum_{t=0}^{\infty} \gamma^t r(s_t)] \\
&= -\mathbb{E}_{s_0}[V_{\pi}(s_0)] + \mathbb{E}_{\tau | \tilde{\pi}} [\sum_{t=0}^{\infty} \gamma^t r(s_t)] \\
&= -\eta(\pi) + \eta(\tilde{\pi})
\end{align*}
$$

Therefore,
$$
\eta(\tilde{\pi}) = \eta({\pi}) + \mathbb{E}_{s_0, a_0, ..., \sim \tilde{\pi}} [\sum_{t=0}^{\infty}\gamma^t A_{\pi}(s_t, a_t)]
$$
whereas
$$
L_{\pi}(\tilde{\pi}) = \eta({\pi}) + \mathbb{E}_{\tau \sim {\pi}} [\sum_{t=0}^{\infty}\gamma^t A_{\pi}(s_t, a_t)]
$$

The difference in these equations is whether the states are sampled using $\pi$ or $\tilde{\pi}$
and we can bound their difference

## Brining in Importance Sampling

$$
\begin{align*}
\mathcal{J}(\theta) - \mathcal{J}(\theta_0) &= \mathbb{E}_{\tau \sim (p_0, \pi_\theta, p)} [\sum_{t=0}^{T-1}\gamma^t A_{\pi_{\theta_0}}(s_t, a_t)]] \\
&= \mathbb{E}_{s_0 \sim p_0, a_0 \sim \pi_\theta(.|s_0)} [A^{\pi_{\theta_0}}(s_0, a_0) + \mathbb{E}^{\pi_\theta} [\sum_{t=0}^{T-1}\gamma^t A_{\pi_{\theta_0}}(s_t, a_t)|s_0, a_0]] \\
&= \mathbb{E}_{s_0 \sim p_0, a_0 \sim \pi_\theta(.|s_0), a_0^{'} \sim \pi_{\theta_0}(.|s_0)} [\frac{\pi_{\theta}(a^{'}_0|s_0)}{\pi_{\theta_0}(a_0^{'}|s_0)}A^{\pi_{\theta_0}}(s_0, a_0^{'}) + \mathbb{E}^{\pi_\theta} [\sum_{t=0}^{T-1}\gamma^t A_{\pi_{\theta_0}}(s_t, a_t)|s_0, a_0]] \\
&= \mathbb{E}_{\tau \sim (p_0, \pi_{\theta}, p), a_t^{'} \sim \pi_{\theta_0}(.|s_t)} [\sum_{t=0}^{T-1}\gamma^t \frac{\pi_{\theta}(a^{'}_t|s_t)}{\pi_{\theta_0}(a_t^{'}|s_t)}A^{\pi_{\theta_0}}(s_t, a_t^{'}) ]
\end{align*}
$$

Therefore,

$$
\begin{align*}
\mathcal{J}(\theta) = \mathcal{J}(\theta_0) + \mathbb{E}_{\tau \sim (p_0, \pi_{\theta}, p), a_t^{'} \sim \pi_{\theta_0}(.|s_t)} [\sum_{t=0}^{T-1}\gamma^t \frac{\pi_{\theta}(a^{'}_t|s_t)}{\pi_{\theta_0}(a_t^{'}|s_t)}A^{\pi_{\theta_0}}(s_t, a_t^{'}) ]
\end{align*}
$$

## The Inconvenience

Expectation $\mathbb{E}$ depends on $\theta$ and this is inconvenient because:
$$
\begin{align*}
\mathbb{E}_{s_0, a_0, ..., \sim \tilde{\pi}} [\sum_{t=0}^{\infty}\gamma^t A_{\pi}(s_t, a_t)] &= \sum_{t=0}^{\infty}\sum_s P(s_t = s|\tilde{\pi}) \sum_a \tilde{\pi}(a|s) \gamma^t A_{\pi}(s, a) \\
&= \sum_s \sum_{t=0}^{\infty} \gamma^t P(s_t = s|\tilde{\pi}) \sum_a \tilde{\pi}(a|s) A_{\pi}(s, a) \\
&= \sum_s \rho_{\tilde{\pi}}(s) \sum_a \tilde{\pi}(a|s) A_{\pi}(s, a)
\end{align*}
$$
where $\rho_{\pi}(s) = P(s_0 = s) + \gamma P(s_1 = s) + \gamma^2 P(s_2 = s) + ...$, the discounted visitation frequency. 

Therefore, the complex dependency of $\rho_{\tilde{\pi}}(s)$ on $\tilde{\pi}$ makes it difficult to optimize the above equation.

$$
L_{\pi}(\tilde{\pi}) = \eta({\pi}) + \mathbb{E}_{\tau \sim {\pi}} [\sum_{t=0}^{\infty}\gamma^t A_{\pi}(s_t, a_t)]
$$

$L_{\pi}$ is a local approximation to $\eta$, which uses the visitation frequency to $\rho_\pi$
rather than $\rho_{\tilde{\pi}}$ ignoring changes in state visitation density due to changes in the policy.
If policy is parameterized, 
$$
L_{\pi_{\theta_0}}(\pi_{\theta_0}) = \eta(\pi_{\theta_0})
$$
$$
\nabla_\theta L_{\pi_{\theta_0}}(\pi_{\theta})|_{\theta = \theta_0} = \nabla_\theta \eta(\pi_{\theta})|_{\theta=\theta_0}
$$
## Back to Ernest

Thus, Define $\mathcal{K}(\theta, \theta_0)$ which is exactly same as $L_{\pi_{\theta_0}}$ as defined in the TRPO paper.

$$
\begin{align*}
\mathcal{K}(\theta, \theta_0) = \mathbb{E}_{\tau \sim (p_0, \pi_{\theta_0}, p), a_t^{'} \sim \pi_{\theta_0}(.|s_t)} [\sum_{t=0}^{T-1}\gamma^t \frac{\pi_{\theta}(a^{'}_t|s_t)}{\pi_{\theta_0}(a_t^{'}|s_t)}A^{\pi_{\theta_0}}(s_t, a_t^{'}) ] + C
\end{align*}
$$

Then $\mathcal{J}(\theta) \approx \mathcal{K}(\theta, \theta_0)$ for $\theta \approx \theta_0$ 
Here $C$ represents constants independent of $\theta$

At $\theta = \theta_0$, in first-order they are equal, but in second-order they are unequal.
$\nabla^2 \mathcal{J}(\theta)|_{\theta = \theta_0} \neq \nabla^2 \mathcal{K}(\theta)|_{\theta = \theta_0}$

## Sampling Trajectories

Now we can safely use the trajectories from old policy to update the parameters of the changing current policy.

For $\theta = \theta_0$, if we sample IID trajectories $\tau^{(1)}, ..., \tau^{(N)} \sim (p_0, \pi_{\theta_0}, p)$, then
$$
\begin{align*}
\mathcal{K}(\theta, \theta_0) &= \mathbb{E}_{\tau \sim (p_0, \pi_{\theta_0}, p)} [\sum_{t=0}^{T-1}\gamma^t \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_0}(a_t|s_t)}A^{\pi_{\theta_0}}(s_t, a_t) ] \\
&\mathop{\approx}^{(*)} \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T^{(i)}-1} \gamma^t  \frac{\pi_{\theta}(s_t^{(i)}, a_t^{(i)})}{\pi_{\theta_0}(s_t^{(i)}, a_t^{(i)})}\hat{A}_t^{(i)}
\end{align*}
$$

where $\hat{A}_t \approx A^{\pi_{\theta_0}}(s_t,a_t)$ is an advantage estimate.

The approximation $(*)$ is accurate if $N$ is large and when $\pi_{\theta} \approx \pi_{\theta_0}$ (In general, importance sampling estimators become inaccurate when the sampling distribution $\pi_{\theta_0}$ is too far from the nominal distribution $\pi_\theta$

We therefore maximize the _surrogate objective_ to a certain trust-region constraint

$$
\mathop{\text{maximize}}_{\theta \in \mathbb{R}^p} \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T^{(i)}-1} \gamma^t  \frac{\pi_{\theta}(s_t^{(i)}, a_t^{(i)})}{\pi_{\theta_0}(s_t^{(i)}, a_t^{(i)})}\hat{A}_t^{(i)} \text{ subject to } \theta \text{ and } \theta_0 \text{ close}
$$

The trust-region constraint is needed for 2 reasons: $\mathcal{K}(\theta; \theta_0) \approx \mathcal{J}(\theta)$ and $(*)$

# Surrogate Objective - Ernest's Interpretation

## Sample Efficiency

For RL, in simulated environments sample efficiency (the ability to learn with fewer data points) is not a concern and only compute efficiency matters.

But, in many problems, samples are obtained my interacting with environment and therefore can be expensive.

Ex:
1. Having a physical robot take certain actions
2. LLM writing a completion and having human provide feedback on whether the completion is good.

## Can we learn from episode $\tau$?

A2C uses an episode $\tau$ to perform $T$ SGD updates. Can we learn more from $\tau$?
Can we be more sample efficient?

We do so via _Surrogate Objective_


# TRPO: Trust-Region Policy Optimization

TRPO solves a _sequence_ of trust-region optimization problems to improve the policy

The trust region is defined by the KL-Divergence of the Policies.

Note that $\hat{A}_t = \hat{Q_t} - V_\phi(s_t)$ depends on $\theta_\text{curr}$ through $\hat{Q_t}$ and $\phi$ through $V_{\phi}$
(we discuss the choice of $\hat{A_t}$ soon)

![[Pasted image 20260331095749.png]]


