[[4782 Guide + Topics]]

# Recap: Generative Models

What should be clear at this point is that generative modeling is a fundamentally challenging task.

**Generative Adversarial Networks (GANs)** train a generator to synthesize data that fools a discriminator trained to distinguish between real and fake samples. This adversarial setup has led to **remarkable sample quality**, especially in high-resolution image generation. However, GANs come with their own challenges: **mode collapse**, **unstable training**, and the lack of an **explicit density model** make them difficult to train and evaluate probabilistically.

With Variational Autoencoders (VAEs), we approached this challenge as an encoder-decoder problem: mapping an input $x$ to a latent representation $z = q_{\phi}(x)$ using a learned encoder, and then reconstructing the input via $x \approx p_{\theta}(z)$. While VAEs offer fast inference, they often struggle with generating high-quality, detailed samples. 

So we’re left with a tradeoff:
- VAEs: stable training, but blurry generations.
- GANs: sharp images, but unstable and non-probabilistic.

However, both are lightning fast. Perhaps this is part of the issue.
For example, for the VAE, the one-shot nature of both encoding and decoding makes it so that we have to compress the input into a latent space in a single step and attempt to reconstruct it just as quickly. This abruptness can limit its expressivity. 

So what if we slowed things down? What if we transformed data into noise—and noise back into data—_gradually_?

This is the key intuition behind diffusion models, which were introduced in the previous lecture.
# Recap: Denoising Diffusion Models

The diffusion models we study consist of two parts, a *forward diffusion process* and a *reverse denoising process*. 
- The forward diffusion process gradually adds noise to the input.
- The reverse denoising process learns to undo the noise at each step to reconstruct data.

Each process occurs over several time steps. 

## Forward Process

The forward process $q$ defines a Markov chain that adds Gaussian noise to the original data $\mathbf{x}_0$ over $T$ steps to get increasingly noisy versions $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$, until such a point that $x_T \approx N(0,I)$. 

While we can iteratively add noise $\epsilon \sim N(0,I)$ to each $x_t$, it is advantageous to imagine a closed form. That is:

$$q_{(x_t | x_0) = N(\sqrt{\bar{\alpha_t}}x_0, (1-\bar{\alpha}_t)I)}$$
We can compute this as 
$$x_t = \sqrt{\bar{\alpha_t}}x_0 + (1-\bar{\alpha}_t)\epsilon,\epsilon \sim N(0,I) $$
where $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$, with $\alpha_t \in (0,1)$ defines the noise schedule.

### Noise Schedules

In diffusion models, noise schedules determine how much Gaussian noise is added to a clean data point (i.e. image) at each step of the forward diffusion process.

In doing so, the noise schedule defines **how quickly or slowly** this corruption happens.

## Reverse Process
The forward process adds noise step-by-step to get $x_T$. It destroys the original image $x_0$. To be able to generate new data, we need to build a model that reconstructs data starting from pure noise. 

The reverse process does exactly this: it learns to denoise from $x_T$ to $x_0$ using a learned generative model. 

We model the reverse process as a Markov chain of Gaussians.

**Final timestep distribution:**

We assume the final noisy image is standard Gaussian:

$$
p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})
$$

This serves as the starting point for generation. Now, we want our model to learn to denoise each timestep, until we are back at a good reconstruction of the original input data $x_0$. 
Formally, this idea is captured by
$$
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})
$$
where
- The mean $\mu_\theta(\mathbf{x}_t, t)$ is predicted by a neural network.
- The variance $\sigma_t^2$ is often fixed or learned, depending on the implementation.

In other words, we are learning to model the conditional distribution of a cleaner image given a noisier one.
The full generative model is a chain of these reverse steps:
$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)
$$
Of course, the challenge lies in learning $p_{\theta}$, and learning it *well*. 
# Diffusion Learning Objective

Our objective is actually quite similar to the VAE. In an ideal world, we'd match the distribution $p(x)$ from which the data is drawn from. Of course, this is usually not tractable.

Instead, let's see if we can come up with an Evidence Lower Bound that *is tractable*. That is, we want to come up with some lower bound on $log(p(x))$ that we can seek to maximize with a model. 

## ELBO Derivation 

1.
We want the marginal likelihood of our data point $x_0$ given a latent noise sequence $x_{1:T}​$.
$$
\log p(x_0)
= \log \int p(x_{0:T}) \,dx_{1:T}
\quad\text{where}\quad
p(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t).
$$

Next, we introduce a variational posterior (an identity term here) and recognize the expression as an expectation over the forward process.

$$
\log p(x_0)
= \log \int p(x_{0:T})\frac{q(x_{1:T}\mid x_0)}{q(x_{1:T}\mid x_0)}\,dx_{1:T}
= \log \mathbb{E}_{q(x_{1:T}\mid x_0)}\!\Bigl[\tfrac{p(x_{0:T})}{q(x_{1:T}\mid x_0)}\Bigr].
$$

Since $\log$ is concave, we can apply Jensen's Inequality
$$
\log p(x_0)
\;\ge\;
\mathbb{E}_{q(x_{1:T}\mid x_0)}\!\Bigl[\log\tfrac{p(x_{0:T})}{q(x_{1:T}\mid x_0)}\Bigr]
\;\equiv\;
\mathcal{L}(x_0).
$$

Next, we plug in the forward and reverse factors we derived. 


$p(x_{0:T})=p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t)$ for the forward process.
and 
$q(x_{1:T}\mid x_0)=\prod_{t=1}^T q(x_t\mid x_{t-1})$ for the reverse process.

With some algebra, we get
$$
\mathcal{L}(x_0)
= \mathbb{E}_{q}
\Bigl[\log p(x_T)
+ \sum_{t=1}^T \log p_\theta(x_{t-1}\mid x_t)
- \sum_{t=1}^T \log q(x_t\mid x_{t-1})
\Bigr].
$$

Subsequently, we re-index and collect terms into: 
1. Reconstruction
   $$
   \mathbb{E}_{q(x_1\mid x_0)}\bigl[\log p_\theta(x_0\mid x_1)\bigr].
   $$
This term assess to what degree our learned $p_{\theta}$ allows us to recover the original data $x_0$. 


2. **Prior matching**: 
   $$
   -\mathbb{E}_{q(x_{T-1}\mid x_0)}\bigl[D_{\mathrm{KL}}(q(x_T\mid x_{T-1})\|p(x_T))\bigr].
   $$
This term assesses our ability to fully destroy the input image (that is, make it so that $p(x_T) = N(0,I)$).

3. **Consistency/ de-noising matching term**: 
   $$
   -\sum_{t=1}^{T-1}
   \mathbb{E}_{q(x_{t-1},x_{t+1}\mid x_0)}
   \bigl[D_{\mathrm{KL}}(q(x_t\mid x_{t-1})\|p_\theta(x_t\mid x_{t+1}))\bigr].
   $$
Finally, the last term is the objective we strive to learn. It asks,  "for all $t$ up to $T$, if we start from $x_0$ and add noise for $t$ steps using the forward process $q$, to what degree can we recover $x_{t-1}$ using our learned reverse model $p_{\theta}$?"
In other words, it ensures that each reverse denoising step matches the corresponding forward noise step.

Hence,
$$
\boxed{
\log p(x_0)\ge
\underbrace{\mathbb{E}_{q(x_1\mid x_0)}[\log p_\theta(x_0\mid x_1)]}_{\text{reconstruction}}
\;-\;
\underbrace{\mathbb{E}_{q(x_{T-1}\mid x_0)}\bigl[D_{\mathrm{KL}}(q(x_T\mid x_{T-1})\|p(x_T))\bigr]}_{\text{prior matching}}
\;-\;
\sum_{t=1}^{T-1}
\underbrace{\mathbb{E}_{q(x_{t-1},x_{t+1}\mid x_0)}\bigl[D_{\mathrm{KL}}(q(x_t\mid x_{t-1})\|p_\theta(x_t\mid x_{t+1}))\bigr]}_{\text{consistency}}
}
$$

We take the denoising matching term as our learning objective. 

This is the **only** non-zero, parameterized term in our ELBO. **Minimizing it** forces the learned reverse kernels   $p_{\theta}(x_{t-1}|x_t)$ to match the true posteriors $q(x_{t-1}|x_t,x_0)$ at every timestep.

In fact, observe that $q(x_{t-1} | x_t,x_0)$ is Gaussian and can be expressed as:
$$
q(x_{t-1}\mid x_t, x_0)
\;=\;
\mathcal{N}\!\bigl(x_{t-1};\,\tilde\mu_t(x_t, x_0),\,\tilde\beta_t\,I\bigr),
$$

Recall that we defined the noise‐scheduling hyperparameters as
$$
\begin{aligned}
\beta_t &\;:\;\text{preset variance at step }t,\\
\alpha_t &:= 1 - \beta_t,\\
\bar\alpha_t &:= \prod_{s=1}^t \alpha_s,\\
\bar\alpha_{t-1} &:= \prod_{s=1}^{t-1} \alpha_s,
\end{aligned}
$$

and can then write the closed-form posterior variance and mean as
$$
\begin{aligned}
\tilde\beta_t
&:= \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\,\beta_t,\\[6pt]
x_t
&= \sqrt{\bar\alpha_t}\,x_0 \;+\;\sqrt{1 - \bar\alpha_t}\;\epsilon,\quad
\epsilon\sim\mathcal{N}(0,I),\\[6pt]
\tilde\mu_t(x_t,x_0)
&:= \frac{1}{\sqrt{\alpha_t}}
   \Bigl(x_t \;-\;\frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\,\epsilon\Bigr).
\end{aligned}
$$

# Parameterizing the Denoising Model

Since both  
$$q(x_{t-1}\mid x_t, x_0)$$  
and  
$$p_\theta(x_{t-1}\mid x_t)$$  
are Gaussian, their KL divergence reduces to a simple squared‐difference between means:

$$
L_{t-1}
\;=\;
D_{KL}\bigl(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\bigr)
\;=\;
\mathbb{E}_{q}\!\biggl[\frac{1}{2\sigma_t^2}\,\bigl\|\tilde\mu_t(x_t,x_0)\;-\;\mu_\theta(x_t,t)\bigr\|^2\biggr]
\;+\;C.
$$
Here, $\sigma_t^2$ is the (shared) variance of both Gaussians (often set to $\tilde\beta_t$).  
The constant $C$ absorbs all terms independent of $\theta$.

Recall from the forward process that we define the noised data at time \(t\) by
$$
x_t \;=\; \sqrt{\bar\alpha_t}\,x_0 \;+\;\sqrt{1-\bar\alpha_t}\,\epsilon,
\quad
\epsilon\sim\mathcal{N}(0,I),
$$
where
$$
\alpha_t \;=\; 1 - \beta_t,
\quad
\bar\alpha_t \;=\;\prod_{s=1}^t \alpha_s.
$$

By Bayes’ rule on two Gaussians, the true posterior  
$$q(x_{t-1}\mid x_t,x_0)$$  
is also Gaussian with
$$
\tilde\mu_t(x_t,x_0)
\;=\;
\frac{1}{\sqrt{1-\beta_t}}
\Bigl(x_t \;-\;\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\Bigr),
\quad
\tilde\beta_t
=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t.
$$
Thus, we can seek to predict this $\tilde\mu_t$. But instead of doing this directly Ho et al. proposed letting the network predict the noise $\epsilon$ instead.
That is, training a network that would model
   $$
   \epsilon_\theta(x_t, t)\;\approx\;\epsilon.
   $$
such that we could then model $\tilde \mu_t$ as 
   $$
   \mu_\theta(x_t,t)
   =\frac{1}{\sqrt{1-\beta_t}}
   \Bigl(x_t \;-\;\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Bigr).
   $$

Finally, we can substitute this $\mu_\theta$ into the KL to obtain a mean‐squared error on the noise:

$$
L_{t-1}
=\;\mathbb{E}_{x_0\sim q(x_0),\,\epsilon\sim\mathcal{N}(0,I)}\!\Biggl[
\frac{\beta_t^2}{2\,\sigma_t^2\,(1-\beta_t)\,(1-\bar\alpha_t)}
\bigl\|\epsilon \;-\;\epsilon_\theta(\underbrace{\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon}_{x_t},\,t)\bigr\|^2
\Biggr]
\;+\;C.
$$

In practice, the collected constants are often dropped from the training objective: 
$$
L_{t-1}
=\;\mathbb{E}_{x_0\sim q(x_0),\,\epsilon\sim\mathcal{N}(0,I)}\!\Biggl[
\bigl\|\epsilon \;-\;\epsilon_\theta(\underbrace{\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon}_{x_t},\,t)\bigr\|^2
\Biggr]
\;+\;C.
$$
This is actually quite interpretable. We train by taking the mean squared error between the learned and true noise at each step!

# Implementation Considerations: Network Architectures

Diffusion models typically use a U-Net backbone with ResNet blocks and self-attention to implement the noise-prediction network $\epsilon_\theta(x_t, t)$. You may recall the U-Net architecture from our lecture on Modern Vision Networks.
### U-Net Structure  
- **Encoder (down-sampling path)**  
  - Sequential ResNet blocks reduce spatial resolution  
  - Optionally interleave self-attention at intermediate resolutions to capture long-range dependencies  
- **Bottleneck**  
  - Lowest resolution features processed by ResNet + attention  
- **Decoder (up-sampling path)**  
  - Mirror of encoder: ResNet blocks + up-sampling (nearest or transpose-convolution)  
  - Skip-connections from encoder layers inject high-resolution detail. This is a critical features that enables high quality reconstructions.

### The Time Factor 

Diffusion models have the consideration of time *t* that needs to be preserved and represented. For this, we borrow sinusoidal positional embeddings from the transformer. 

### Self-Attention

Within the U-Net, we place self-attention layers at one or more resolutions (often in high level feature maps) to help model global correlations.

The aforementioned architecture is defined in **Ho et al. (NeurIPS 2020)** , who proposed the original DDPM U-Net, and **Dhariwal & Nichol (NeurIPS 2021)**, who made crucial improvements. 


