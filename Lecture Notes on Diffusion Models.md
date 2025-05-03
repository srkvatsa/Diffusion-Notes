


With Variational Auto-encoders, we explored the task as an encoder-decoder problem, mapping the input $x$ to a latent representation $z = q_{\phi}(x)$, where $q_{\phi}$ is the learned encoder, and then attempting to reconstruct $x$ with $x= p_{\theta}(x)$. 


However, we ran into issues with the quality of image generation. Though these models are fast, that makes them incredibly challenging to use in practice.

Part of the problem, in some sense, is that we are encoding and decoding too *quickly*. We want to get a latent representation in just one step, and then a reconstruction in just one step. 

What if we did these processes gradually?