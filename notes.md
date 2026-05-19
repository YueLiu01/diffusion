# Correct SEDD Formulation for Measurement-Altered Criticality

## Purpose

This note specifies the correct Score-Entropy Discrete Diffusion (SEDD) formulation for diagonal weak-measurement protocols in measurement-altered criticality. It is intended as an implementation guide for humans or AI coding agents.

The key principle is:

```math
\boxed{
\text{The clean snapshot } z \text{ is used to generate training targets, but it is not an input to the score network.}
}
```

The score network takes only the corrupted/noisy configuration and the noise level.

---

## 1. Basic variables

We consider spin configurations

```math
z,x,s\in\{\pm1\}^L .
```

The meanings are:

| Symbol | Meaning |
|---|---|
| $z$ | Clean projective snapshot sampled from the critical quantum state |
| $x$ | Noisy/corrupted configuration used as the SEDD diffusion variable |
| $s$ | Weak-measurement record at a particular measurement strength |
| $\beta$ | Weak-measurement strength |
| $\tau$ | Noise/SNR parameter, $\tau=\tanh(2\beta)$ |
| $F_i x$ | Configuration obtained from $x$ by flipping site $i$ |

The clean data distribution is

```math
z\sim p_0(z),
```

where $p_0(z)$ is the projective snapshot distribution of the initial state, for example the critical TFIM ground state in the $Z$-basis.

---

## 2. Weak-measurement channel as a discrete corruption process

For diagonal site-independent weak measurements, the weak record is generated from the clean snapshot through a binary channel:

```math
q_\beta(s|z)
=
\prod_{i=1}^L
\frac{1+\tau s_i z_i}{2},
\qquad
\tau=\tanh(2\beta).
```

Equivalently,

```math
s_i=z_i\eta_i,
```

where

```math
\Pr(\eta_i=+1)=\frac{1+\tau}{2},
\qquad
\Pr(\eta_i=-1)=\frac{1-\tau}{2}.
```

Thus the bit-flip probability is

```math
p_{\mathrm{flip}}(\beta)
=
\frac{1-\tanh(2\beta)}{2}.
```

The limits are:

```math
\beta=0 \Longleftrightarrow \tau=0:
\quad
s \text{ is pure noise},
```

```math
\beta\to\infty \Longleftrightarrow \tau\to1:
\quad
s=z.
```

Thus $\beta$ is an inverse noise scale, while $\tau$ is a signal-to-noise parameter.

---

## 3. Noisy marginal distribution

For a given noise level $\tau$, define the noisy marginal distribution

```math
p_\tau(x)
=
\sum_z p_0(z)q_\tau(x|z),
```

where

```math
q_\tau(x|z)
=
\prod_i \frac{1+\tau x_i z_i}{2}.
```

The SEDD score network learns local probability ratios of this noisy distribution:

```math
R_i^\star(x,\tau)
=
\frac{p_\tau(F_i x)}{p_\tau(x)}.
```

Equivalently, the network predicts log-ratios

```math
u_{\theta,i}(x,\tau)
\approx
\log R_i^\star(x,\tau)
=
\log \frac{p_\tau(F_i x)}{p_\tau(x)}.
```

The network output has shape $[B,L]$, one log-ratio per site.

---

## 4. Correct network input

The score network input is

```math
\boxed{
(x,\tau)
}
```

or equivalently

```math
\boxed{
(x,\beta)
}
```

where $x$ is the current noisy/corrupted spin configuration.

The clean snapshot $z$ is **not** an input to the neural network.

Correct:

```python
u = model(x, tau)      # or model(x, beta)
```

Incorrect:

```python
u = model(z, x, tau)   # wrong: leaks the clean sample
```

The clean snapshot $z$ appears only in the training loss because it is needed to construct the denoising target.

---

## 5. SEDD denoising target

For a training pair

```math
z\sim p_0(z),
\qquad
x\sim q_\tau(x|z),
```

the known single-site forward-channel ratio is

```math
a_i(x,z;\tau)
=
\frac{q_\tau(F_i x|z)}{q_\tau(x|z)}.
```

For the independent binary channel,

```math
a_i(x,z;\tau)
=
\frac{1-\tau x_i z_i}{1+\tau x_i z_i}.
```

In terms of measurement strength $\beta$,

```math
a_i(x,z;\beta)
=
\frac{1-\tanh(2\beta)x_i z_i}
     {1+\tanh(2\beta)x_i z_i}
=
e^{-4\beta x_i z_i}.
```

This target uses the clean $z$, but $z$ is not fed into the score network.

---

## 6. SEDD loss

Let

```math
R_{\theta,i}(x,\tau)=e^{u_{\theta,i}(x,\tau)}.
```

The SEDD score-entropy loss is

```math
\mathcal L_{\mathrm{SEDD}}(\theta)
=
\mathbb{E}_{z,\tau,x}
\left[
\frac1L\sum_{i=1}^L
\left(
e^{u_{\theta,i}(x,\tau)}
-
a_i(x,z;\tau)u_{\theta,i}(x,\tau)
\right)
\right],
```

where

```math
z\sim p_0(z),
\qquad
\tau\sim \rho(\tau),
\qquad
x\sim q_\tau(x|z).
```

At the optimum,

```math
e^{u_{\theta,i}(x,\tau)}
=
\mathbb{E}[a_i(x,z;\tau)|x,\tau]
=
\frac{p_\tau(F_i x)}{p_\tau(x)}.
```

Therefore the trained network learns the local probability ratio of the noisy distribution.

---

## 7. Derivation of the optimum

For fixed $(x,\tau)$, the loss contribution for site $i$ is

```math
\mathbb{E}\left[
R_i-a_i\log R_i
\mid x,\tau
\right],
```

where

```math
R_i=R_{\theta,i}(x,\tau).
```

Differentiating with respect to $R_i$ gives

```math
1-\frac{\mathbb{E}[a_i|x,\tau]}{R_i}=0.
```

Thus

```math
R_i^\star(x,\tau)
=
\mathbb{E}[a_i(x,z;\tau)|x,\tau].
```

Now,

```math
\begin{aligned}
\mathbb{E}[a_i|x,\tau]
&=
\sum_z p(z|x,\tau)
\frac{q_\tau(F_i x|z)}{q_\tau(x|z)}
\\
&=
\sum_z
\frac{p_0(z)q_\tau(x|z)}{p_\tau(x)}
\frac{q_\tau(F_i x|z)}{q_\tau(x|z)}
\\
&=
\frac{1}{p_\tau(x)}
\sum_z p_0(z)q_\tau(F_i x|z)
\\
&=
\frac{p_\tau(F_i x)}{p_\tau(x)}.
\end{aligned}
```

This is the SEDD denoising trick: the true noisy-distribution ratio is unknown, but the conditional corruption ratio is known and gives the correct target after averaging over clean data.

---

## 8. Training algorithm

At each optimization step:

1. Sample a minibatch of clean snapshots

   ```math
   z\sim p_0(z).
   ```

2. Sample noise levels $\tau$ or $\beta$.

3. Generate noisy configurations

   ```math
   x\sim q_\tau(x|z).
   ```

4. Compute the known target ratios

   ```math
   a_i(x,z;\tau)
   =
   \frac{1-\tau x_i z_i}{1+\tau x_i z_i}.
   ```

5. Feed only $(x,\tau)$ into the network.

6. Minimize

   ```math
   \frac1L\sum_i
   \left[
   e^{u_{\theta,i}(x,\tau)}
   -
   a_i(x,z;\tau)u_{\theta,i}(x,\tau)
   \right].
   ```

---

## 9. Minimal PyTorch-style training step

```python
import torch


def training_step(model, optimizer, z, beta_sampler, eps=1e-6, logu_clip=None):
    """
    z: clean snapshots, shape [B, L], values in {-1, +1}
    model input: (x, beta) or (x, tau)
    model output: u, shape [B, L], log probability ratios
    """
    B, L = z.shape
    device = z.device

    # 1. Sample measurement/noise strength.
    beta = beta_sampler(B).to(device)          # shape [B, 1] or [B]
    if beta.ndim == 1:
        beta = beta[:, None]

    tau = torch.tanh(2.0 * beta)               # shape [B, 1]

    # 2. Generate noisy configuration x from clean z.
    p_same = (1.0 + tau) / 2.0                 # shape [B, 1]
    same = torch.rand_like(z.float()) < p_same
    x = torch.where(same, z, -z)

    # 3. Compute known SEDD target ratio.
    xz = x * z
    numerator = 1.0 - tau * xz
    denominator = 1.0 + tau * xz

    # Numerical safety.
    numerator = numerator.clamp_min(eps)
    denominator = denominator.clamp_min(eps)

    a = numerator / denominator

    # 4. Score network input does NOT include z.
    u = model(x, beta)                         # or model(x, tau)

    if logu_clip is not None:
        u = torch.clamp(u, -logu_clip, logu_clip)

    # 5. SEDD score-entropy loss.
    loss = (torch.exp(u) - a * u).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()
```

The core implementation rule is:

```python
u = model(x, beta)
```

not

```python
u = model(z, x, beta)
```

---

## 10. Sampling / reverse denoising

After training, the network provides ratios

```math
R_{\theta,i}(x,\tau)
=
\exp u_{\theta,i}(x,\tau)
\approx
\frac{p_\tau(F_i x)}{p_\tau(x)}.
```

To generate clean samples from noise, use the learned ratios to run the reverse discrete diffusion process from a smaller-$\tau$ noisy distribution toward $\tau=1$.

For measurement-altered criticality at a particular measurement strength $\beta_0$, define

```math
\tau_0=\tanh(2\beta_0).
```

Given an observed or synthetically generated weak record

```math
s\sim p_{\tau_0}(s),
```

initialize

```math
x_{\tau_0}=s.
```

Then run the reverse denoising dynamics from $\tau_0$ to $\tau=1$, using the learned ratios at the current state $x$ and current noise level $\tau$. The final state is an approximate sample from

```math
p(z|s,\tau_0).
```

Schematically,

```math
x_{\tau_0}=s
\longrightarrow
x_{\tau_1}
\longrightarrow
\cdots
\longrightarrow
x_{\tau=1}\approx z.
```

The score network is evaluated as

```math
u_\theta(x,\tau),
```

where $x$ is the current reverse-diffusion state.

---

## 11. Estimating measurement-altered observables

For a fixed weak record $s$ at noise level $\tau_0$, generate posterior samples

```math
z^{(1)},\ldots,z^{(M)}
\sim
p_\theta(z|s,\tau_0).
```

Estimate the posterior one-point function by

```math
\widehat m_i(s;\tau_0)
=
\frac1M\sum_{a=1}^M z_i^{(a)}.
```

Then estimate

```math
A_1(\tau_0)
=
\frac1L\sum_i
\mathbb{E}_{s\sim p_{\tau_0}}
\left[
m_i(s;\tau_0)^2
\right]
```

using

```math
\widehat A_1(\tau_0)
=
\frac{1}{N_sL}
\sum_{\alpha=1}^{N_s}
\sum_{i=1}^L
\widehat m_i(s_\alpha;\tau_0)^2.
```

For finite $M$, $\widehat m_i^2$ has an upward Monte Carlo bias. A safer estimator uses two independent posterior sample batches:

```math
\widehat m_i^{(1)}(s)
=
\frac1M\sum_{a=1}^M z_{i,1}^{(a)},
```

```math
\widehat m_i^{(2)}(s)
=
\frac1M\sum_{a=1}^M z_{i,2}^{(a)}.
```

Then use

```math
\widehat A_1(\tau_0)
=
\frac{1}{N_sL}
\sum_{\alpha=1}^{N_s}
\sum_i
\widehat m_i^{(1)}(s_\alpha)
\widehat m_i^{(2)}(s_\alpha).
```

The same logic applies to two-point or general $Z$-diagonal observables $O(z)$:

```math
A_O(\tau_0)
=
\mathbb{E}_s
\left[
\mathbb{E}[O(z)|s,\tau_0]^2
\right].
```

---

## 12. Direct denoiser baseline

For estimating only one-point functions, a simpler baseline is a direct posterior-mean denoiser

```math
f_\phi(x,\tau)_i
\approx
\mathbb{E}[z_i|x,\tau].
```

Train it with mean-square loss:

```math
\mathcal L_{\mathrm{MSE}}
=
\mathbb{E}_{z,\tau,x}
\left[
\frac1L\sum_i
\left(
f_{\phi,i}(x,\tau)-z_i
\right)^2
\right].
```

Again, the input is only $(x,\tau)$, while $z$ is the target.

This baseline is useful for $A_1$, but SEDD is more general because it learns a reverse generative process and can sample posterior configurations.

---

## 13. Noise-level sampling

The model should be trained over the range of $\tau$ values needed at inference.

Possible choices:

### Uniform in $\tau$

```math
\tau\sim \operatorname{Uniform}(\tau_{\min},\tau_{\max}).
```

### Uniform in $\beta$

```math
\beta\sim \operatorname{Uniform}(\beta_{\min},\beta_{\max}),
\qquad
\tau=\tanh(2\beta).
```

### Uniform in diffusion time

Define

```math
\tau=e^{-2\gamma t},
```

and sample

```math
t\sim \operatorname{Uniform}(t_{\min},t_{\max}).
```

Avoid exact endpoints:

```math
\tau=0
```

is the pure-noise limit, and

```math
\tau=1
```

can produce singular target ratios. Use cutoffs such as

```math
\tau_{\min}>0,
\qquad
\tau_{\max}<1.
```

### Uniform in diffusion time

Define

```math
\ell=-\frac12\log\tau
=-\frac12\log\tanh(2\beta).
```

Then sample

```math
\ell\sim \operatorname{Uniform}(\ell_{\min},\ell_{\max}),
\qquad
\tau=e^{-2\ell},
\qquad
\beta=\frac12\operatorname{atanh}(e^{-2\ell}).
```

This is often useful when training across a wide noise range because it samples logarithmically in $\tau$. Use $\ell_{\min}>0$ to avoid the singular $\tau=1$ endpoint.

---

## 14. Network architecture

The model should implement

```math
u_\theta(x,\tau):\{\pm1\}^L\times\mathbb{R}\to\mathbb{R}^L.
```

Recommended inputs per site:

```math
x_i
```

plus a global noise embedding of $\tau$, $\beta$, or $\log\tau$.

Possible architectures:

1. **Dilated 1D CNN**

   Good for periodic spin chains and multiscale critical correlations.

2. **Transformer encoder**

   Good for long-range dependencies, more expensive.

3. **Hybrid CNN + attention**

   Useful when local structure and long-range correlations both matter.

For periodic boundary conditions, use circular padding or explicit periodic positional handling.

For translation-invariant systems, the architecture should share weights across sites.

---

## 15. Symmetries and data augmentation

For the TFIM order-parameter snapshot distribution without longitudinal field, the clean distribution is invariant under global spin flip:

```math
z\to -z.
```

The noisy channel respects

```math
(z,x)\to(-z,-x).
```

Use global spin-flip augmentation:

```python
if torch.rand(()) < 0.5:
    z = -z
    x = -x
```

For periodic boundary conditions, also use cyclic shifts:

```math
z_i\to z_{i+\ell},
\qquad
x_i\to x_{i+\ell}.
```

---

## 16. Validation checks

### 16.1 Shape and leakage check

Confirm that the model call has the form

```python
u = model(x, tau)
```

and not

```python
u = model(z, x, tau)
```

The clean $z$ may appear only in the loss construction.

### 16.2 Small-system exact validation

For small $L$, enumerate all $z$ and compute

```math
p(z|x,\tau)
=
\frac{p_0(z)q_\tau(x|z)}
     {\sum_{z'}p_0(z')q_\tau(x|z')}.
```

Compare posterior means and correlations from the sampler against exact values.

### 16.3 Calibration identity

For any observable $O(z)$,

```math
\mathbb{E}_s
\left[
\mathbb{E}[O|s]^2
\right]
=
\mathbb{E}_{z,s}
\left[
O(z)\mathbb{E}[O|s]
\right].
```

Use held-out $(z,s)$ pairs to check this identity with model estimates.

### 16.4 Dependence on noise level

At $\tau=0$, the noisy configuration is independent of $z$. The posterior should reduce to the prior.

At $\tau\to1$, the noisy configuration nearly equals $z$. The posterior should concentrate around the observed configuration.

---

## 17. Common implementation mistakes

### Mistake 1: Feeding clean $z$ into the network

Wrong:

```python
u = model(z, x, tau)
```

Correct:

```python
u = model(x, tau)
```

### Mistake 2: Sampling $x$ uniformly at all noise levels

At $\tau=0$, $x$ is uniform. At finite $\tau$, $x$ must be generated by corrupting clean snapshots:

```math
x\sim q_\tau(x|z).
```

### Mistake 3: Using one fixed corrupted copy per clean snapshot

Better: generate $x$ on the fly during training. This gives many noisy views of each clean snapshot and matches diffusion training practice.

### Mistake 4: Evaluating outside the trained noise range

A model trained only on a limited interval of $\tau$ should not be trusted far outside that interval.

### Mistake 5: Squaring noisy posterior Monte Carlo estimates directly

For

```math
\mathbb{E}_s[\mathbb{E}[O|s]^2],
```

use independent posterior batches or bias correction when estimating the square.

---

## 18. Minimal implementation checklist

1. Load clean snapshots $z\in\{\pm1\}^{N\times L}$.
2. Sample noise level $\tau$ or $\beta$.
3. Generate noisy $x\sim q_\tau(x|z)$ on the fly.
4. Compute

   ```math
   a_i=(1-\tau x_i z_i)/(1+\tau x_i z_i).
   ```

5. Feed only $(x,\tau)$ or $(x,\beta)$ to the model.
6. Output $u_{\theta,i}$ for all sites.
7. Train with

   ```math
   \exp(u_{\theta,i})-a_i u_{\theta,i}.
   ```

8. Use learned ratios to run reverse discrete diffusion.
9. Estimate posterior observables from generated samples.
10. Validate with small systems and calibration identities.

---

## 19. Short conceptual summary

The correct SEDD analogy is exactly the same as image or language diffusion:

```math
\text{clean data } z
\longrightarrow
\text{corrupted data } x.
```

The score network sees only the corrupted data and the noise level:

```math
\boxed{
u_\theta = u_\theta(x,\tau).
}
```

The clean data $z$ is used only to compute the denoising target in the loss:

```math
\boxed{
a_i(x,z;\tau)
=
\frac{q_\tau(F_i x|z)}{q_\tau(x|z)}.
}
```

For diagonal weak measurements, the weak record $s$ at strength $\beta$ is one instance of such a corrupted configuration, with

```math
\tau=\tanh(2\beta).
```

Thus, for measurement-altered criticality, train SEDD on synthetic corruptions of clean projective snapshots and use reverse denoising from the observed weak record to sample the posterior clean configuration.
