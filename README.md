# 📈 RewardAdaptiveScheduler

A **reward-driven adaptive learning rate scheduler** for PyTorch.  
Unlike classic schedulers that depend on epochs or loss values,  
this scheduler adapts **directly to reward signals** (from RL, bandits, or hybrid setups).  

---

## ✨ Features
- Works with any `torch.optim.Optimizer` (AdamW, SGD, Adafactor, …).
- Driven purely by **reward signals** (`observe(R=...)` after each step).
- Supports multiple **adaptation rules**:
  - **Scaling** — linear mapping from reward → LR
  - **Trend** — multiplicative update when reward improves/drops
  - **Variance** — risk-sensitive, shrinks LR on noisy rewards
  - **Kelly** — Kelly-criterion inspired bet sizing
- Optional momentum adaptation (`SGD` style).
- Safe clamping with `[min_lr, max_lr]` and `[min_momentum, max_momentum]`.
- Compatible with Hugging Face `Trainer` via callback.

---

## 📐 Mathematical Formulation

Let:
- \( R_t \) : reward at step \( t \)  
- \( \tilde R_t \) : EMA-smoothed reward (if enabled)  
- \( \eta_t \) : learning rate at step \( t \)  
- \( \eta_0 \) : base learning rate  
- \( \eta_{\min}, \eta_{\max} \) : LR bounds  
- \( m_t \) : momentum at step \( t \) (optional)  

### Scaling Rule
\[
\eta_{t+1} = \mathrm{clip}\!\left(
\eta_0 \left[ 1 + \frac{ \tilde R_t - c }{ s } \right], \,
\eta_{\min}, \eta_{\max}
\right)
\]
- \( c \): reward center  
- \( s \): reward span  

---

### Trend Rule
\[
\eta_{t+1} =
\begin{cases}
\eta_t \cdot u, & \tilde R_t > \tilde R_{t-1} \\
\eta_t \cdot d, & \tilde R_t \leq \tilde R_{t-1}
\end{cases}
\]
- \( u > 1 \): up factor  
- \( 0 < d < 1 \): down factor  

---

### Variance Rule
Given window \( W_t = \{ \tilde R_{t-k+1}, \ldots, \tilde R_t \} \),  
\[
\eta_{t+1} = \mathrm{clip}\!\left(
\frac{ \eta_0 }{ 1 + \alpha \cdot \mathrm{Var}(W_t) }, \,
\eta_{\min}, \eta_{\max}
\right)
\]
- \( \alpha \): variance caution coefficient  

---

### Kelly Rule
\[
m_t = \tanh \!\left( \frac{\tilde R_t}{T} \right), \quad
\eta_{t+1} = \mathrm{clip}\!\left(
\eta_t \cdot (1 + g m_t), \,
\eta_{\min}, \eta_{\max}
\right)
\]
- \( T \): temperature (reward scaling)  
- \( g \): gain (how aggressively LR adapts)  

---

## 📝 Pseudocode

### General Training Loop
```text
initialize model, optimizer, scheduler
for step = 1...N:
    loss, reward = forward(batch)
    loss.backward()
    optimizer.step()

    # reward feedback -> scheduler
    scheduler.observe(R=reward)

    optimizer.zero_grad()
````

### Scheduler Observe Method

```text
function observe(R):
    R_t <- to_float(R)
    if EMA enabled:
        R_t <- beta * R_{t-1} + (1-beta) * R_t

    switch rule:
        case "scaling":
            eta <- eta_0 * (1 + (R_t - center)/span)
        case "trend":
            if R_t > R_{t-1}: eta <- eta * up
            else: eta <- eta * down
        case "variance":
            window <- append(R_t)
            eta <- eta_0 / (1 + alpha * var(window))
        case "kelly":
            m <- tanh(R_t / temp)
            eta <- eta * (1 + gain * m)
        case "none":
            eta <- eta

    eta <- clamp(eta, min_lr, max_lr)
    update optimizer.param_groups["lr"] = eta
```

---

## 🚀 Quick Usage

```python
from torch.optim import AdamW
from reward_adaptive_scheduler import RewardAdaptiveScheduler

optimizer = AdamW(model.parameters(), lr=5e-5)

scheduler = RewardAdaptiveScheduler(
    optimizer,
    rule="kelly",          # scaling | trend | variance | kelly | none
    ema_reward_beta=0.9,
    min_lr=1e-6,
    max_lr=1e-3,
    kelly_gain=0.2,
    kelly_temp=1.0,
)

for batch in dataloader:
    loss, reward = compute_loss_and_reward(batch)
    loss.backward()
    optimizer.step()
    scheduler.observe(R=reward)   # reward signal goes here
    optimizer.zero_grad()
```

---

## 🤖 Integration with Hugging Face Trainer

```python
from transformers import Trainer, TrainingArguments
from reward_callback import RewardObserveCallback

trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    train_dataset=train_ds,
    optimizers=(optimizer, None),
    callbacks=[RewardObserveCallback(scheduler, reward_fn)],
)
```

Where `reward_fn(step)` returns the latest reward.

---

## 🔬 Diagnostics

* Log LR and rewards:

  ```python
  writer.add_scalar("train/lr", scheduler.get_lr()[0], step)
  writer.add_scalar("train/reward", reward, step)
  ```
* Check scheduler state:

  ```python
  state = scheduler.state_dict()
  scheduler.load_state_dict(state)  # restores exact LR state
  ```

---

## 📊 Example Plot

![LR vs Reward](docs/lr_vs_reward.png)

---

## ⚠️ Notes & Pitfalls

* Always call `observe(R=...)` **after** `optimizer.step()`.
* If rewards are noisy → use EMA or pre-normalization.
* Kelly rule is robust for bounded noisy rewards; Trend rule is more aggressive.
* LR bounds prevent runaway instability.

---

## 📚 References

* Kelly, J. L. (1956). *A New Interpretation of Information Rate*.
* Sutton & Barto (2018). *Reinforcement Learning: An Introduction*.
* PyTorch docs: [`torch.optim.lr_scheduler`](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).
