# 🎙️ Speaker Script — Deep Learning System Design (20-Min Demo)

---

> **Color key:**  
> 🗣️ Say this out loud  
> 👆 Do this action in the notebook  
> 💡 Teaching insight to emphasize  
> ⏱️ Time marker

---

## PRE-SESSION CHECKLIST (Before going live)

- [ ] Colab notebook open, runtime connected (check ✅ in top right)
- [ ] Runtime → Change runtime type → **GPU** selected
- [ ] Run the **imports cell** in advance (saves 30 sec live)
- [ ] Browser zoom at 130–140% so text is readable
- [ ] Camera framing: face + hands visible, notebook on second half of screen

---

## ⏱️ 0:00 — OPENING (90 seconds)

🗣️ *"Welcome. Today's session is on Deep Learning System Design — and I want to be upfront about what that means. Most courses stop at 'here's how to train a model.' I want to go one level deeper: how do you design a system that's not just accurate, but fast, memory-efficient, and actually shippable to production.*

*We'll cover this through a live coding session — you'll see real design decisions, real trade-offs, and real benchmarks. No slides, no hand-waving.*

*Here's our agenda for the next 20 minutes."*

👆 **Scroll to show the agenda table in the first markdown cell**

🗣️ *"Five topics. The first two are theory and architecture. The second three are hands-on code. Let's go."*

---

## ⏱️ 1:30 — PART 1: THE 4 PILLARS (2.5 minutes)

👆 **Scroll to the 4 Pillars markdown section. Point to the ASCII diagram.**

🗣️ *"When I interview engineers for senior ML roles — and I've done a lot of this — I can immediately tell who has built production systems and who hasn't. The ones who haven't go straight to: pick a model, train it, done.*

*The ones who have, they ask: what's my data throughput? What's my P99 latency budget? What's my memory envelope at scale?*

*Those four pillars on screen aren't sequential phases — they interact. A bad data pipeline can completely mask a great model. A great model with a slow inference path is useless for real-time applications."*

💡 **Pause here. Make this land:**

🗣️ *"One stat I always share: in most production DL systems, the model itself is maybe 20% of the code. The pipeline, the infra, the serving layer — that's the other 80%. Today we touch all of it."*

---

## ⏱️ 4:00 — PART 2: DATA PIPELINE (4 minutes)

👆 **Run the imports cell** (already pre-run ideally)

🗣️ *"Alright. Let's get into the code. I'm going to start where any system design starts: the data."*

👆 **Show and run the data pipeline cell**

🗣️ *"A few design decisions I want to call out here — and these aren't obvious if you're new:*

*First: notice we have separate transforms for training and validation. Training has random crop, flip, color jitter. Validation has none. This is intentional — you want your validation to measure the real performance of the model, not the augmented version of it.*

*Second: look at the DataLoader arguments. `num_workers=2` means two CPU processes pre-loading data in parallel while the GPU is running the forward pass. `pin_memory=True` pins the host memory for faster CPU-to-GPU transfers. `persistent_workers=True` avoids respawning those processes every epoch.*

*On that last one — by default, PyTorch kills the worker processes at the end of each epoch and respawns them at the start of the next. That's process forking, library reimporting, dataset reinitialization — all over again, every epoch. `persistent_workers=True` keeps them alive and idle between epochs. They wake up when the next epoch starts, already warm. Over 100 epochs with even a moderately expensive dataset setup, that's minutes of wasted time eliminated for free.*

*These look like minor details. In practice, misconfiguring these is the #1 cause of GPU underutilization I see on ML teams."*

👆 **Run the benchmark cell**

🗣️ *"This is the throughput benchmark — and I want you to internalize this pattern. Before you blame your model for being slow, always measure your pipeline throughput first. If your pipeline can't feed your GPU fast enough, no model optimization will help."*

👆 **Run the visualization cell**

🗣️ *"And here's what our augmented samples look like. You can see the crops, the flips. This diversity is what forces the model to learn invariant features."*

---

## ⏱️ 8:00 — PART 3: MODEL ARCHITECTURE (5 minutes)

👆 **Scroll to the ResNet markdown diagram**

🗣️ *"Now the model. I'm building a ResNet — a Residual Network — from scratch. Let me show you why the architecture matters at a systems level.*

*The key idea here is this diagram."*

👆 **Trace the skip connection in the ASCII art**

🗣️ *"Instead of learning the full mapping from input to output, a residual block learns just the difference — the residual. This is the innovation that allowed training networks of hundreds of layers, because gradients can flow directly through the skip connection without vanishing.*

*From a system design perspective, this matters because skip connections add memory overhead — you have to store the intermediate activations. That's a design trade-off: deeper network → better accuracy, but higher memory cost."*

👆 **Show and run the model definition cells**

🗣️ *"Two things I want to highlight in the code:*

*One: `inplace=True` on the ReLU. This is a small but real memory saving — it modifies the tensor in place rather than allocating a new one.*

*Two: The `_init_weights` method. Kaiming initialization — named after Kaiming He, the same researcher who invented ResNets. Without proper initialization, deep networks can start with vanishing or exploding gradients before the first update even happens.*

*And at the bottom: look at the memory estimate. About 23MB for FP32. With mixed precision, that drops to ~11MB. Before we even run a single training step, we already know our memory budget."*

💡 **If asked what BN is — keep it tight:**

🗣️ *"BatchNorm normalizes the activations within each layer across the batch — it forces the output of every layer to have roughly zero mean and unit variance. Why does that matter? Because as weights update during training, the distribution of activations feeding into the next layer keeps shifting. Later layers have to constantly adapt to a moving target, which slows training and makes you very sensitive to learning rate. BN stabilizes that distribution so each layer trains against a consistent input. The practical payoff: faster convergence, higher tolerance for large learning rates, and some built-in regularization from the per-batch noise in the statistics.*

*This is also exactly why `drop_last=True` matters in the DataLoader — BN computes its statistics per batch. A final batch of 3 samples instead of 128 gives you completely unreliable statistics and can destabilize training, especially early on."*

👆 **Show training config cell, then run the training loop**

🗣️ *"Three infrastructure decisions here that distinguish production training from notebook training:*

*AdamW over Adam — it fixes a subtle but important bug in how weight decay interacts with adaptive gradients.*

*Cosine annealing — the learning rate doesn't step down suddenly. It follows a smooth cosine curve. This tends to find better minima.*

*And mixed precision — the GradScaler automatically scales the loss to prevent gradient underflow in FP16. One line of code, double the throughput."*

🗣️ *"I'm going to let this run. While it's training, let me mention gradient clipping in the loop — `clip_grad_norm_ with max_norm=1.0`. This is your safety valve. If a bad batch causes a huge gradient, this prevents it from blowing up your optimizer state."*

👆 **Wait for training to finish, then run the learning curves cell**

🗣️ *"Look at these learning curves. The train/val gap tells you immediately whether you're overfitting or underfitting. And notice the LR plot — that smooth cosine decay. This is what healthy training dynamics look like."*

---

## ⏱️ 13:00 — PART 4: INFERENCE OPTIMIZATION (4 minutes)

👆 **Scroll to the inference markdown section**

🗣️ *"Now we flip the problem completely. Training is a one-time cost. Inference is forever.*

*If your model serves 10 million users a day, a 50ms latency improvement compounds into massive cost savings and user experience gains. This is where systems engineers earn their keep."*

👆 **Run the inference benchmark cell**

🗣️ *"Watch what happens as batch size increases. Batch=1 is your online serving case — a user sends one image and waits for an answer. Batch=128 is your offline scoring case — you're processing a warehouse of images and only care about total throughput.*

*Notice: latency increases sub-linearly. At batch=64, you have 64x the compute work, but maybe only 8x the latency. That's GPU parallelism working for you."*

👆 **Run the quantization cell**

🗣️ *"Now, quantization. This is a big one for production deployment.*

*Dynamic quantization converts Linear layer weights from FP32 to INT8 — no retraining needed. Three lines of code.*

*Look at the output: model size compressed 2-3x. And critically — the accuracy drop is near zero. Under 1% is our threshold. If it's within that, we ship the quantized model.*

*For mobile or edge deployment, this is non-negotiable. On devices without a GPU, INT8 inference is your only viable option."*

👆 **Run the inference server cell**

🗣️ *"Finally, the serving abstraction. In production you don't call a model directly — you go through an inference server. I've simplified it here, but this pattern is exactly how TorchServe, Triton, and BentoML work. The server handles batching, tracks latency SLAs — Service Level Agreements, the formal commitments you make to users or downstream systems about response time and availability, for example 'P99 latency under 50ms, 99.9% uptime' — and exposes stats for monitoring.*

*Now look at the three columns in the output — Latency, Acc, and Avg conf. Each one serves a different operational purpose."*

💡 **Point to each column as you explain it:**

🗣️ *"**Latency** we've already discussed — that's your SLA metric. You set a P99 budget and you monitor against it.*

***Acc — accuracy** — is your correctness signal on this specific batch. In a real server you wouldn't compute accuracy at runtime because you rarely have ground-truth labels on live traffic. But you would log predictions and run periodic offline evaluation against a labeled held-out set. If batch accuracy starts drifting down unexpectedly, that's your earliest warning of model drift or a data distribution shift upstream.*

***Avg conf — average confidence** — is the one I'd argue is actually more actionable in production than accuracy. It's the mean of the softmax probability on the predicted class. A high-confidence wrong prediction is your most dangerous failure mode — the model is certain and it's wrong. A low-confidence correct prediction tells you the model is on the boundary and you should collect more data there.*

*In practice, you set a confidence threshold — say 0.7. Predictions above it go straight through. Predictions below it get flagged for human review or routed to a fallback model. That's how you build a graceful degradation layer into an ML system, rather than a binary pass/fail."*

💡 **This is a good moment to pause and ask the audience:** *"If your Avg conf drops from 0.85 to 0.65 in production overnight with no model change — what's your first hypothesis?"* (Answer: input distribution shift — the incoming data looks different from what the model was trained on.)

---

## ⏱️ 17:00 — PART 5: TRADE-OFFS & WRAP-UP (3 minutes)

👆 **Run the trade-off visualization cell**

🗣️ *"This is my favorite chart to show in system design discussions — because it forces people to stop thinking in one dimension.*

*Each point is a real model family. The x-axis is inference latency at batch size 1 — that's your online serving cost. The y-axis is top-1 accuracy on ImageNet. The size of the point loosely reflects parameter count.*

*The key concept here is the **Pareto frontier** — the upper-left edge of the cloud. A model is on the Pareto frontier if no other model is strictly better on both axes simultaneously. MobileNetV2 is on the frontier: for its latency, nothing is more accurate. ViT-B/16 is also on the frontier: for its accuracy, nothing is faster. ResNet-50 sits in the middle — the workhorse that dominated production for years because it hit a sweet spot most teams could live with.*

*Everything below and to the right of the frontier is dominated — you can find something faster, or something more accurate, at the same cost. Those models are rarely the right choice."*

💡 **Point to our TinyResNet in red:**

🗣️ *"Our model sits off the ImageNet frontier — and that's completely fine. It wasn't designed for ImageNet. It was designed for 32×32 CIFAR images with a tight memory budget. The lesson isn't that it's worse — it's that **every model should be evaluated on the Pareto frontier of its own task and constraint space.***

*This is the conversation you want to have before writing a single line of code. What are my axes? What are my hard constraints? Which trade-offs am I actually willing to make?"*

---

👆 **Point to the Use Case table in the markdown**

🗣️ *"Look at how differently the priority ordering shifts by use case:*

*Autonomous driving — latency is non-negotiable. A 200ms delay at 60mph means your car has traveled 5 meters blind. No accuracy gain justifies that.*

*Medical imaging — accuracy dominates. A missed tumor is catastrophic. A 2-second response time is completely acceptable if the radiologist is reviewing it anyway.*

*Ads click-through prediction — cost and throughput drive everything. You're scoring billions of impressions per day. Even a 10% compute reduction at that scale translates to millions of dollars annually. And approximate is fine — the difference between 0.812 and 0.814 AUC is statistically invisible to the business.*

*These aren't hypothetical. These are the real conversations that happen in system design reviews at Google, Meta, and Amazon. The model is 20% of the decision. The other 80% is understanding where you sit on these axes."*

---

👆 **Point to the Scaling Laws table in the markdown**

🗣️ *"The other thing I want you to internalize before we close is the Scaling Laws table — because this is where a lot of engineering intuition breaks down.*

*The instinct when a model isn't accurate enough is to throw more at it — more data, bigger model, more compute. And those things do help. But the returns are logarithmic, not linear. Look at the numbers:*

*Doubling your training data buys you roughly 1–3% accuracy improvement. Not 50%. Not even 10%. 1–3%. That means if you're at 80% accuracy and you need to hit 85%, no amount of data alone gets you there — you need to rethink the architecture or the task framing.*

*Doubling model size is similar — 1–2% gain. And now you've also doubled your inference cost, your memory footprint, and your serving bill. That trade-off needs to be conscious, not reflexive.*

*The most underappreciated row is batch size. Doubling batch size does not give you a free 2x speedup. Larger batches have a generalization penalty — the gradient updates become smoother and the model tends to converge to sharper, less generalizable minima. The fix is to scale the learning rate proportionally — the linear scaling rule — but that introduces its own instability at very large scales. This is a whole area of active research, and it's why you can't just throw 8 GPUs at a problem and expect 8x faster convergence.*

*The compute row is the one that changed how the entire industry thinks about training large models. The Chinchilla paper from DeepMind in 2022 showed that for a given compute budget, most labs had been massively overtraining large models on too little data. The optimal strategy — what they called compute-optimal training — is to scale model size and dataset size in roughly equal proportion. Concretely: if you double your compute, you should make the model about 1.4x larger AND train on about 1.4x more tokens, rather than just making the model bigger. GPT-3 at 175 billion parameters, for example, was significantly undertrained by this standard. Chinchilla at 70 billion parameters, trained on 4x more data, matched or beat it on most benchmarks at a fraction of the inference cost. This is why model size alone is no longer a reliable proxy for capability — training efficiency matters just as much."*

*The practical takeaway: scaling is a tool of last resort, not first resort. Exhaust your pipeline optimizations, your augmentation strategy, your architecture choices first. Scale compute when those are maxed out — and even then, know exactly what you're buying."*

💡 **Optional follow-up question for the audience:** *"If you had a fixed compute budget and had to choose between 2x more data or 2x more model parameters — which would you pick, and why does the answer depend on where you are in training?"* (Answer: early in training, more data wins because the model is underfitting; late in training when the model has saturated on the data, more capacity can help — but the real answer is always: measure both.)

---

👆 **Scroll to the Summary markdown cell**

🗣️ *"Let me wrap with the five mental models I want you to leave with — and I mean these as actual decision rules, not just principles:*

*One: **Profile before you optimize.** Never guess the bottleneck — measure it. We benchmarked the data pipeline before touching the model. We measured inference latency before quantizing. Every optimization we made was in response to data, not intuition.*

*Two: **Accuracy is necessary but not sufficient.** P99 latency, memory footprint, inference cost — they all have to work within their respective budgets. A 95% accurate model that misses its SLA is not deployable.*

*Three: **Start simple, measure, iterate.** A logistic regression baseline before any neural network. Not because it will win — it won't — but because it gives you a floor to beat, a debugging baseline, and a sanity check on your data pipeline. Teams that skip this step waste weeks chasing model bugs that were actually data bugs.*

*Four: **Mixed precision is free performance.** One flag, zero architecture changes, roughly double throughput on modern GPUs. There is no reason to train in FP32 anymore.*

*Five: **Quantize before you scale.** The instinct when accuracy is insufficient is to reach for a bigger model. The better instinct is to first squeeze every bit of efficiency out of the current one — quantization, compilation, better data, better augmentation. Scaling compute is expensive and slow. Optimization is fast and often free.*

*The engineers who internalize these aren't just better at ML — they're better at shipping. And shipping is what actually matters."*

🗣️ *"I'm happy to take questions — on any of the design decisions we made today, or on how these trade-offs play out in specific domains you're working in."*

---

## POST-SESSION NOTES

**Common questions to anticipate:**
- *"Why not use transfer learning?"* → Great question — for real production systems, we would fine-tune a pre-trained backbone (e.g., ResNet-50 on ImageNet). I kept it from-scratch here to make every design decision explicit.
- *"How do you choose batch size?"* → Rule of thumb: largest that fits in memory. Then tune LR proportionally (linear scaling rule). For very large batches, consider gradient accumulation.
- *"When would you use model parallelism vs. data parallelism?"* → Data parallelism first (easiest, scales well). Model parallelism only when a single model doesn't fit on one GPU — common with LLMs over 7B parameters.
- *"What's the difference between TorchServe and Triton?"* → TorchServe is PyTorch-native, easier to set up. Triton is hardware-optimized (NVIDIA), supports multiple frameworks, better for high-throughput production at scale.
