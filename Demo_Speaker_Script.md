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

🗣️ *"Finally, the serving abstraction. In production you don't call a model directly — you go through an inference server. I've simplified it here, but this pattern is exactly how TorchServe, Triton, and BentoML work. The server handles batching, tracks latency SLAs, and exposes stats for monitoring.*

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

🗣️ *"This is my favorite chart to show in system design discussions.*

*Each point is a model — different parameter count, different latency, different accuracy. The key insight: look at the Pareto frontier along the top-left edge. Those are the efficient models. Everything below and to the right is dominated — worse accuracy, higher latency.*

*Our TinyResNet, in red — it's not on the ImageNet frontier, but it was designed for a different task and constraint. This is how you think about model selection: you don't pick the highest accuracy model, you pick the model that's optimal for your specific accuracy, latency, and cost constraints."*

👆 **Scroll to the Summary markdown cell**

🗣️ *"Let me wrap with the five mental models I want you to leave with:*

*One: Profile before you optimize. Never guess the bottleneck — measure it.*

*Two: Accuracy is necessary but not sufficient. Latency, memory, cost — they all have to work.*

*Three: Start simple. A logistic regression baseline before any neural network.*

*Four: Mixed precision is free performance. Always on.*

*Five: Quantize before scaling. A small efficient model beats a large unoptimized one every time.*

*These are the principles that separate ML engineers who write notebooks from the ones who build systems that serve millions of users.*

*I'm happy to take questions — on any of the design decisions we made, or on how these principles apply to other domains."*

---

## POST-SESSION NOTES

**Common questions to anticipate:**
- *"Why not use transfer learning?"* → Great question — for real production systems, we would fine-tune a pre-trained backbone (e.g., ResNet-50 on ImageNet). I kept it from-scratch here to make every design decision explicit.
- *"How do you choose batch size?"* → Rule of thumb: largest that fits in memory. Then tune LR proportionally (linear scaling rule). For very large batches, consider gradient accumulation.
- *"When would you use model parallelism vs. data parallelism?"* → Data parallelism first (easiest, scales well). Model parallelism only when a single model doesn't fit on one GPU — common with LLMs over 7B parameters.
- *"What's the difference between TorchServe and Triton?"* → TorchServe is PyTorch-native, easier to set up. Triton is hardware-optimized (NVIDIA), supports multiple frameworks, better for high-throughput production at scale.
