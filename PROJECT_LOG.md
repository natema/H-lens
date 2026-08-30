## 2026-08-29 16:12
Starting to work on project PROJECT_IDEA.md. 
Let's get the baselines. 

## 2026-08-29 16:42
I'm looking at the 5 exampel probes for the deepseek j-lens failures and I find them weird, I'm trying to understand if I'm getting something wrong..

## 2026-08-29 16:55
Ok setup is proceeding, we should be able to reproduce the j-lens failures on Qwen3.5-4B soon, hopefully. 

## 2026-08-29 17:07
Ok we are done kind-of replicating the 5 famous j-lens failures, except that on Qwen3.5-4B the Einstein failure doesn't transfer from DeepSeek (I'll check if somebody signaled that on HF or elsewhere). 
One interesting thing is that the " Japan" example is not considering "Japan" without leading space, which slightly changes the numbers (but qualitatively they are the same, e.g. Logit lens gets one hit but it is still a poor result w.r.t. R-lens). 
We checkpoint here the 1st hour of the project. 

## 2026-08-29 18:31
Let's resume and test the diagonal-hessian assumption via a randomized test. 

## 2026-08-29 19:04
Tests so far suggests that the diagonal hessian accounts for less than 1% of the Frobenius norm (what Codex calls the "energy"). 
We stop here waiting for a Codex reset...
1h30 into the project. 

## 2026-08-30 15:06
We spent 30m looking at the "energy" estimation for the Hessian and 1% of diagonal mass is not a bad result for an Hessian with millions of entries. 
We still have to understand the details of the method used by Codex for the randomized estimation of the Hessian, but I'm confident that it's a standard approach. 
We thus prioritize proceeding with adding the Hessian correction and computing such a "j^2"-lens (better to call it H-lens, from Hessian). 
2h into the project. 

## 2026-08-30 15:59
The first test of the H-lens idea was negative: the H-lens worsen the ranking of the relevant concepts in our probe. 
I looked at how codex did the estimate of the Hessian and I was surprised to see that it used finite differences to estimate the derivative. I am repeating the experiment using forward-mode autodiff. 
Almost 3h into the project. 

## 2026-08-30 16:17
I want to look at more examples to further test the failure of the H-lens idea, so I'm investing some time in collecting a richer dataset (first we are searching previous works, then we could consider finding new examples ourselves). 
Meanwhile, autodiff is half-through the calculation of the diagonal of H. 

## 2026-08-30 16:29
We are going to spend more time to build a better set of cases to test the H-lens idea. 
So far, we spent 3h30 roughly on the project. 
We pause here for some time. 

## 2026-08-30 17:17
Let's resume working. 
In the meantime, the calculation with autodiff finished. Let's see if anything changes. 

## 2026-08-30 17:39
We switched to Claude right now (instead of buing pro/max of one of them, we bought the cheap ones of both atm). 
We are enlarging the battery of tests, adding those used to compare R-lens and J-lens. 

## 2026-08-30 18:00
Our battery of test now has ~30 cases taken from the R-lens work. 
To do a serious falsification, we need to recompute the diagonal H at layer 12 (so far we did it at layer 6). 
This will take ~30m probably. We will work on a systematic falsification why we think if we might be missing a proper implementation of the idea of leveraging the Hessian. 
>4h into the project so far. 

## 2026-08-30 18:19
Well, regarding how the J-lens is computed... it turns out that the number of input-output token pairs I was using for estimating the diagonal Hessian was redicolously small compared to the thousend of pairs used for the J-lens... my bad, I trusted Codex too naively. 
We are now scheduling a job on the Jean Zay cluster to compute the H-lens in a decently noise-robust way.
Towards 4h30 into the project. 
