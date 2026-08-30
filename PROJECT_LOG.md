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


