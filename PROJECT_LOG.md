This project journal will try to keep a log of the time I spent on the project, with notes on its advancement. 
I have to carry on the project interleaved with my work, so the time spent on it will be rather fragmented, but hopefully I can realise the 16h in 4-5 days of sparse sessions. 

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

## 2026-08-30 18:43
5h into the project. 
Doing a serious fit of the H-lens will only cost few hours on a H100, so we scheduled that job. 
The result can be predicted to be solidly negative. The Hessian/curvature provide qualitatively different information, parhaps. 
We'll think about it while the H-lens is computed. 

## 2026-08-30 18:56
Important note: it looks like we don't have a good dataset of examples of failures for the J-lens. 
When we are confident that the H-lens idea fails, if we still have time, we could dedicate the rest of the time to try to build such dataset, maybe adding a new model like Qwen3.8-27B who is making headlines as a strong model. 

## 2026-08-30 19:27
I spent 30 more minutes on checking the experiments that I'm running on the cluster, so I guess I should add that on the project time. 
5h30. 

## 2026-08-31 15:45
I have been spending 45m looking at the new results obtained from the cluster. 
The serious problem we have, which we should have thought about at the very beginning, is that we do not have a robust way to measure if the H-lens is better or not. So now we are searching if there are more robust datasets of J-lens failures, and consideirng investing time in building a proper benchmark. 
6h15 into the project. 

## 2026-08-31 16:52
I had a hindrance and i'm just resuming working on the project now. 
Before the interruption I think I spent 30 more minutes on the project, thinking about how to do the dataset. 
Let's consider ourselves at 6h45 into the project. 

## 2026-08-31 19:50
I have worked 1h and a bit more on producing the dataset of J-lens possible failures. 
I'm leveraging GLM5.2 as a judge. 
Hopefully by tomorrow we have a first version of the dataset... we are still working on setting up the script that generates examples for a possible list of concepts. 
We can now consider ourselves at 8h into the project (halfway through it). 

## 2026-08-31 20:28
8h40 into the project, and we are generating the dataset. 
However, I'm inspecting the generated examples and I don't like them. 
There is an important confusion regarding the fact that whatever follows the probe that we feed in the J-lens is lost (only the prefix counts). 
So, we are trying to fix and rerun a better generative process. 

## 2026-08-31 20:40
Ok enough for today. 
We fixed the previous problem: now the probe is always at the end of the sentence. 
8h50. 7h10 to go before wrapping up. 
We will refine the dataset and then test again our H-lens idea. 

## 2026-09-01 12:00
We are briefly resuming work. 
The dataset generation finished and we plan to spend some time inspecting and curating it. 
Today is Sept. 1st and the deadline is Sept. 4th so today afternoon we plan to consume most of the remaining 7h for the project so that we can wrap it up. 
Ideally we will get a decent candidate dataset for benchmarking alternatives to the J-lens and make the failure of the current implementation of the H-lens idea more solid. 

## 2026-09-01 12:21
Lunch break. 9h10 spent. 
The dataset looks promising. 
Lots of things to check. 

## 2026-09-01 14:37
Resuming work. 
A subtle problem that is stealing some time is the fact that the J-lens ranking was using bfloat16 but that creates a lot of ties that make computing the top-10 rankings ambiguous, so we will probably spend 1h of GPU compute to recompute the J-lens rankings in float32. 
9h30 spent. 

## 2026-09-01 15:09
The precision error in the ranking is something we should keep in mind to report explicitly in the final report. 
9h50 spent. 
In the interest of time, we will accept some noise in the current version of the dataset. 

## 2026-09-01 15:33
10h10 into the project. 
We have been inspecting manually some examples of our dataset and we are now calling GLM5.2 on the items to ask it to rate whether they are a good example of J-lens failing. 
This way, we will restrict the dataset to examples that are good according to this filter. 

## 2026-09-01 15:40
Interesting, the grading by GLM5.2 seems (preliminary run) aligned with the ranking of the concept provided by the J-lens. 
10h20 almost. 
Our dataset, filtered like this, should now provide a decent signal. 

## 2026-09-01 16:00
I inspected more carefully the prompt used by the grader and it's not good IMO, so I am re-grading them with a better prompt. 
10h40 spent. 5h20 to go. 

## 2026-09-01 16:11
10h50 spent. We need to make a long break. In 15m the float32-precision dataset will be done, and we can benchmark the H-lens on it. 
We will then decide if testing another idea that came out while working on this project, or just spend the rest of the time in refining the dataset. 
Refining the dataset should be the wiser option. 

## 2026-09-01 17:33
We can resume for few minutes. Let's see if we have time to run H-lens on the new dataset. 

## 2026-09-01 18:46
We had to interrupt after few minutes before. 
Now resuming work. We can count 11h. 

## 2026-09-01 19:04
11h20 spent. The H-lens didn't provide meaningful improvements. We could spend more compute on estimating it more carefully perhaps, but it doesn't seem worth, there is no signal that it contributes a meaningful correction. 
We will check a bit more the creation of the dataset, while we start collecting what should go in the final writeup. 
One option to leverage the dataset we just created is to benchmark the R-lens on it, on layer 6. 
That's an interesting use of the dataset and we will probably do that. 

## 2026-09-01 19:19
While the experiments run (including the R-lens test), we are reading jspace.py to check Claude didn't make wrong assumptions.
11h35 spent. 

## 2026-09-01 19:46
12h00 spent. 4h remaining. 

## 2026-09-01 19:57
12h10 spent. We got a good validation for the R-lens. Our dataset seems to have value!
So far, we saw that the H-lens doesn't improve. To validate this, we created a dataset that might be useful. 
The remaining 4 hourse will probably be spent checking the details of what we did and writing the results (for which we actually have extra hours). 
The strong examples in the dataset are only around 800, so we will probably check them one by one. 
