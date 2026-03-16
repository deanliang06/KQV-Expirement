# KQV-Expirement
Lol I hope this works 


Just Query For miniBert using attn_auto:
10 epochs (Attention Based)
lr=1e-5 
- really stable training (like linear decrease)
- MSE Loss: 0.03253064677119255
- Cos Loss: 0.8284776148705194
- KL-Div Loss: 8.344977569223019

lr=3e-5
- very unstable the first 7 epochs but then sudden decrease (I think too unstable)
- MSE Loss: 0.014594662003219128
- Cos Loss: 0.9301420738698675
- KL-Div Loss: 3.671249432171016

lr=1e-6
- very slow but decreasing
- MSE Loss: 0.11709991097450256
- Cos Loss: 0.48006998839388093
- KL-Div Loss: 31.123475297443555

lr=3e-6
- kinda slow but decreasing
- MSE Loss: 0.08504840731620789
- Cos Loss: 0.5828825340861767
- KL-Div Loss: 21.301326486835517

10 epochs (U-net Based (unet_auto) and minibert)
lr=3e-4 
- stable but then kinda bottoms-out
- MSE Loss: 0.006003474351018667
- Cos Sim: 0.9713925784904194
- KL-Div Loss: 1.489053214886804

lr=1e-4
- stable but then kinda bottoms-out
- MSE Loss: 0.004913791548460722
- Cos Sim: 0.9768609692847806
- KL-Div Loss: 1.2195472710272846

lr=1e-3
- quite stable 
- MSE Loss: 0.005955997854471207
- Cos Sim: 0.9707387335939345
- KL-Div Loss: 1.4785209607454686

lr=3e-5
- really stable and good
- MSE Loss: 0.004621141590178013
- Cos Sim: 0.9782083167749293
- KL-Div Loss: 1.1464806598775528


Unet with distilGPT-2
Average Perplexity: 3148.377924321774
MSE Loss: 23.93071746826172
Cos Sim: 0.9973875254720121

LAMBADA eval
-----------------
Original distilbert accurcy: 0.11419677734375%
Our represnetation model accurcy: 0.0008072853088378906%
Random initialization accurcy: 8e-5%