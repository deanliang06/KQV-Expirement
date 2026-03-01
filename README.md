# KQV-Expirement
Lol I hope this works 


Just Query:
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

10 epochs (U-net Based)
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