## Star-removal census: v2_e4

input `Neutralised.tif`
starless `starless_Neutralised_v2e4.fit`  score **+0.5084**

| metric | value |
|---|---|
| sites scored | 172542 / 172542 |
| clean | 76.01% (131156) |
| artifact | 23.36% (40299) |
| missed | 0.63% (1087) |
| artifact median residual | 16.1 x noise sigma |
| dark holes | 36.2% of artifacts, median depth 16.7 sigma |
| residual stars (est. total) | 9798 |
| mask smooth-leak fraction | 33.1% |

### By peak (bg-subtracted, 0-1)

| peak bin | n | clean % | artifact % | missed % |
|---|---|---|---|---|
| 0.03-0.1 | 2059 | 0.0 | 64.0 | 36.0 |
| 0.1-0.2 | 7737 | 6.8 | 89.9 | 3.3 |
| 0.2-0.35 | 25519 | 50.2 | 49.6 | 0.3 |
| 0.35-0.5 | 40769 | 74.7 | 25.2 | 0.0 |
| 0.5-0.65 | 78486 | 88.9 | 11.1 | 0.0 |
| >0.65 | 17972 | 97.7 | 2.3 | 0.0 |

### By FWHM (px)

| FWHM bin | n | clean % | artifact % | missed % |
|---|---|---|---|---|
| <1.5 | 18954 | 90.2 | 9.8 | 0.0 |
| 1.5-2 | 86593 | 82.4 | 17.5 | 0.0 |
| 2-3 | 56049 | 68.7 | 31.2 | 0.2 |
| 3-4 | 8054 | 45.2 | 52.5 | 2.3 |
| 4-6 | 2538 | 21.3 | 58.7 | 20.0 |
| 6-10 | 273 | 7.3 | 16.5 | 76.2 |
| >10 | 81 | 1.2 | 6.2 | 92.6 |
