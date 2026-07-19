## Star-removal census: v2_e1

input `Neutralised.tif`
starless `starless_Neutralised_v2e1.fit`  score **+0.4915**

| metric | value |
|---|---|
| sites scored | 172542 / 172542 |
| clean | 74.44% (128433) |
| artifact | 24.86% (42888) |
| missed | 0.71% (1221) |
| artifact median residual | 16.6 x noise sigma |
| dark holes | 42.5% of artifacts, median depth 17.2 sigma |
| residual stars (est. total) | 10667 |
| mask smooth-leak fraction | 34.9% |

### By peak (bg-subtracted, 0-1)

| peak bin | n | clean % | artifact % | missed % |
|---|---|---|---|---|
| 0.03-0.1 | 2059 | 0.0 | 63.0 | 37.0 |
| 0.1-0.2 | 7737 | 6.6 | 89.0 | 4.3 |
| 0.2-0.35 | 25519 | 48.7 | 50.9 | 0.3 |
| 0.35-0.5 | 40769 | 72.5 | 27.4 | 0.1 |
| 0.5-0.65 | 78486 | 87.1 | 12.9 | 0.0 |
| >0.65 | 17972 | 97.5 | 2.5 | 0.0 |

### By FWHM (px)

| FWHM bin | n | clean % | artifact % | missed % |
|---|---|---|---|---|
| <1.5 | 18954 | 89.8 | 10.1 | 0.0 |
| 1.5-2 | 86593 | 79.8 | 20.1 | 0.1 |
| 2-3 | 56049 | 67.9 | 31.9 | 0.2 |
| 3-4 | 8054 | 45.9 | 51.3 | 2.8 |
| 4-6 | 2538 | 21.4 | 58.4 | 20.2 |
| 6-10 | 273 | 7.3 | 16.5 | 76.2 |
| >10 | 81 | 1.2 | 8.6 | 90.1 |
