## Star-removal census: v2_w32

input `Neutralised.tif`
starless `starless_Neutralised_v2w32.fit`  score **+0.2023**

| metric | value |
|---|---|
| sites scored | 172542 / 172542 |
| clean | 48.63% (83908) |
| artifact | 50.38% (86926) |
| missed | 0.99% (1708) |
| artifact median residual | 19.0 x noise sigma |
| dark holes | 30.0% of artifacts, median depth 18.6 sigma |
| residual stars (est. total) | 55486 |
| mask smooth-leak fraction | 29.3% |

### By peak (bg-subtracted, 0-1)

| peak bin | n | clean % | artifact % | missed % |
|---|---|---|---|---|
| 0.03-0.1 | 2059 | 0.0 | 46.1 | 53.8 |
| 0.1-0.2 | 7737 | 2.9 | 91.6 | 5.5 |
| 0.2-0.35 | 25519 | 19.5 | 80.1 | 0.4 |
| 0.35-0.5 | 40769 | 35.8 | 64.1 | 0.1 |
| 0.5-0.65 | 78486 | 61.0 | 38.9 | 0.0 |
| >0.65 | 17972 | 90.1 | 9.9 | 0.0 |

### By FWHM (px)

| FWHM bin | n | clean % | artifact % | missed % |
|---|---|---|---|---|
| <1.5 | 18954 | 58.2 | 41.7 | 0.1 |
| 1.5-2 | 86593 | 54.5 | 45.4 | 0.1 |
| 2-3 | 56049 | 42.6 | 57.0 | 0.4 |
| 3-4 | 8054 | 19.2 | 76.0 | 4.8 |
| 4-6 | 2538 | 8.9 | 62.4 | 28.6 |
| 6-10 | 273 | 5.5 | 19.0 | 75.5 |
| >10 | 81 | 0.0 | 11.1 | 88.9 |
