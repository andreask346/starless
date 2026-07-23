## Star-removal census: star_v3_e1

input `Neutralised.tif`
starless `starless_Neutralised_v3e1.fit`  score **+0.7032**

| metric | value |
|---|---|
| sites scored | 172542 / 172542 |
| clean | 89.76% (154866) |
| artifact | 9.99% (17231) |
| missed | 0.26% (445) |
| artifact median residual | 11.4 x noise sigma |
| dark holes | 28.1% of artifacts, median depth 12.2 sigma |
| residual stars (est. total) | 5210 |
| mask smooth-leak fraction | 47.7% |

### By peak (bg-subtracted, 0-1)

| peak bin | n | clean % | artifact % | missed % |
|---|---|---|---|---|
| 0.03-0.1 | 2059 | 1.5 | 84.9 | 13.6 |
| 0.1-0.2 | 7737 | 32.4 | 66.4 | 1.1 |
| 0.2-0.35 | 25519 | 77.0 | 22.8 | 0.2 |
| 0.35-0.5 | 40769 | 92.7 | 7.3 | 0.0 |
| 0.5-0.65 | 78486 | 98.1 | 1.9 | 0.0 |
| >0.65 | 17972 | 99.5 | 0.5 | 0.0 |

### By FWHM (px)

| FWHM bin | n | clean % | artifact % | missed % |
|---|---|---|---|---|
| <1.5 | 18954 | 90.8 | 9.2 | 0.0 |
| 1.5-2 | 86593 | 94.3 | 5.7 | 0.0 |
| 2-3 | 56049 | 88.8 | 11.2 | 0.0 |
| 3-4 | 8054 | 66.6 | 32.6 | 0.8 |
| 4-6 | 2538 | 34.0 | 59.5 | 6.5 |
| 6-10 | 273 | 10.6 | 49.5 | 39.9 |
| >10 | 81 | 1.2 | 14.8 | 84.0 |
