## Star-removal census: star_v3_e2_512

input `Neutralised.tif`
starless `starless_Neutralised_v3e2.fit`  score **+0.7092**

| metric | value |
|---|---|
| sites scored | 172542 / 172542 |
| clean | 90.67% (156438) |
| artifact | 9.09% (15692) |
| missed | 0.24% (412) |
| artifact median residual | 10.3 x noise sigma |
| dark holes | 33.0% of artifacts, median depth 11.9 sigma |
| residual stars (est. total) | 2973 |
| mask smooth-leak fraction | 44.4% |

### By peak (bg-subtracted, 0-1)

| peak bin | n | clean % | artifact % | missed % |
|---|---|---|---|---|
| 0.03-0.1 | 2059 | 1.1 | 86.7 | 12.1 |
| 0.1-0.2 | 7737 | 27.7 | 71.1 | 1.2 |
| 0.2-0.35 | 25519 | 80.3 | 19.4 | 0.2 |
| 0.35-0.5 | 40769 | 95.3 | 4.7 | 0.0 |
| 0.5-0.65 | 78486 | 98.2 | 1.8 | 0.0 |
| >0.65 | 17972 | 99.3 | 0.7 | 0.0 |

### By FWHM (px)

| FWHM bin | n | clean % | artifact % | missed % |
|---|---|---|---|---|
| <1.5 | 18954 | 95.6 | 4.4 | 0.0 |
| 1.5-2 | 86593 | 94.9 | 5.1 | 0.0 |
| 2-3 | 56049 | 88.9 | 11.1 | 0.0 |
| 3-4 | 8054 | 67.2 | 32.2 | 0.6 |
| 4-6 | 2538 | 34.8 | 59.2 | 6.0 |
| 6-10 | 273 | 11.4 | 48.0 | 40.7 |
| >10 | 81 | 1.2 | 12.3 | 86.4 |
