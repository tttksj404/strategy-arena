# RaceLens Monetization Setup

Paid billing remains off. Rewarded ads are a separate flow:

```sh
EXPO_PUBLIC_MONETIZATION=0
EXPO_PUBLIC_RACELENS_REWARDED_ADS=1
```

When store review is ready:

1. Set `EXPO_PUBLIC_MONETIZATION=1` in the Expo build profile.
2. Configure the API server with `RACELENS_APPLE_SHARED_SECRET` for iOS receipt checks.
3. Configure `RACELENS_GOOGLE_SA_JSON` for Android receipt checks.
4. Install and wire the native IAP SDK only after store products exist.
5. Create the Android app and rewarded unit in AdMob, then configure the SSV callback as `https://<production-domain>/api/rewarded-ad/ssv`.
6. Put the real app ID in `EXPO_PUBLIC_RACELENS_ADMOB_ANDROID_APP_ID` and the same real rewarded unit ID in both `EXPO_PUBLIC_RACELENS_ADMOB_REWARDED_AD_UNIT_ID` and the Oracle server's `RACELENS_ADMOB_REWARDED_AD_UNIT_ID`.
7. Keep `EXPO_PUBLIC_RACELENS_ADMOB_TEST_MODE=0` in production. The EAS `preview` profile uses Google's official test ad automatically.
8. Enable `RACELENS_REWARDED_ADS_ENABLED=1` and restore `RACELENS_FREE_DAILY_ANALYSIS_LIMIT=3` only after the SSV callback is saved in AdMob.

The app waits for a signed server callback before spending the extra analysis credit. Direct client calls to `/api/rewarded-ad/claim` cannot create production credits.
