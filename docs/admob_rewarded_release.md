# AdMob Rewarded Release Handoff

The code is prepared for this product contract:

- The first 3 analyses each day are free.
- After that, a user may voluntarily finish one rewarded ad to receive exactly 1 additional analysis.
- The Android app never mints a production credit directly. Google calls the public SSV endpoint, the server verifies Google's ECDSA signature, and the transaction ID is stored once to prevent replay.
- Gambling-site ads, betting links, and stake/purchase flows are outside the app contract.

## 1. AdMob console setup

Create the Android app with package `com.tttksj.racelens`, then create a rewarded ad unit with:

- Reward amount: `1`
- Reward item: `analysis_credit`
- SSV callback URL: `https://168-107-2-218.sslip.io/api/rewarded-ad/ssv`

Keep the resulting app ID (`ca-app-pub-...~...`) and rewarded unit ID (`ca-app-pub-.../...`) outside git.

## 2. Test APK

The EAS `preview` profile already enables rewarded ads with Google's official sample app ID and `TestIds.REWARDED`. It is isolated from production ad inventory and production credits:

```sh
npm run build:android:preview
```

The local web preview also has an in-memory 3-use quota and a clearly labelled `PREVIEW TEST AD`:

```sh
npm run preview
```

## 3. Production configuration

Set these in the EAS `production` environment:

```text
EXPO_PUBLIC_RACELENS_REWARDED_ADS=1
EXPO_PUBLIC_RACELENS_ADMOB_TEST_MODE=0
EXPO_PUBLIC_RACELENS_ADMOB_ANDROID_APP_ID=<real app ID>
EXPO_PUBLIC_RACELENS_ADMOB_REWARDED_AD_UNIT_ID=<real rewarded unit ID>
```

Set these on the Oracle API server using the exact same rewarded unit ID:

```text
RACELENS_FREE_DAILY_ANALYSIS_LIMIT=3
RACELENS_REWARDED_ADS_ENABLED=1
RACELENS_ADMOB_REWARDED_AD_UNIT_ID=<real rewarded unit ID>
```

The GitHub `Oracle Deploy` workflow exposes corresponding inputs. It refuses to enable rewarded ads without a valid unit ID.

## 4. Final gates

Run the release checks and then build the AAB:

```sh
npm run qa:submission
npm run build:android:production
```

The production build fails closed when IDs are blank/malformed, test mode is on, Google sample IDs are used, or the mobile/server rewarded unit IDs do not match. After deployment, verify one complete real-device path: 3 free analyses, rewarded ad, SSV receipt, one extra analysis, and no second credit from the same transaction.
