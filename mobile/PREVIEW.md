# RaceLens Local Preview

Use this when you want to run the latest local web export against the live RaceLens API without changing the production Android build.

```sh
npm run preview
```

The command builds `dist-preview/` and starts the local preview server:

- This PC: `http://localhost:4173`
- Phone on the same Wi-Fi: `http://<this PC LAN IP>:4173`

The UI comes from the current local `mobile/` code. API, recent race, legal, health, and prediction requests are same-origin requests to the preview server, and the preview server proxies them to `https://168-107-2-218.sslip.io`. CORS behavior, rate limits, and data availability are still determined by the live server.

Port `4173` is required. `src/services/raceApi.ts` only enables same-origin API fallback for `localhost:4173`; non-local LAN hosts also use same-origin. Running this on a different port can make the app fall back to offline/local behavior instead of the live API proxy.

This is only a local experience preview. It is unrelated to the production EAS Android App Bundle. Production Android output is still built separately with:

```sh
eas build --platform android --profile production
```

The preview server and `dist-preview/` are not uploaded to the store.

The local preview enforces its own in-memory 3-use quota. After the third analysis it shows a clearly labelled `PREVIEW TEST AD`; completing it grants one local preview analysis without changing production usage or credits. Restarting the preview server resets this local test quota.

For an installable Android test, the EAS `preview` profile builds an APK with Google's official test app/ad IDs. It never requests live ads. A production AAB is blocked until real AdMob IDs are configured.

## Configuration

Defaults are usually correct:

- `PORT=4173`
- `PREVIEW_DIST=dist-preview`
- `PREVIEW_UPSTREAM=https://168-107-2-218.sslip.io`

Example:

```sh
PORT=4173 PREVIEW_UPSTREAM=https://168-107-2-218.sslip.io npm run preview:serve
```
