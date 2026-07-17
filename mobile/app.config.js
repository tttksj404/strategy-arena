const fs = require('fs');
const path = require('path');

function enabled(value) {
  return ['1', 'true', 'yes', 'on'].includes(String(value || '').trim().toLowerCase());
}

function resolveServiceFile(configDir, configuredPath) {
  const servicePath = configuredPath || '';
  if (!servicePath) return '';
  const absolutePath = path.isAbsolute(servicePath) ? servicePath : path.join(configDir, servicePath);
  return fs.existsSync(absolutePath) ? servicePath : '';
}

module.exports = ({ config }) => {
  const configDir = __dirname;
  const firebaseEnabled = enabled(process.env.EXPO_PUBLIC_RACELENS_FIREBASE_ENABLED);
  const androidGoogleServicesFile = resolveServiceFile(
    configDir,
    process.env.GOOGLE_SERVICES_JSON || process.env.RACELENS_FIREBASE_ANDROID_SERVICES_FILE || './google-services.json'
  );
  const iosGoogleServicesFile = resolveServiceFile(
    configDir,
    process.env.RACELENS_FIREBASE_IOS_SERVICES_FILE || './GoogleService-Info.plist'
  );
  const firebasePlugins = firebaseEnabled
    ? [
        '@react-native-firebase/app',
        '@react-native-firebase/analytics',
        '@react-native-firebase/crashlytics'
      ]
      : [];
  const buildPropertiesPlugin = firebaseEnabled
    ? [
        'expo-build-properties',
        {
          ios: {
            forceStaticLinking: ['RNFBAnalytics', 'RNFBApp', 'RNFBCrashlytics'],
            useFrameworks: 'static'
          }
        }
      ]
    : 'expo-build-properties';
  const rewardedAdsEnabled = enabled(process.env.EXPO_PUBLIC_RACELENS_REWARDED_ADS);
  const rewardedAdTestMode = enabled(process.env.EXPO_PUBLIC_RACELENS_ADMOB_TEST_MODE);
  const rewardedAndroidAppId = (
    process.env.EXPO_PUBLIC_RACELENS_ADMOB_ANDROID_APP_ID ||
    (rewardedAdTestMode ? 'ca-app-pub-3940256099942544~3347511713' : '')
  ).trim();
  const rewardedAdUnitId = (process.env.EXPO_PUBLIC_RACELENS_ADMOB_REWARDED_AD_UNIT_ID || '').trim();
  if (process.env.EAS_BUILD_PROFILE === 'production') {
    if (!rewardedAdsEnabled || !/^ca-app-pub-\d+~\d+$/.test(rewardedAndroidAppId) || !/^ca-app-pub-\d+\/\d+$/.test(rewardedAdUnitId)) {
      throw new Error('Production build requires rewarded ads plus real AdMob Android app and rewarded unit IDs.');
    }
    if (rewardedAdTestMode || rewardedAndroidAppId === 'ca-app-pub-3940256099942544~3347511713' || rewardedAdUnitId === 'ca-app-pub-3940256099942544/5224354917') {
      throw new Error('Production build must not use AdMob test mode or Google test IDs.');
    }
  }
  const rewardedAdPlugins = rewardedAdsEnabled && rewardedAndroidAppId
    ? [['react-native-google-mobile-ads', { androidAppId: rewardedAndroidAppId }]]
    : [];

  return {
    ...config,
    android: {
      ...config.android,
      ...(firebaseEnabled && androidGoogleServicesFile ? { googleServicesFile: androidGoogleServicesFile } : {})
    },
    ios: {
      ...config.ios,
      ...(firebaseEnabled && iosGoogleServicesFile ? { googleServicesFile: iosGoogleServicesFile } : {})
    },
    extra: {
      ...config.extra,
      apiBaseUrl: process.env.EXPO_PUBLIC_RACELENS_API_BASE_URL || config.extra?.apiBaseUrl || '',
      analyticsUrl: process.env.EXPO_PUBLIC_RACELENS_ANALYTICS_URL || config.extra?.analyticsUrl || '',
      accountDeletionUrl: process.env.EXPO_PUBLIC_RACELENS_ACCOUNT_DELETION_URL || process.env.RACELENS_ACCOUNT_DELETION_URL || config.extra?.accountDeletionUrl || '',
      billingMode: process.env.EXPO_PUBLIC_RACELENS_BILLING_MODE || process.env.RACELENS_BILLING_MODE || config.extra?.billingMode || '',
      firebaseAuthEnabled: process.env.EXPO_PUBLIC_RACELENS_FIREBASE_AUTH_ENABLED || config.extra?.firebaseAuthEnabled || '',
      firebaseEnabled: process.env.EXPO_PUBLIC_RACELENS_FIREBASE_ENABLED || config.extra?.firebaseEnabled || '',
      offlineExampleEnabled: process.env.EXPO_PUBLIC_RACELENS_OFFLINE_EXAMPLE || config.extra?.offlineExampleEnabled || '',
      privacyPolicyUrl: process.env.EXPO_PUBLIC_RACELENS_PRIVACY_URL || process.env.RACELENS_PRIVACY_URL || config.extra?.privacyPolicyUrl || '',
      rewardedAds: process.env.EXPO_PUBLIC_RACELENS_REWARDED_ADS || config.extra?.rewardedAds || '',
      rewardedAdsPreview: process.env.EXPO_PUBLIC_RACELENS_REWARDED_ADS_PREVIEW || config.extra?.rewardedAdsPreview || '',
      rewardedAdTestMode: process.env.EXPO_PUBLIC_RACELENS_ADMOB_TEST_MODE || config.extra?.rewardedAdTestMode || '',
      rewardedAdUnitId: process.env.EXPO_PUBLIC_RACELENS_ADMOB_REWARDED_AD_UNIT_ID || config.extra?.rewardedAdUnitId || '',
      supportEmail: process.env.EXPO_PUBLIC_RACELENS_SUPPORT_EMAIL || process.env.RACELENS_SUPPORT_EMAIL || config.extra?.supportEmail || '',
      supportUrl: process.env.EXPO_PUBLIC_RACELENS_SUPPORT_URL || process.env.RACELENS_SUPPORT_URL || config.extra?.supportUrl || '',
      termsUrl: process.env.EXPO_PUBLIC_RACELENS_TERMS_URL || process.env.RACELENS_TERMS_URL || config.extra?.termsUrl || ''
    },
    plugins: [
      ...(config.plugins || []),
      'expo-secure-store',
      buildPropertiesPlugin,
      ...rewardedAdPlugins,
      ...firebasePlugins
    ]
  };
};
