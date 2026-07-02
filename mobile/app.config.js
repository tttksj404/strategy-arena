module.exports = ({ config }) => ({
  ...config,
  extra: {
    ...config.extra,
    apiBaseUrl: process.env.EXPO_PUBLIC_RACELENS_API_BASE_URL || config.extra?.apiBaseUrl || ''
  }
});
