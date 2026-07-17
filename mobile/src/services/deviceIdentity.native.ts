import * as SecureStore from 'expo-secure-store';

import {
  createClientDeviceId,
  installIdStorageKey,
  isClientDeviceId
} from './deviceIdentity.shared';

let pendingDeviceId: Promise<string> | undefined;

async function loadClientDeviceId(): Promise<string> {
  const fallbackDeviceId = createClientDeviceId();
  try {
    const storedDeviceId = await SecureStore.getItemAsync(installIdStorageKey);
    if (isClientDeviceId(storedDeviceId)) return storedDeviceId;
    await SecureStore.setItemAsync(installIdStorageKey, fallbackDeviceId);
    return fallbackDeviceId;
  } catch (error) {
    if (error instanceof Error) return fallbackDeviceId;
    throw error;
  }
}

export function getClientDeviceId(): Promise<string> {
  pendingDeviceId ??= loadClientDeviceId();
  return pendingDeviceId;
}
