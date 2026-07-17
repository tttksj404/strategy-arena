import {
  createClientDeviceId,
  installIdStorageKey,
  isClientDeviceId
} from './deviceIdentity.shared';

let pendingDeviceId: Promise<string> | undefined;

function ownerProDeviceId(): string | null {
  if (typeof globalThis.location === 'undefined') return null;
  const deviceId = new URLSearchParams(globalThis.location.hash.slice(1)).get('pro');
  return isClientDeviceId(deviceId) ? deviceId : null;
}

function clearOwnerProFragment(): void {
  if (typeof globalThis.location === 'undefined' || typeof globalThis.history === 'undefined') return;
  globalThis.history.replaceState(null, '', `${globalThis.location.pathname}${globalThis.location.search}`);
}

async function loadClientDeviceId(): Promise<string> {
  const fallbackDeviceId = createClientDeviceId();
  const linkedOwnerDeviceId = ownerProDeviceId();
  if (linkedOwnerDeviceId) {
    try {
      globalThis.localStorage?.setItem(installIdStorageKey, linkedOwnerDeviceId);
    } finally {
      clearOwnerProFragment();
    }
    return linkedOwnerDeviceId;
  }

  try {
    const storedDeviceId = globalThis.localStorage?.getItem(installIdStorageKey) ?? null;
    if (isClientDeviceId(storedDeviceId)) return storedDeviceId;
    globalThis.localStorage?.setItem(installIdStorageKey, fallbackDeviceId);
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
