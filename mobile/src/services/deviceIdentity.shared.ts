export const installIdStorageKey = 'racelens.install-id.v1';

export function createClientDeviceId(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return `dev_${globalThis.crypto.randomUUID()}`;
  }
  return `dev_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 14)}`;
}

export function isClientDeviceId(value: string | null): value is string {
  return typeof value === 'string' && /^dev_[A-Za-z0-9_-]{12,96}$/.test(value);
}
