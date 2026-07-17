import { createClientDeviceId } from './deviceIdentity.shared';

const fallbackDeviceId = createClientDeviceId();

export async function getClientDeviceId(): Promise<string> {
  return fallbackDeviceId;
}
