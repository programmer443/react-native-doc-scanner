import { useEffect } from 'react';
import { useCameraPermission } from 'react-native-vision-camera';

/** Thin wrapper that requests camera permission as soon as the scanner mounts. */
export function useScannerPermissions() {
  const { hasPermission, requestPermission } = useCameraPermission();

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  return { hasPermission, requestPermission };
}
