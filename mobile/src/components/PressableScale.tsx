import { useRef } from 'react';
import { Animated, Pressable } from 'react-native';
import type { GestureResponderEvent, PressableProps, StyleProp, ViewStyle } from 'react-native';

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

type PressableScaleProps = Omit<PressableProps, 'style'> & {
  style?: StyleProp<ViewStyle>;
  scaleTo?: number;
};

/**
 * Shared press-feedback wrapper. Scales content to `scaleTo` (default 0.985 per
 * DESIGN.md 5. LensCard motion spec) on press-in and restores it on press-out.
 * Only opacity/transform are touched, so it never affects layout or QA-critical
 * final positions/sizes (DESIGN.md 6. Motion rules).
 */
export function PressableScale({
  style,
  scaleTo = 0.985,
  onPressIn,
  onPressOut,
  ...rest
}: PressableScaleProps) {
  const scale = useRef(new Animated.Value(1)).current;

  function handlePressIn(event: GestureResponderEvent) {
    Animated.timing(scale, {
      duration: 120,
      toValue: scaleTo,
      useNativeDriver: true
    }).start();
    onPressIn?.(event);
  }

  function handlePressOut(event: GestureResponderEvent) {
    Animated.timing(scale, {
      duration: 180,
      toValue: 1,
      useNativeDriver: true
    }).start();
    onPressOut?.(event);
  }

  return (
    <AnimatedPressable
      {...rest}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      style={[style, { transform: [{ scale }] }]}
    />
  );
}
