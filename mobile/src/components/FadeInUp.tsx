import type { ReactNode } from 'react';
import { useEffect, useRef } from 'react';
import { Animated, Easing } from 'react-native';
import type { StyleProp, ViewStyle } from 'react-native';

type FadeInUpProps = {
  children: ReactNode;
  style?: StyleProp<ViewStyle>;
  distance?: number;
  duration?: number;
};

/**
 * Mount-only reveal used for primary analysis-result cards (DESIGN.md 5.
 * LensCard motion: "fade-up on mount"). Opacity 0->1 and translateY
 * distance->0, opacity/transform only so it never changes layout or the
 * final resting position/size QA scripts check for.
 */
export function FadeInUp({ children, style, distance = 8, duration = 200 }: FadeInUpProps) {
  const progress = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const animation = Animated.timing(progress, {
      duration,
      easing: Easing.out(Easing.cubic),
      toValue: 1,
      useNativeDriver: true
    });
    animation.start();
    return () => animation.stop();
    // Runs once for this mounted instance; new results get a fresh mount
    // because AnalyzeScreen only exists while its tab is active.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Animated.View
      style={[
        style,
        {
          opacity: progress,
          transform: [
            {
              translateY: progress.interpolate({
                inputRange: [0, 1],
                outputRange: [distance, 0]
              })
            }
          ]
        }
      ]}
    >
      {children}
    </Animated.View>
  );
}
