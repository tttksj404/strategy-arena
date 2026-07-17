import Svg, { Circle, Path } from 'react-native-svg';

import { palette } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';

type BrandMarkProps = {
  readonly accessibilityLabel?: string;
  readonly mode: ThemeMode;
  readonly size?: number;
};

function polar(cx: number, cy: number, radius: number, degrees: number) {
  const radians = ((degrees - 90) * Math.PI) / 180;
  return {
    x: cx + radius * Math.cos(radians),
    y: cy + radius * Math.sin(radians)
  };
}

export function BrandMark({ accessibilityLabel, mode, size = 28 }: BrandMarkProps) {
  const colors = palette(mode);
  const start = polar(512, 512, 424, -50);
  const end = polar(512, 512, 424, 50);
  const arc = `M ${start.x} ${start.y} A 424 424 0 0 1 ${end.x} ${end.y}`;

  return (
    <Svg
      accessibilityLabel={accessibilityLabel}
      accessibilityRole={accessibilityLabel ? 'image' : undefined}
      accessible={Boolean(accessibilityLabel)}
      width={size}
      height={size}
      viewBox="0 0 1024 1024"
    >
      <Circle cx={512} cy={512} r={300} fill="none" stroke={colors.accentSignal} strokeWidth={88} />
      <Circle cx={512} cy={512} r={256} fill={colors.brandLensInner} />
      <Circle cx={512} cy={512} r={96} fill={colors.brandWhite} />
      <Circle cx={512} cy={512} r={40} fill={colors.brandBackground} />
      <Path d={arc} fill="none" stroke={colors.accentGold} strokeLinecap="round" strokeWidth={56} />
    </Svg>
  );
}
