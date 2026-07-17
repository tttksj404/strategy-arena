export type RewardedAdResult =
  | { readonly status: 'earned' }
  | { readonly status: 'dismissed' }
  | { readonly status: 'unavailable'; readonly message: string };
