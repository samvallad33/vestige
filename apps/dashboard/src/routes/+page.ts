// Standalone promo deploy: when VITE_ROOT_REDIRECT=/launch, the root route
// prerenders the launch page too. In the normal dashboard build this route stays
// tiny and redirects to /graph on mount.
export const ssr = true;
export const prerender = true;
