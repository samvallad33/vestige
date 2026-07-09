// The Observatory is a WebGPU client-only surface: it reads
// window.location for its ?demo= contract and boots a GPU device on mount.
// SSR would 500 on `window` and can never render the field anyway.
export const ssr = false;
export const prerender = false;
