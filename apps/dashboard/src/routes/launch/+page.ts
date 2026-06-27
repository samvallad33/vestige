// The launch page prerenders the visible overlay, wordmark bridge, and inert
// canvas elements as HTML. The raw WebGPU/canvas engine still boots only in
// onMount, so browser GPU APIs never execute during prerender.
export const ssr = true;
export const prerender = true;
