// The launch landing page is a client-only WebGL experience (live 3D memory
// graph + Memory Cinema). Disable SSR so Three.js never runs during prerender;
// the page is still prerendered as a static shell that hydrates on the client.
export const ssr = false;
export const prerender = true;
