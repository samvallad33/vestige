// fieldEvents — a tiny pub/sub bridge so cognitive events (a firewall
// quarantine, a memory decaying, a new memory landing) can drive the WebGPU
// backdrop's reactive effects. The BackdropEngine subscribes; the dashboard
// (or the launch-footage demo) publishes. Decoupled: the engine never imports
// dashboard state, the dashboard never imports the engine.

/** The reactive effects the field can play. Each maps to a shader impulse. */
export type FieldEventKind =
	| 'firewall' // a poisoned memory was quarantined — violet arc + crimson pulse
	| 'decay' // a memory decayed / was suppressed — a cold plume fades out
	| 'birth'; // a new memory landed — a bright synapse bloom

export type FieldEvent = {
	kind: FieldEventKind;
	/** normalized origin in [0,1]x[0,1] (defaults to center) for scoped effects */
	x?: number;
	y?: number;
	/** 0..1 strength multiplier (defaults to 1) */
	intensity?: number;
};

type Listener = (e: FieldEvent) => void;

const listeners = new Set<Listener>();

/** Fire a field effect. Safe to call from anywhere, any time. */
export function emitFieldEvent(e: FieldEvent): void {
	for (const l of listeners) {
		try {
			l(e);
		} catch {
			// a broken listener must never break the publisher
		}
	}
}

/** Subscribe (the BackdropEngine). Returns an unsubscribe fn. */
export function onFieldEvent(l: Listener): () => void {
	listeners.add(l);
	return () => listeners.delete(l);
}

/** Stable numeric code for the shader uniform (0 = none). */
export function fieldEventCode(kind: FieldEventKind): number {
	switch (kind) {
		case 'firewall':
			return 1;
		case 'decay':
			return 2;
		case 'birth':
			return 3;
		default:
			return 0;
	}
}
