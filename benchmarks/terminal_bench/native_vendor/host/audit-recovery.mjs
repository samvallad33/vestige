import { readFileSync } from 'node:fs';
import { fail } from './runtime.mjs';
export const OWNER_APPROVAL = 'mem-00000000000af67b';
export const CHECKPOINT_PREFIX = 'OpenCode preserved-history Hook Ledger checkpoint\n';
export function nativeBody(result) {
  if (result?.isError) throw fail('native checkpoint operation refused');
  const body = result?.structuredContent ?? JSON.parse(result?.content?.find(x => x.type === 'text')?.text);
  if (!body || typeof body !== 'object' || body.error) throw fail('invalid native checkpoint response');
  return body;
}
export function verifiedReceipt(result, id) {
  const body = nativeBody(result), v = body.attestation?.verification;
  if (v?.chainValid !== true || v.gateAllowed !== true || v.locallyVerified !== true
    || !body.receipt?.receipt_id || !body.receipt.retrieved?.includes(id))
    throw fail('native Strata checkpoint receipt did not verify');
  return body;
}
export async function loadRecovery(native, path, ledger) {
  let checkpoint;
  try { checkpoint = JSON.parse(readFileSync(path, 'utf8')); }
  catch { throw fail('historical audit gaps require the approved native recovery checkpoint'); }
  if (checkpoint.ownerApprovalMemoryId !== OWNER_APPROVAL || checkpoint.historicalBreaks !== 2
    || checkpoint.historicalBreaks !== ledger.chain?.breaks || checkpoint.ledger !== ledger.ledger
    || !Number.isInteger(checkpoint.linesChecked) || checkpoint.linesChecked > ledger.chain?.linesChecked
    || !/^[a-f0-9]{64}$/.test(checkpoint.tip || '') || !checkpoint.memoryId || !checkpoint.receiptId)
    throw fail('recovery checkpoint baseline mismatch');
  const memory = nativeBody(await native.rpc('tools/call', { name: 'memory',
    arguments: { action: 'get', id: checkpoint.memoryId } }));
  if (memory.node?.id !== checkpoint.memoryId || memory.node.content !== checkpoint.content
    || checkpoint.content !== CHECKPOINT_PREFIX + JSON.stringify(checkpoint.baseline))
    throw fail('native recovery checkpoint content mismatch');
  const b = checkpoint.baseline;
  if (b.ownerApprovalMemoryId !== OWNER_APPROVAL || b.ledger !== checkpoint.ledger
    || b.tip !== checkpoint.tip || b.linesChecked !== checkpoint.linesChecked
    || b.historicalBreaks !== checkpoint.historicalBreaks || b.historicalGapsResolved !== false
    || b.policy !== 'verify-every-new-event; halt-any-new-gap; never-auto-rebaseline')
    throw fail('native recovery checkpoint policy mismatch');
  const receipt = verifiedReceipt(await native.rpc('tools/call', { name: 'receipt',
    arguments: { action: 'get', receipt_id: checkpoint.memoryId } }), checkpoint.memoryId);
  // Reads update salience, so latest verified effect can differ from admission.
  return { ...checkpoint, chainValid: true, verificationReceiptId: receipt.receipt.receipt_id };
}
