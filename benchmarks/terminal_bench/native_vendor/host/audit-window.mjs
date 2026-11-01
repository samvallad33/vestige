import { createHash } from 'node:crypto';
const sorted = v => Array.isArray(v) ? v.map(sorted) : v && typeof v === 'object'
  ? Object.fromEntries(Object.keys(v).sort().map(k => [k, sorted(v[k])])) : v;
export function nativeDigest(row) {
  const body = { ...row }; delete body.digest;
  return createHash('sha256').update(JSON.stringify(sorted(body))).digest('hex');
}
const reject = message => { throw new Error('FAIL_CLOSED_INTERDICTION: ' + message); };
export function verifyWindow(before, after, expected, checkpoint) {
  if (before.chain?.status !== 'ok' && before.chain?.status !== 'empty') {
    if (before.chain?.status !== 'broken' || !checkpoint?.ownerApprovalMemoryId || !checkpoint?.memoryId
      || !checkpoint?.receiptId || checkpoint.chainValid !== true
      || checkpoint.historicalBreaks !== before.chain?.breaks)
      reject('historical audit gaps need an explicitly approved, verified native checkpoint');
  }
  if (!Number.isInteger(before.chain?.breaks) || after.chain?.breaks !== before.chain.breaks
    || (after.chain?.status !== before.chain.status && !(before.chain.status === 'empty' && after.chain.status === 'ok')))
    reject('new native ledger integrity break');
  const advance = after.chain.linesChecked - before.chain.linesChecked;
  if (!Number.isInteger(before.chain.linesChecked) || !Number.isInteger(advance) || advance <= 0)
    reject('no fresh native ledger record');
  const tip = before.recent?.at(-1)?.digest || '0'.repeat(64), rows = after.recent || [];
  let start = rows.findIndex(r => r.digest === tip);
  if (start >= 0) start++;
  else if (rows[0]?.prev === tip) start = 0;
  else reject('new ledger window does not join the witnessed tip');
  const fresh = rows.slice(start);
  if (fresh.length !== advance) reject('incomplete fresh native ledger window');
  let previous = tip, found;
  for (const row of fresh) {
    if (row.prev !== previous || nativeDigest(row) !== row.digest) reject('new ledger link or digest mismatch');
    previous = row.digest;
    if (row.event === expected.event && row.session === expected.session
      && (row.call || '') === (expected.call || '') && (row.agent || '') === (expected.agent || '')) found = row;
  }
  if (!found) reject('missing matching current hook acknowledgment');
  return { row: found, tip: previous, historicalBreaks: before.chain.breaks,
    historyClaim: before.chain.status === 'broken' ? 'historical gaps preserved; current window verified' : 'verified' };
}
